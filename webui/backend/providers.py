"""
Multi-provider support for the AI Chat Platform.

Provides a uniform async streaming interface across:
- OpenAI-compatible endpoints (local llama-server, OpenAI API, Ollama, etc.)
- Anthropic API (translated to OpenAI SSE format)
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Protocol, runtime_checkable

import httpx


# ── Protocol ─────────────────────────────────────────────────────────────

@runtime_checkable
class Provider(Protocol):
    name: str
    provider_type: str

    async def chat_stream(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[bytes]:
        """Yields raw SSE bytes in ``data: {...}\n\n`` format."""
        ...


# ── OpenAI-compatible provider ───────────────────────────────────────────

class OpenAICompatibleProvider:
    """Covers: local llama-server, OpenAI API, Ollama, any OpenAI-compatible endpoint."""

    provider_type = "openai_compatible"

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str = "",
        default_model: str = "",
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = _resolve_env(api_key)
        self.default_model = default_model

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def chat_stream(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[bytes]:
        body: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for key in ("top_p", "top_k", "repeat_penalty", "min_p"):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        if tools:
            body["tools"] = tools

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0)
        ) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=body,
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk


# ── Anthropic provider ───────────────────────────────────────────────────

class AnthropicProvider:
    """Uses the Anthropic API, translating streaming events to OpenAI SSE format."""

    provider_type = "anthropic"

    def __init__(
        self,
        name: str,
        api_key: str = "",
        default_model: str = "claude-sonnet-4-20250514",
    ):
        self.name = name
        self.api_key = _resolve_env(api_key)
        self.default_model = default_model

    async def chat_stream(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[bytes]:
        try:
            import anthropic
        except ImportError:
            error_chunk = _make_sse_error(
                "Anthropic provider requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            )
            yield error_chunk
            return

        if not self.api_key:
            yield _make_sse_error("Anthropic API key not configured.")
            return

        # Convert OpenAI message format to Anthropic format
        system_text = ""
        anthro_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                anthro_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        chosen_model = model or self.default_model

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        create_kwargs: dict[str, Any] = {
            "model": chosen_model,
            "messages": anthro_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system_text.strip():
            create_kwargs["system"] = system_text.strip()

        # Generate a fake completion id
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        async with client.messages.stream(**{k: v for k, v in create_kwargs.items() if k != "stream"}) as stream:
            async for event in stream:
                sse_data = _anthropic_event_to_openai(
                    event, completion_id, created, chosen_model
                )
                if sse_data is not None:
                    yield f"data: {json.dumps(sse_data)}\n\n".encode()

        # Send the final [DONE]
        yield b"data: [DONE]\n\n"


def _anthropic_event_to_openai(
    event, completion_id: str, created: int, model: str
) -> dict | None:
    """Convert an Anthropic streaming event to an OpenAI-format SSE chunk dict."""
    import anthropic.types as at

    if isinstance(event, at.RawContentBlockDeltaEvent):
        delta = event.delta
        if hasattr(delta, "text"):
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": delta.text},
                    "finish_reason": None,
                }],
            }
    elif isinstance(event, at.RawMessageDeltaEvent):
        stop = getattr(event.delta, "stop_reason", None)
        if stop:
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
    return None


# ── Registry ─────────────────────────────────────────────────────────────

_PROVIDER_TYPES = {
    "openai_compatible": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
}


class ProviderRegistry:
    def __init__(self):
        self.providers: dict[str, Provider] = {}
        self._raw_configs: dict[str, dict] = {}

    def add(self, provider_id: str, provider: Provider, raw_config: dict | None = None):
        self.providers[provider_id] = provider
        if raw_config is not None:
            self._raw_configs[provider_id] = raw_config

    def get(self, provider_id: str) -> Provider | None:
        return self.providers.get(provider_id)

    def remove(self, provider_id: str):
        self.providers.pop(provider_id, None)
        self._raw_configs.pop(provider_id, None)

    def list_providers(self) -> list[dict]:
        """Return provider info list with masked API keys."""
        result = []
        for pid, prov in self.providers.items():
            info: dict[str, Any] = {
                "id": pid,
                "name": prov.name,
                "type": prov.provider_type,
            }
            if hasattr(prov, "default_model"):
                info["default_model"] = prov.default_model
            if hasattr(prov, "base_url"):
                info["base_url"] = prov.base_url
            # Never expose real API keys
            raw = self._raw_configs.get(pid, {})
            if raw.get("api_key"):
                info["api_key"] = "****"
            else:
                info["api_key"] = ""
            result.append(info)
        return result

    def load_from_file(self, path: Path):
        """Load providers from a JSON config file."""
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for pid, cfg in data.items():
            ptype = cfg.get("type", "openai_compatible")
            cls = _PROVIDER_TYPES.get(ptype)
            if cls is None:
                print(f"Warning: unknown provider type '{ptype}' for '{pid}', skipping.")
                continue
            provider = _build_provider(cls, cfg)
            self.add(pid, provider, raw_config=cfg)

    def save_to_file(self, path: Path):
        """Save current provider configs to JSON."""
        data: dict[str, dict] = {}
        for pid in self.providers:
            raw = self._raw_configs.get(pid)
            if raw:
                data[pid] = raw
            else:
                # Reconstruct from provider object
                prov = self.providers[pid]
                cfg: dict[str, Any] = {
                    "type": prov.provider_type,
                    "name": prov.name,
                }
                if hasattr(prov, "base_url"):
                    cfg["base_url"] = prov.base_url
                if hasattr(prov, "default_model"):
                    cfg["default_model"] = prov.default_model
                # Don't save resolved keys — save the original env var syntax
                cfg["api_key"] = ""
                data[pid] = cfg
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


# ── Helpers ──────────────────────────────────────────────────────────────

_ENV_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_env(value: str) -> str:
    """Resolve ``${ENV_VAR}`` references in a string."""
    if not value:
        return value
    def _replacer(m):
        return os.environ.get(m.group(1), "")
    return _ENV_RE.sub(_replacer, value)


def _build_provider(cls, cfg: dict) -> Provider:
    """Construct a provider from a config dict."""
    if cls is OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            name=cfg.get("name", ""),
            base_url=cfg.get("base_url", ""),
            api_key=cfg.get("api_key", ""),
            default_model=cfg.get("default_model", ""),
        )
    elif cls is AnthropicProvider:
        return AnthropicProvider(
            name=cfg.get("name", ""),
            api_key=cfg.get("api_key", ""),
            default_model=cfg.get("default_model", "claude-sonnet-4-20250514"),
        )
    raise ValueError(f"Unknown provider class: {cls}")


def _make_sse_error(message: str) -> bytes:
    """Create an SSE error chunk in OpenAI format."""
    data = {
        "error": {
            "message": message,
            "type": "provider_error",
        }
    }
    return f"data: {json.dumps(data)}\n\n".encode()


def ensure_default_config(path: Path, llama_base_url: str) -> None:
    """Create default providers.json if it doesn't exist."""
    if path.exists():
        return
    default = {
        "local": {
            "type": "openai_compatible",
            "name": "Local (llama-server)",
            "base_url": f"{llama_base_url}/v1",
            "api_key": "",
            "default_model": "qwen3.5-9b",
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(default, f, indent=4)
