"""
AI Chat Platform Backend — FastAPI application.

Phase 1: Proxy mode + Phase 2: Multi-provider support.
- Manages llama-server as a subprocess
- Routes /v1/chat/completions through provider registry
- Serves the web UI as static files

Run: python -m webui.backend.app
"""

import asyncio
import json
import signal
import sys
import webbrowser
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .providers import (
    ProviderRegistry,
    OpenAICompatibleProvider,
    AnthropicProvider,
    ensure_default_config,
    _PROVIDER_TYPES,
    _build_provider,
)
from .tools import ToolRegistry, register_builtin_tools, chat_with_tools
from .mcp_client import MCPManager, MCPServerConfig

# ── State ─────────────────────────────────────────────────────────────────

llama_process = None
llama_base_url = f"http://{config.HOST}:{config.LLAMA_SERVER_PORT}"

registry = ProviderRegistry()
tool_registry = ToolRegistry()
mcp_manager = MCPManager()


# ── llama-server management ───────────────────────────────────────────────

async def start_llama_server():
    """Start llama-server as a managed subprocess."""
    global llama_process

    cmd = [
        str(config.LLAMA_SERVER),
        "-m", config.MODEL_PATH,
        "-ngl", str(config.GPU_LAYERS),
        "-c", str(config.CONTEXT_SIZE),
        "-t", str(config.THREADS),
        "--host", config.HOST,
        "--port", str(config.LLAMA_SERVER_PORT),
        "-rea", "off",
        "--jinja",
        "-ctk", config.KV_CACHE_TYPE_K,
        "-ctv", config.KV_CACHE_TYPE_V,
    ]

    print(f"Starting llama-server...")
    print(f"  Model  : {config.MODEL_PATH}")
    print(f"  Context: {config.CONTEXT_SIZE}")
    print(f"  GPU    : {config.GPU_LAYERS} layers")
    print(f"  KV type: K={config.KV_CACHE_TYPE_K}, V={config.KV_CACHE_TYPE_V}")
    print(f"  Port   : {config.LLAMA_SERVER_PORT} (internal)")

    llama_process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    # Wait for health check
    async with httpx.AsyncClient() as client:
        for i in range(120):  # 120 seconds max
            try:
                r = await client.get(f"{llama_base_url}/health", timeout=2.0)
                if r.status_code == 200:
                    print(f"  llama-server ready!")
                    return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            await asyncio.sleep(1.0)

    print("ERROR: llama-server failed to start within 120s")
    if llama_process:
        llama_process.kill()
    sys.exit(1)


async def stop_llama_server():
    """Stop the managed llama-server subprocess."""
    global llama_process
    if llama_process and llama_process.returncode is None:
        print("Stopping llama-server...")
        llama_process.terminate()
        try:
            await asyncio.wait_for(llama_process.wait(), timeout=10)
        except asyncio.TimeoutError:
            llama_process.kill()
        llama_process = None


# ── FastAPI lifecycle ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: launch llama-server, load providers. Shutdown: kill server."""
    await start_llama_server()

    # Ensure default providers.json exists
    ensure_default_config(config.PROVIDERS_FILE, llama_base_url)

    # Load providers from file
    registry.load_from_file(config.PROVIDERS_FILE)

    # Ensure "local" provider always exists pointing at llama-server
    if registry.get("local") is None:
        local_cfg = {
            "type": "openai_compatible",
            "name": "Local (llama-server)",
            "base_url": f"{llama_base_url}/v1",
            "api_key": "",
            "default_model": "qwen3.5-9b",
        }
        registry.add(
            "local",
            OpenAICompatibleProvider(
                name=local_cfg["name"],
                base_url=local_cfg["base_url"],
                default_model=local_cfg["default_model"],
            ),
            raw_config=local_cfg,
        )
        registry.save_to_file(config.PROVIDERS_FILE)

    print(f"  Providers loaded: {', '.join(registry.providers.keys())}")

    # Register built-in tools
    register_builtin_tools(tool_registry)
    print(f"  Tools loaded: {', '.join(t['name'] for t in tool_registry.list_tools())}")

    # Load and connect MCP servers
    mcp_configs = mcp_manager.load_from_file(config.MCP_SERVERS_FILE)
    mcp_tool_count = 0
    for mc in mcp_configs:
        if mc.auto_start:
            try:
                await mcp_manager.add_server(mc)
            except Exception as e:
                print(f"  WARNING: MCP server '{mc.name}' failed to start: {e}")
        else:
            mcp_manager.configs[mc.name] = mc  # track but don't connect
    mcp_tool_count = mcp_manager.bridge_to_registry(tool_registry)
    if mcp_tool_count:
        print(f"  MCP tools bridged: {mcp_tool_count}")

    yield

    await mcp_manager.shutdown()
    await stop_llama_server()


app = FastAPI(title="AI Chat Platform", lifespan=lifespan)


# ── Context eviction (application-level KV compression) ──────────────

# With q4_0 KV (4x) + context eviction (3x) = ~12x effective compression
# This keeps conversations flowing beyond what raw context would allow
EVICTION_TARGET_TOKENS = 40000  # Start compressing when messages exceed this
KEEP_RECENT_MESSAGES = 6        # Always keep last N message pairs intact
SUMMARY_INSTRUCTION = "Condense the following conversation history into a brief summary of key points, decisions, and context. Be concise but preserve important details:\n\n"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def evict_context(messages: list[dict]) -> list[dict]:
    """
    Application-level context eviction.

    When conversation history exceeds EVICTION_TARGET_TOKENS, compress older
    messages by keeping only a summary. This is the eviction half of 12x:
    - q4_0 KV quantization handles the storage compression (4x)
    - This function handles the token reduction (3x)
    - Combined: ~12x effective compression

    Strategy (mirrors our entropy-adaptive research):
    1. System message: always keep (like sink tokens)
    2. Recent messages: always keep last N (like recent tokens in eviction)
    3. Old messages: summarize into a condensed block (eviction of middle tokens)
    """
    total_tokens = sum(estimate_tokens(m.get("content", "") or "") for m in messages)

    if total_tokens <= EVICTION_TARGET_TOKENS:
        return messages  # No eviction needed

    # Separate system message, old messages, and recent messages
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= KEEP_RECENT_MESSAGES * 2:
        return messages  # Too few messages to compress

    # Split into old (to compress) and recent (to keep)
    keep_count = KEEP_RECENT_MESSAGES * 2  # pairs of user+assistant
    old_msgs = non_system[:-keep_count]
    recent_msgs = non_system[-keep_count:]

    # Build a summary of old messages
    old_text_parts = []
    for m in old_msgs:
        role = m.get("role", "unknown")
        content = m.get("content", "") or ""
        if content:
            # Truncate very long individual messages
            if len(content) > 500:
                content = content[:500] + "..."
            old_text_parts.append(f"{role}: {content}")

    if not old_text_parts:
        return messages

    summary_text = "\n".join(old_text_parts)

    # Create a compressed system message with the summary
    compressed_history = (
        f"[Compressed conversation history - {len(old_msgs)} earlier messages summarized]\n"
        f"{summary_text[:3000]}"  # Cap at ~750 tokens
    )

    # Rebuild: system + compressed history + recent
    result = list(system_msgs)
    result.append({
        "role": "system",
        "content": compressed_history,
    })
    result.extend(recent_msgs)

    old_tokens = sum(estimate_tokens(m.get("content", "") or "") for m in old_msgs)
    new_tokens = estimate_tokens(compressed_history)
    print(f"  [eviction] Compressed {len(old_msgs)} old messages: "
          f"~{old_tokens} tokens -> ~{new_tokens} tokens "
          f"({old_tokens / max(new_tokens, 1):.1f}x reduction)")

    return result


# ── Chat completions (provider-routed) ────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Route chat completions through the provider registry."""
    body = await request.json()

    # Extract provider field (default to "local")
    provider_id = body.pop("provider", "local")
    provider = registry.get(provider_id)

    if provider is None:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Unknown provider: {provider_id}", "type": "invalid_request"}},
        )

    # Extract standard params
    messages = body.get("messages", [])
    model = body.get("model", "")
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 8192)
    tools = body.get("tools")

    # Apply context eviction: compress old messages to fit more in context
    messages = evict_context(messages)

    # Extra params
    extra = {}
    for key in ("top_p", "top_k", "repeat_penalty", "min_p"):
        if key in body:
            extra[key] = body[key]

    # Use tool-calling loop if tools are enabled AND provider supports it
    # Only the main local provider (with --jinja) reliably supports tool calling
    # Skip tools for older servers (bitnet) and external APIs without tool support
    supports_tools = provider_id == "local" or provider_id.startswith("openai") or provider_id.startswith("anthropic")
    if tool_registry.has_enabled_tools() and supports_tools:
        async def stream_with_tools():
            async for chunk in chat_with_tools(
                provider=provider,
                messages=messages,
                registry=tool_registry,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                **extra,
            ):
                yield chunk

        return StreamingResponse(
            stream_with_tools(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def stream_provider():
        async for chunk in provider.chat_stream(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **extra,
        ):
            yield chunk

    return StreamingResponse(
        stream_provider(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Provider management API ──────────────────────────────────────────────

@app.get("/api/providers")
async def list_providers():
    """List all configured providers (API keys masked)."""
    return registry.list_providers()


@app.post("/api/providers/{provider_id}")
async def add_or_update_provider(provider_id: str, request: Request):
    """Add or update a provider configuration."""
    cfg = await request.json()

    ptype = cfg.get("type", "openai_compatible")
    cls = _PROVIDER_TYPES.get(ptype)
    if cls is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown provider type: {ptype}"},
        )

    provider = _build_provider(cls, cfg)
    registry.add(provider_id, provider, raw_config=cfg)
    registry.save_to_file(config.PROVIDERS_FILE)

    return {"status": "ok", "id": provider_id}


@app.delete("/api/providers/{provider_id}")
async def delete_provider(provider_id: str):
    """Delete a provider. The 'local' provider cannot be deleted."""
    if provider_id == "local":
        return JSONResponse(
            status_code=400,
            content={"error": "Cannot delete the 'local' provider."},
        )
    if registry.get(provider_id) is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Provider '{provider_id}' not found."},
        )
    registry.remove(provider_id)
    registry.save_to_file(config.PROVIDERS_FILE)
    return {"status": "ok"}


@app.post("/api/providers/{provider_id}/test")
async def test_provider(provider_id: str):
    """Test a provider by sending a tiny request."""
    provider = registry.get(provider_id)
    if provider is None:
        return JSONResponse(status_code=404, content={"error": f"Provider '{provider_id}' not found."})
    try:
        # Collect a few tokens to verify it works
        text = ""
        async for chunk in provider.chat_stream(
            messages=[{"role": "user", "content": "Say OK"}],
            model="", temperature=0.0, max_tokens=5,
        ):
            text += chunk.decode("utf-8", errors="ignore")
        return {"status": "ok", "preview": text[:200]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Tool management API ──────────────────────────────────────────────────

@app.get("/api/tools")
async def list_tools():
    """List all tools with enabled status."""
    return tool_registry.list_tools()


@app.post("/api/tools/{name}/toggle")
async def toggle_tool(name: str):
    """Toggle a tool's enabled/disabled status."""
    tool = tool_registry.get(name)
    if tool is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Tool '{name}' not found."},
        )
    tool_registry.set_enabled(name, not tool.enabled)
    return {"status": "ok", "name": name, "enabled": tool.enabled}


@app.post("/api/tools/{name}/test")
async def test_tool(name: str, request: Request):
    """Test a tool with provided arguments."""
    tool = tool_registry.get(name)
    if tool is None:
        return JSONResponse(status_code=404, content={"error": f"Tool '{name}' not found."})
    try:
        body = await request.json()
        args = body.get("arguments", {})
        result = await tool.callable(**args)
        return {"status": "ok", "result": result[:5000]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── MCP server management API ───────────────────────────────────────────

@app.get("/api/mcp/servers")
async def list_mcp_servers():
    """List all MCP servers with connection status."""
    return mcp_manager.list_servers()


@app.post("/api/mcp/servers")
async def add_mcp_server(request: Request):
    """Add a new MCP server, connect to it, and bridge its tools."""
    body = await request.json()

    name = body.get("name")
    command = body.get("command")
    if not name or not command:
        return JSONResponse(
            status_code=400,
            content={"error": "Both 'name' and 'command' are required."},
        )

    cfg = MCPServerConfig(
        name=name,
        command=command,
        args=body.get("args", []),
        env=body.get("env", {}),
        auto_start=body.get("auto_start", True),
    )

    try:
        tool_names = await mcp_manager.add_server(cfg)
        bridged = mcp_manager.bridge_to_registry(tool_registry)
        mcp_manager.save_to_file(config.MCP_SERVERS_FILE)
        return {
            "status": "ok",
            "name": name,
            "tools": tool_names,
            "bridged": bridged,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to connect MCP server '{name}': {e}"},
        )


@app.delete("/api/mcp/servers/{name}")
async def remove_mcp_server(name: str):
    """Disconnect, unbridge, and remove an MCP server."""
    if name not in mcp_manager.configs:
        return JSONResponse(
            status_code=404,
            content={"error": f"MCP server '{name}' not found."},
        )

    mcp_manager.unbridge_from_registry(tool_registry, name)
    await mcp_manager.remove_server(name)
    mcp_manager.save_to_file(config.MCP_SERVERS_FILE)
    return {"status": "ok"}


# ── Proxy: /health ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Proxy health check to llama-server."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{llama_base_url}/health")
            return Response(content=r.content, status_code=r.status_code)
        except httpx.ConnectError:
            return Response(content='{"status": "error"}', status_code=503)


# ── Static files: serve the web UI ────────────────────────────────────────

@app.get("/")
async def serve_index():
    """Serve the main chat UI."""
    return FileResponse(config.WEBUI_DIR / "index.html")


# Mount static files for CSS, JS, etc. (after explicit routes)
app.mount("/", StaticFiles(directory=str(config.WEBUI_DIR)), name="static")


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("AI Chat Platform")
    print("=" * 60)
    print(f"  Backend: http://{config.HOST}:{config.BACKEND_PORT}")
    print(f"  llama  : http://{config.HOST}:{config.LLAMA_SERVER_PORT} (internal)")
    print()

    # Open browser after a short delay
    import threading
    def open_browser():
        import time
        time.sleep(3)
        webbrowser.open(f"http://{config.HOST}:{config.BACKEND_PORT}")
    threading.Thread(target=open_browser, daemon=True).start()

    # Run uvicorn
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.BACKEND_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
