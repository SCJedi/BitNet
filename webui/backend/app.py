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

# ── State ─────────────────────────────────────────────────────────────────

llama_process = None
llama_base_url = f"http://{config.HOST}:{config.LLAMA_SERVER_PORT}"

registry = ProviderRegistry()


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
    ]

    print(f"Starting llama-server...")
    print(f"  Model  : {config.MODEL_PATH}")
    print(f"  Context: {config.CONTEXT_SIZE}")
    print(f"  GPU    : {config.GPU_LAYERS} layers")
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

    yield
    await stop_llama_server()


app = FastAPI(title="AI Chat Platform", lifespan=lifespan)


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

    # Extra params
    extra = {}
    for key in ("top_p", "top_k", "repeat_penalty", "min_p"):
        if key in body:
            extra[key] = body[key]

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
