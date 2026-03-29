"""
AI Chat Platform Backend — FastAPI application.

Phase 1: Proxy mode.
- Manages llama-server as a subprocess
- Proxies /v1/chat/completions with SSE streaming
- Serves the web UI as static files

Run: python -m webui.backend.app
"""

import asyncio
import signal
import sys
import webbrowser
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config

# ── State ─────────────────────────────────────────────────────────────────

llama_process = None
llama_base_url = f"http://{config.HOST}:{config.LLAMA_SERVER_PORT}"


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
    """Startup: launch llama-server. Shutdown: kill it."""
    await start_llama_server()
    yield
    await stop_llama_server()


app = FastAPI(title="AI Chat Platform", lifespan=lifespan)


# ── Proxy: /v1/chat/completions ───────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy chat completions to llama-server with SSE streaming."""
    body = await request.body()
    headers = {"Content-Type": "application/json"}

    async def stream_proxy():
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            async with client.stream(
                "POST",
                f"{llama_base_url}/v1/chat/completions",
                content=body,
                headers=headers,
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        stream_proxy(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
