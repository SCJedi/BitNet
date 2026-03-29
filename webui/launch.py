#!/usr/bin/env python3
"""
Launcher for Qwen3.5-9B Chat UI.
Starts llama-server with the correct model and opens the browser.
"""

import subprocess
import sys
import time
import signal
import webbrowser
import urllib.request
import urllib.error
import os

# ── Configuration ──────────────────────────────────────────────────────────
LLAMA_SERVER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "llama-cpp-latest", "bin", "llama-server.exe",
)
MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "qwen3.5-9b", "Qwen3.5-9B-Q4_K_M.gguf",
)
WEBUI_DIR = os.path.dirname(os.path.abspath(__file__))
HOST = "127.0.0.1"
PORT = 8080
GPU_LAYERS = 99
CONTEXT = 8192
THREADS = 4
# ───────────────────────────────────────────────────────────────────────────

URL = f"http://{HOST}:{PORT}"
process = None


def start_server():
    """Launch llama-server as a subprocess."""
    cmd = [
        LLAMA_SERVER,
        "-m", MODEL,
        "-ngl", str(GPU_LAYERS),
        "-c", str(CONTEXT),
        "-t", str(THREADS),
        "--host", HOST,
        "--port", str(PORT),
        "--path", WEBUI_DIR,
    ]
    print(f"Starting llama-server...")
    print(f"  Model : {os.path.basename(MODEL)}")
    print(f"  URL   : {URL}")
    print(f"  GPU   : {GPU_LAYERS} layers")
    print(f"  Ctx   : {CONTEXT} tokens")
    print()
    return subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )


def wait_for_server(timeout=120):
    """Poll /health until the server is ready."""
    health_url = f"{URL}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
        elapsed = int(time.time() - start)
        if elapsed % 10 == 0 and elapsed > 0:
            print(f"  Waiting for server... ({elapsed}s)")
    return False


def shutdown(signum=None, frame=None):
    """Gracefully kill the server."""
    global process
    if process and process.poll() is None:
        print("\nShutting down llama-server...")
        if sys.platform == "win32":
            process.terminate()
        else:
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    print("Done.")
    sys.exit(0)


def main():
    global process

    # Check prerequisites
    if not os.path.isfile(LLAMA_SERVER):
        print(f"ERROR: llama-server not found at:\n  {LLAMA_SERVER}")
        sys.exit(1)
    if not os.path.isfile(MODEL):
        print(f"ERROR: Model not found at:\n  {MODEL}")
        sys.exit(1)
    if not os.path.isfile(os.path.join(WEBUI_DIR, "index.html")):
        print(f"ERROR: index.html not found in:\n  {WEBUI_DIR}")
        sys.exit(1)

    # Register signal handlers
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, shutdown)

    # Start server
    process = start_server()

    # Wait for ready
    print("Waiting for model to load...")
    if not wait_for_server():
        print("ERROR: Server did not become ready within 120 seconds.")
        shutdown()

    print(f"\nServer ready! Opening browser...")
    webbrowser.open(URL)

    print(f"Chat UI running at {URL}")
    print("Press Ctrl+C to stop.\n")

    # Block until server exits or user interrupts
    try:
        process.wait()
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
