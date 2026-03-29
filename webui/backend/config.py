"""Configuration for the AI chat platform backend."""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
WEBUI_DIR = BACKEND_DIR.parent
PROJECT_ROOT = WEBUI_DIR.parent

LLAMA_SERVER = PROJECT_ROOT / "tools" / "llama-cpp-latest" / "bin" / "llama-server.exe"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "qwen3.5-9b" / "Qwen3.5-9B-Q4_K_M.gguf"

# ── Config files (created on first run) ───────────────────────────────────
PROVIDERS_FILE = BACKEND_DIR / "providers.json"
MCP_SERVERS_FILE = BACKEND_DIR / "mcp_servers.json"

# ── Server settings ───────────────────────────────────────────────────────
HOST = os.environ.get("CHAT_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("CHAT_PORT", "8000"))
LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_PORT", "8080"))

# ── llama-server settings ─────────────────────────────────────────────────
GPU_LAYERS = int(os.environ.get("GPU_LAYERS", "99"))
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", "32768"))
THREADS = int(os.environ.get("THREADS", "4"))

# ── Model override ────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL))
