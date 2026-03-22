"""Path configuration and environment variable handling.

Supports multiple models with auto-detection:
  - Qwen3.5-4B Q4_K_M (preferred, uses standalone llama.cpp with GPU)
  - BitNet-b1.58-2B-4T (fallback, uses BitNet llama-cli, CPU only)

Override with env vars:
  BITNET_MODEL   - path to GGUF model file
  BITNET_CLI     - path to llama-cli (or llama-completion) binary
  BITNET_THREADS - number of CPU threads (default: 4)
  BITNET_CTX_SIZE - context size (default: 2048)
"""

import os
import sys
from pathlib import Path


def _find_bitnet_root() -> Path:
    """Walk up from tools/ to find the BitNet root (contains CMakeLists.txt)."""
    current = Path(__file__).resolve().parent.parent  # tools/
    for _ in range(10):
        if (current / "CMakeLists.txt").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # Fallback: assume tools/ is directly under BitNet root
    return Path(__file__).resolve().parent.parent.parent


BITNET_ROOT = _find_bitnet_root()

# Model registry: (model_path, cli_path, use_gpu, is_chat_model)
_MODELS = [
    {
        "name": "Qwen3.5-4B-Q4_K_M",
        "model": BITNET_ROOT / "models" / "qwen3.5-4b" / "Qwen3.5-4B-Q4_K_M.gguf",
        "cli": BITNET_ROOT / "tools" / "llama-cpp-latest" / "bin" / "llama-cli.exe",
        "use_gpu": True,
        "is_chat_model": True,
    },
    {
        "name": "BitNet-b1.58-2B-4T",
        "model": BITNET_ROOT / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf",
        "cli": BITNET_ROOT / "build" / "bin" / "llama-cli.exe",
        "use_gpu": False,
        "is_chat_model": False,
    },
]


def _auto_detect_model() -> dict:
    """Find the best available model+cli pair."""
    for entry in _MODELS:
        if entry["model"].exists() and entry["cli"].exists():
            return entry
    return _MODELS[-1]  # fallback to BitNet even if missing (validate() will catch it)


_active = _auto_detect_model()

LLAMA_CLI = Path(os.environ.get("BITNET_CLI", str(_active["cli"]))).resolve()
MODEL_PATH = Path(os.environ.get("BITNET_MODEL", str(_active["model"]))).resolve()
MODEL_NAME = _active["name"]
USE_GPU = _active["use_gpu"]
IS_CHAT_MODEL = _active["is_chat_model"]

THREADS = int(os.environ.get("BITNET_THREADS", "4"))
CTX_SIZE = int(os.environ.get("BITNET_CTX_SIZE", "2048"))
MAX_INPUT_CHARS = 6000


def validate():
    """Check that required binaries and model exist."""
    errors = []
    if not LLAMA_CLI.exists():
        errors.append(f"llama-cli not found: {LLAMA_CLI}")
    if not MODEL_PATH.exists():
        errors.append(f"Model not found: {MODEL_PATH}")
    if errors:
        for e in errors:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
