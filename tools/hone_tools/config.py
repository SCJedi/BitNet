"""Path configuration and environment variable handling.

Resolution order (highest priority first):
  1. Environment variables: HONE_CLI, HONE_MODEL (BITNET_* accepted as fallback)
  2. User config file: ~/.config/hone/config.json
  3. Auto-detect repo structure (for in-repo development)

Override with env vars:
  HONE_MODEL    - path to GGUF model file
  HONE_CLI      - path to llama-cli (or llama-completion) binary
  HONE_THREADS  - number of CPU threads (default: 4)
  HONE_CTX_SIZE - context size (default: 2048)

Legacy aliases (still supported):
  BITNET_MODEL, BITNET_CLI, BITNET_THREADS, BITNET_CTX_SIZE

User config file (~/.config/hone/config.json):
  {
    "model": "/path/to/model.gguf",
    "cli": "/path/to/llama-cli",
    "threads": 4,
    "ctx_size": 2048,
    "use_gpu": true,
    "is_chat_model": true
  }

# TODO: implement `hone setup` subcommand to interactively create
# ~/.config/hone/config.json (prompt for model path, cli path,
# validate they exist, write the JSON).
"""

import json
import os
import sys
from pathlib import Path


def _env_with_fallback(primary: str, fallback: str) -> str | None:
    """Read env var with HONE_* preferred, BITNET_* as fallback."""
    return os.environ.get(primary) or os.environ.get(fallback)


# ---------------------------------------------------------------------------
# Config file support
# ---------------------------------------------------------------------------

def _config_dir() -> Path:
    """Return the platform-appropriate config directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "hone"


def _load_user_config() -> dict:
    """Load ~/.config/hone/config.json if it exists."""
    config_file = _config_dir() / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Repo auto-detection (fallback for in-repo usage)
# ---------------------------------------------------------------------------

def _find_bitnet_root() -> Path | None:
    """Walk up from tools/ to find the repo root (contains CMakeLists.txt)."""
    current = Path(__file__).resolve().parent.parent  # tools/
    for _ in range(10):
        if (current / "CMakeLists.txt").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


# Model registry for in-repo auto-detection
def _build_repo_models(bitnet_root: Path) -> list[dict]:
    return [
        {
            "name": "Qwen3.5-4B-Q4_K_M",
            "model": bitnet_root / "models" / "qwen3.5-4b" / "Qwen3.5-4B-Q4_K_M.gguf",
            "cli": bitnet_root / "tools" / "llama-cpp-latest" / "bin" / "llama-cli.exe",
            "use_gpu": True,
            "is_chat_model": True,
        },
        {
            "name": "BitNet-b1.58-2B-4T",
            "model": bitnet_root / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf",
            "cli": bitnet_root / "build" / "bin" / "llama-cli.exe",
            "use_gpu": False,
            "is_chat_model": False,
        },
    ]


def _auto_detect_repo_model(bitnet_root: Path) -> dict | None:
    """Find the best available model+cli pair from the repo."""
    for entry in _build_repo_models(bitnet_root):
        if entry["model"].exists() and entry["cli"].exists():
            return entry
    return None


# ---------------------------------------------------------------------------
# Unified resolution
# ---------------------------------------------------------------------------

def _resolve_config() -> dict:
    """Resolve model/cli paths using the 3-tier priority system.

    Returns a dict with keys: model, cli, name, use_gpu, is_chat_model,
    threads, ctx_size.
    """
    env_cli = _env_with_fallback("HONE_CLI", "BITNET_CLI")
    env_model = _env_with_fallback("HONE_MODEL", "BITNET_MODEL")
    user_cfg = _load_user_config()

    threads_env = _env_with_fallback("HONE_THREADS", "BITNET_THREADS")
    ctx_env = _env_with_fallback("HONE_CTX_SIZE", "BITNET_CTX_SIZE")

    # Start with defaults
    result = {
        "name": "unknown",
        "model": None,
        "cli": None,
        "use_gpu": user_cfg.get("use_gpu", True),
        "is_chat_model": user_cfg.get("is_chat_model", True),
        "threads": int(threads_env if threads_env else user_cfg.get("threads", 4)),
        "ctx_size": int(ctx_env if ctx_env else user_cfg.get("ctx_size", 2048)),
    }

    # Tier 3: Auto-detect from repo structure
    bitnet_root = _find_bitnet_root()
    if bitnet_root:
        repo_model = _auto_detect_repo_model(bitnet_root)
        if repo_model:
            result["name"] = repo_model["name"]
            result["model"] = repo_model["model"]
            result["cli"] = repo_model["cli"]
            result["use_gpu"] = repo_model["use_gpu"]
            result["is_chat_model"] = repo_model["is_chat_model"]

    # Tier 2: User config file overrides
    if user_cfg.get("model"):
        result["model"] = Path(user_cfg["model"]).resolve()
        result["name"] = "user-config"
    if user_cfg.get("cli"):
        result["cli"] = Path(user_cfg["cli"]).resolve()

    # Tier 1: Environment variables (highest priority)
    if env_model:
        result["model"] = Path(env_model).resolve()
        result["name"] = "env-override"
    if env_cli:
        result["cli"] = Path(env_cli).resolve()

    return result


_cfg = _resolve_config()

LLAMA_CLI = _cfg["cli"]
MODEL_PATH = _cfg["model"]
MODEL_NAME = _cfg["name"]
USE_GPU = _cfg["use_gpu"]
IS_CHAT_MODEL = _cfg["is_chat_model"]
THREADS = _cfg["threads"]
CTX_SIZE = _cfg["ctx_size"]
MAX_INPUT_CHARS = 6000


def validate():
    """Check that required binaries and model exist."""
    errors = []

    if LLAMA_CLI is None:
        errors.append(
            "llama-cli binary not found.\n"
            "  Set HONE_CLI=/path/to/llama-cli or create ~/.config/hone/config.json\n"
            "  (Legacy BITNET_CLI is also accepted)\n"
            "  See: https://github.com/SCJedi/hone-cli#configuration"
        )
    elif not LLAMA_CLI.exists():
        errors.append(f"llama-cli not found at: {LLAMA_CLI}")

    if MODEL_PATH is None:
        errors.append(
            "GGUF model not found.\n"
            "  Set HONE_MODEL=/path/to/model.gguf or create ~/.config/hone/config.json\n"
            "  (Legacy BITNET_MODEL is also accepted)\n"
            "  See: https://github.com/SCJedi/hone-cli#configuration"
        )
    elif not MODEL_PATH.exists():
        errors.append(f"Model not found at: {MODEL_PATH}")

    if errors:
        for e in errors:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
