#!/usr/bin/env python3
"""
Setup Qwen3.5-9B Q4_K_M for local inference via llama-cli.
Downloads the GGUF and configures hone-tools.

Run: python setup_qwen35_9b.py
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "qwen3.5-9b"
CONFIG_DIR = Path.home() / ".config" / "hone"
CONFIG_FILE = CONFIG_DIR / "config.json"
LLAMA_CLI = PROJECT_ROOT / "tools" / "llama-cpp-latest" / "bin" / "llama-cli.exe"


def download_gguf():
    """Download Qwen3.5-9B Q4_K_M GGUF from HuggingFace."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Try known repos in order
    repos_and_files = [
        ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
        ("Qwen/Qwen3.5-9B-GGUF", "qwen3.5-9b-q4_k_m.gguf"),
        ("bartowski/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
    ]

    from huggingface_hub import hf_hub_download, list_repo_files

    for repo, default_file in repos_and_files:
        try:
            # List files to find the right Q4_K_M file
            files = list_repo_files(repo)
            q4_files = [f for f in files if "Q4_K_M" in f or "q4_k_m" in f]
            if not q4_files:
                print(f"  {repo}: no Q4_K_M files found")
                continue

            print(f"  Found in {repo}: {q4_files}")

            # Download all Q4_K_M files (may be sharded)
            for fname in q4_files:
                print(f"  Downloading {fname}...")
                path = hf_hub_download(repo, fname, local_dir=str(MODEL_DIR))
                print(f"  -> {path}")

            return q4_files[0]  # Return first filename for config

        except Exception as e:
            print(f"  {repo}: {e}")
            continue

    print("\nERROR: Could not download from any repo.")
    print("Try manually: huggingface-cli download unsloth/Qwen3.5-9B-GGUF --include '*Q4_K_M*' --local-dir models/qwen3.5-9b/")
    return None


def find_gguf():
    """Find existing GGUF file in model dir."""
    if MODEL_DIR.exists():
        ggufs = list(MODEL_DIR.glob("*.gguf"))
        if ggufs:
            return ggufs[0].name
    return None


def update_config(gguf_name):
    """Update hone-tools config to use the 9B model."""
    model_path = str(MODEL_DIR / gguf_name)

    config = {
        "cli": str(LLAMA_CLI),
        "model": model_path,
        "is_chat_model": True,
        "use_gpu": True,
        "threads": 4,
        "ctx_size": 8192,
    }

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve existing config keys
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            existing = json.load(f)
        existing.update(config)
        config = existing

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Config written to: {CONFIG_FILE}")
    print(f"  Model: {model_path}")
    print(f"  CLI: {LLAMA_CLI}")


def test_inference(gguf_name):
    """Quick test that the model loads and generates."""
    import subprocess

    model_path = str(MODEL_DIR / gguf_name)
    cmd = [
        str(LLAMA_CLI),
        "-m", model_path,
        "-ngl", "99",
        "-c", "2048",
        "-t", "4",
        "-rea", "off",
        "-p", "The capital of France is",
        "-n", "20",
    ]

    print(f"\n  Testing inference...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"  SUCCESS - model loaded and generated text")
            return True
        else:
            print(f"  FAILED - exit code {result.returncode}")
            print(f"  stderr: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT - model may be too large for GPU")
        return False
    except FileNotFoundError:
        print(f"  ERROR - llama-cli not found at {LLAMA_CLI}")
        return False


def main():
    print("=" * 60)
    print("Qwen3.5-9B Q4_K_M Setup")
    print("=" * 60)

    # Step 1: Check for existing GGUF
    print("\n[1/4] Checking for existing GGUF...")
    gguf_name = find_gguf()
    if gguf_name:
        print(f"  Found: {MODEL_DIR / gguf_name}")
    else:
        # Step 2: Download
        print("\n[2/4] Downloading GGUF...")
        gguf_name = download_gguf()
        if not gguf_name:
            sys.exit(1)
        # Re-check after download
        gguf_name = find_gguf()
        if not gguf_name:
            print("ERROR: Download succeeded but file not found")
            sys.exit(1)

    # Step 3: Update config
    print("\n[3/4] Updating hone-tools config...")
    update_config(gguf_name)

    # Step 4: Test
    print("\n[4/4] Testing inference...")
    success = test_inference(gguf_name)

    print("\n" + "=" * 60)
    if success:
        print("SETUP COMPLETE - Qwen3.5-9B ready for local inference")
        print(f"\nTest with: echo 'Hello world' | hone-classify sentiment")
    else:
        print("SETUP PARTIAL - Config written but inference test failed")
        print("Check GPU memory and llama-cli path")
    print("=" * 60)


if __name__ == "__main__":
    main()
