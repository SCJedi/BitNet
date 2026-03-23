"""Core inference engine: subprocess wrapper for llama-cli.

Supports two modes:
  - CPU llama-cli (raw completion, older build)
  - Standalone llama.cpp (GPU, chat mode with single-turn for Qwen3.5+)
"""

import re
import subprocess
import sys

from .config import (
    LLAMA_CLI, MODEL_PATH, THREADS, CTX_SIZE, MAX_INPUT_CHARS,
    USE_GPU, IS_CHAT_MODEL, validate,
)


def _dedup_repetition(text: str) -> str:
    """Remove repeated phrases/sentences from model output.

    Preserves line-based structure (e.g., bullet points) when the text
    contains meaningful newlines.
    """
    # If the text has meaningful line structure (bullets, numbered lists),
    # deduplicate by line instead of by sentence to preserve formatting.
    lines = text.split("\n")
    if any(line.strip().startswith(("- ", "* ", "• ", "1.", "2.", "3.")) for line in lines):
        seen = []
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                seen.append(line)  # preserve original indentation
            elif not line_clean:
                seen.append(line)  # preserve blank lines
        return "\n".join(seen)

    # Default: sentence-level deduplication
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return text
    seen = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            seen.append(s_clean)
    return " ".join(seen)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from Qwen3.5 output."""
    # Remove think blocks (may be multiline)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove orphaned opening/closing tags
    text = re.sub(r"</?think>", "", text)
    return text.strip()


def _strip_llama_banner(text: str, prompt: str = "") -> str:
    """Strip llama.cpp banner, ANSI codes, prompt echo, and metadata from stdout.

    The new llama.cpp (b8000+) prints Loading model..., ASCII art banner,
    build info, and ANSI escape codes to stdout even with --simple-io.
    In chat mode, the output is: banner... > {prompt}\n{response}\n[ timing ]
    """
    # Strip ANSI escape codes
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)

    # Remove speed/timing lines and exit message
    text = re.sub(r"\[\s*Prompt:.*?t/s\s*\]", "", text)
    text = re.sub(r"Exiting\.\.\.", "", text)

    # Find the actual response: it comes after the "> prompt" line
    lines = text.split("\n")
    response_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("> ") and response_start is None:
            response_start = i + 1
            break

    if response_start is not None:
        response = "\n".join(lines[response_start:])
        response = response.strip()

        # The chat model echoes the prompt before responding.
        # Remove the prompt echo if present.
        if prompt:
            # Normalize whitespace for comparison
            prompt_norm = prompt.strip()
            # Try to find and remove the prompt echo
            # The echo might not be exact (formatting changes), so try prefix matching
            if response.startswith(prompt_norm):
                response = response[len(prompt_norm):].strip()
            else:
                # Try to find the prompt text within the first portion
                # Look for the last line of the prompt as a marker
                prompt_last_line = prompt_norm.rstrip().split("\n")[-1].strip()
                if prompt_last_line and prompt_last_line in response:
                    idx = response.index(prompt_last_line) + len(prompt_last_line)
                    response = response[idx:].strip()

        return response

    # Fallback: try to strip known banner patterns
    banner_patterns = [
        r"Loading model\.\.\..*?(?=\S)",
        r"▄.*?▀▀",  # ASCII art
        r"build\s+:.*",
        r"model\s+:.*",
        r"modalities\s+:.*",
        r"available commands:.*?(?:\n.*?)*?/read\s+.*",
    ]
    for pat in banner_patterns:
        text = re.sub(pat, "", text, flags=re.DOTALL)

    return text.strip()


def clean_output(text: str, stop_at: str | None = "\n") -> str:
    """Clean model output: strip fences, prefixes, truncate at stop token."""
    # Strip think tags from reasoning models (Qwen3.5, etc.)
    text = _strip_think_tags(text)

    # Strip markdown code fences
    text = re.sub(r"```[\w]*\n?", "", text)
    text = re.sub(r"```", "", text)

    # Strip HTML tags like <br>
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    # Strip markdown headings (e.g., "### Plain English Explanation\n")
    text = re.sub(r"^#{1,6}\s+[^\n]*\n*", "", text, flags=re.MULTILINE)

    # Strip bold markers (**text** -> text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)

    # Strip common prefixes (including style-specific ones)
    # For chat models, the prefix might appear mid-text (after an input echo)
    known_prefixes = ("Answer:", "Output:", "Result:", "Response:",
                      "Formal version:", "Simple version:", "Corrected:",
                      "Bullet points:", "Commit message:", "Summary:",
                      "TL;DR:", "Branch name:", "Function name:",
                      "Filename:", "Class name:", "Variable name:", "JSON:",
                      "Category:", "Classification:", "Text:", "Main point:")
    # First try: prefix at start
    for prefix in known_prefixes:
        if text.lstrip().startswith(prefix):
            text = text.lstrip()[len(prefix):]
            break
    else:
        # Second try: prefix appears later (chat model echoed input before it)
        for prefix in known_prefixes:
            idx = text.find(prefix)
            if idx > 0:
                text = text[idx + len(prefix):]
                break

    # Truncate at stop token
    if stop_at and stop_at in text:
        text = text[: text.index(stop_at)]

    # Remove mid-text prefix echoes (e.g., "text Response: text again")
    for prefix in ("Response:", "Answer:", "Output:", "Result:", "Summary:",
                    "Text:", "Main point:"):
        idx = text.find(prefix)
        if idx > 10:  # only if it appears well into the text
            text = text[:idx]
            break

    # Deduplicate repeated content
    text = _dedup_repetition(text)

    return text.strip()


def complete(
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
    stop_at: str | None = "\n",
) -> str:
    """Run a completion through llama-cli and return cleaned output."""
    validate()

    # Truncate overly long input
    if len(prompt) > MAX_INPUT_CHARS:
        print(
            f"Warning: input truncated from {len(prompt)} to {MAX_INPUT_CHARS} chars",
            file=sys.stderr,
        )
        prompt = prompt[:MAX_INPUT_CHARS]

    if IS_CHAT_MODEL:
        # Use llama-cli in single-turn chat mode for Qwen3.5+ models
        cmd = [
            str(LLAMA_CLI),
            "-m", str(MODEL_PATH),
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--simple-io",
            "--no-display-prompt",
            "--single-turn",
            "-rea", "off",      # disable thinking/reasoning for tool use
            "-t", str(THREADS),
            "-c", str(CTX_SIZE),
        ]
        if USE_GPU:
            cmd.extend(["-ngl", "99"])
        else:
            cmd.extend(["-ngl", "0"])
        timeout = 60  # GPU models may need longer for first load
    else:
        # CPU llama-cli mode (raw completion)
        cmd = [
            str(LLAMA_CLI),
            "-m", str(MODEL_PATH),
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--simple-io",
            "--no-display-prompt",
            "-ngl", "0",
            "-b", "1",
            "-t", str(THREADS),
            "-c", str(CTX_SIZE),
        ]
        timeout = 30

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        output = result.stdout
        # Strip llama.cpp banner from newer builds (chat model mode)
        if IS_CHAT_MODEL:
            output = _strip_llama_banner(output, prompt=prompt)
    except subprocess.TimeoutExpired:
        print(f"Error: inference timed out after {timeout} seconds", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: llama-cli not found at {LLAMA_CLI}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running inference: {e}", file=sys.stderr)
        sys.exit(1)

    return clean_output(output, stop_at=stop_at)
