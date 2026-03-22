"""bt-tldr: Smart TL;DR for diffs, logs, errors, and general text."""

import argparse
import re
import sys

from ..engine import complete

MODE_PROMPTS = {
    "diff": "Summarize this code diff in one line. Focus on what changed and why:\n\n{input}\n\nTL;DR:",
    "log": "Summarize these log entries. What happened and what matters:\n\n{input}\n\nTL;DR:",
    "error": "Explain this error in plain English. What went wrong and how to fix it:\n\n{input}\n\nTL;DR:",
    "generic": "Give a one-line TL;DR of this text:\n\n{input}\n\nTL;DR:",
}


def detect_mode(text: str) -> str:
    """Auto-detect the type of input."""
    # Diff: unified diff markers OR "diff --git" header
    if re.search(r"^(\+\+\+|---|@@|diff\s+--git\s)", text, re.MULTILINE):
        return "diff"
    # Log: ISO timestamps, bracket timestamps, or syslog-style (Mon DD HH:MM:SS)
    if re.search(
        r"(\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}"
        r"|^\[?\d{4}[-/]\d{2}[-/]\d{2}\s"
        r"|^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s)",
        text,
        re.MULTILINE,
    ):
        return "log"
    # Error detection: three tiers
    # Tier 1 - Strong signals (case-insensitive): always indicate errors
    if re.search(
        r"(traceback|exception|segmentation\s+fault"
        r"|sigsegv|sigabrt|sigbus|core\s+dump"
        r"|stack\s*trace|unhandled\b|uncaught\b"
        r"|panic(?:ked)?\s+at\b)",
        text,
        re.IGNORECASE,
    ):
        return "error"
    # Tier 2 - Case-sensitive: ALL-CAPS keywords used in structured error output
    if re.search(
        r"(\bERROR\b|\bERR!|\bFATAL\b|\bBUG\b\s*:|\bPANIC\b)",
        text,
    ):
        return "error"
    # Tier 3 - Contextual (case-insensitive): need error-like punctuation/formatting
    if re.search(
        r"(\berror\s*[\[:(]"          # error followed by : [ (
        r"|\w+error\s*:"              # TypeError:, ValueError:, etc.
        r"|\bfailed\s*:"              # "failed:" pattern
        r"|\babort(?:ed)\b"           # "aborted" (not bare "abort" to avoid false positives)
        r"|\bpanicked?\s+at\b)",      # rust panic
        text,
        re.IGNORECASE,
    ):
        return "error"
    return "generic"


def get_input(args):
    if args.input:
        try:
            with open(args.input) as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        except PermissionError:
            print(f"Error: permission denied: {args.input}", file=sys.stderr)
            sys.exit(1)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("Error: no input (pipe text or use -i FILE)", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="bt-tldr",
        description="Smart TL;DR for diffs, logs, errors, and general text",
        epilog="Example: git diff | bt-tldr",
    )
    parser.add_argument(
        "--mode", "-m", choices=list(MODE_PROMPTS.keys()),
        help="Force mode (default: auto-detect)",
    )
    parser.add_argument("--input", "-i", help="Input file (default: stdin)")
    args = parser.parse_args()

    text = get_input(args).strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    mode = args.mode or detect_mode(text)
    prompt = MODE_PROMPTS[mode].format(input=text)
    result = complete(prompt, max_tokens=100, stop_at=None)
    print(result)


if __name__ == "__main__":
    main()
