"""bt-summarize: Summarize text using BitNet."""

import argparse
import re
import sys

from ..engine import complete


# Minimum input length (chars) to attempt summarization.
# Below this, return input as-is (avoids hallucination on trivial input).
MIN_SUMMARIZE_CHARS = 40


def get_input(args):
    """Read input from file argument or stdin."""
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


def _count_sentences(text: str) -> int:
    """Count sentences in text (split on .!? followed by space or end)."""
    parts = re.split(r'[.!?]+(?:\s|$)', text.strip())
    return len([p for p in parts if p.strip()])


def _trim_to_sentences(text: str, n: int) -> str:
    """Trim text to at most n sentences."""
    sentences = re.findall(r'[^.!?]*[.!?]+', text)
    if len(sentences) <= n:
        return text.strip()
    return ''.join(sentences[:n]).strip()


def _is_echo(text: str, original: str) -> bool:
    """Detect if the output is just echoing the input."""
    t = ' '.join(text.lower().split())
    o = ' '.join(original.lower().split())
    # If output starts with a long prefix of the original
    if len(t) >= len(o) * 0.8 and (
        t.startswith(o[:len(o)//2]) or o.startswith(t[:len(t)//2])
    ):
        return True
    # High similarity ratio
    if len(t) > 0 and len(o) > 0:
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, t, o).ratio()
        if ratio > 0.85:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        prog="bt-summarize",
        description="Summarize text using a local BitNet model",
        epilog="Example: cat article.txt | bt-summarize --sentences 2",
    )
    parser.add_argument(
        "--sentences", "-s", type=int, default=1,
        help="Number of sentences in summary (default: 1)",
    )
    parser.add_argument("--input", "-i", help="Input file (default: stdin)")
    args = parser.parse_args()

    if args.sentences < 1:
        print("Error: --sentences must be >= 1", file=sys.stderr)
        sys.exit(1)

    text = get_input(args).strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    # Guard: trivially short input — return as-is instead of hallucinating
    if len(text) < MIN_SUMMARIZE_CHARS:
        print(text)
        return

    # Guard: if input already has <= requested sentences, return as-is
    input_sentences = _count_sentences(text)
    if input_sentences <= args.sentences:
        print(text)
        return

    n = args.sentences
    # Token budget: ~80 tokens per sentence, minimum 100
    max_tokens = max(100, 80 * n)

    # Prompt: text first, then instruction (prevents echo on long input)
    prompt = (
        f"{text}\n\n"
        f"Write a concise {n}-sentence summary of the above.\n\n"
        f"Summary:"
    )
    result = complete(prompt, max_tokens=max_tokens, stop_at="\n\n")

    # Post-processing: detect and handle echo
    if _is_echo(result, text):
        # Retry with TL;DR style (proven to work on echoing inputs)
        prompt = f"{text}\n\nTL;DR in {n} sentence{'s' if n > 1 else ''}:"
        result = complete(prompt, max_tokens=max_tokens, stop_at="\n\n")

        # If still echoing after retry, force-trim the original
        if _is_echo(result, text):
            result = _trim_to_sentences(text, n)

    # Enforce sentence count: trim if too many
    actual_count = _count_sentences(result)
    if actual_count > n:
        result = _trim_to_sentences(result, n)

    print(result)


if __name__ == "__main__":
    main()
