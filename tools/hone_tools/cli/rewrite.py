"""hone-rewrite: Rewrite text in a given style using local AI."""

import argparse
import sys

from ..engine import complete

STYLE_PROMPTS = {
    "formal": "Rewrite the following text in a formal, professional tone:\n\n{input}\n\nFormal version:",
    "simple": "Rewrite the following text in simple, plain language:\n\n{input}\n\nSimple version:",
    "punctuate": "Add proper punctuation and capitalization to this text:\n\n{input}\n\nCorrected:",
    "bullets": "Convert this text into bullet points:\n\n{input}\n\nBullet points:",
    "commit": "Rewrite as a concise git commit message (imperative mood, max 72 chars):\n\n{input}\n\nCommit message:",
}


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
        prog="hone-rewrite",
        description="Rewrite text in a given style using a local AI model",
        epilog="Example: echo 'hey fix the bug pls' | hone-rewrite --style formal",
    )
    parser.add_argument(
        "--style", "-s", required=True, choices=list(STYLE_PROMPTS.keys()),
        help="Rewriting style",
    )
    parser.add_argument("--input", "-i", help="Input file (default: stdin)")
    args = parser.parse_args()

    text = get_input(args).strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    prompt = STYLE_PROMPTS[args.style].format(input=text)
    max_tok = min(200, int(len(text) / 3 * 2))
    max_tok = max(max_tok, 60)  # floor: ensure enough room for short inputs

    # Commit messages should be single-line; others may need multi-line output
    if args.style == "commit":
        stop = "\n"
    elif args.style in ("bullets", "formal"):
        stop = None  # let token limit control length
    else:
        stop = "\n\n"

    result = complete(prompt, max_tokens=max_tok, stop_at=stop)
    print(result)


if __name__ == "__main__":
    main()
