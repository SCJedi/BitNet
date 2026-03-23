"""hone-classify: Classify text into categories using local AI."""

import argparse
import difflib
import sys

from ..engine import complete

PRESETS = {
    "sentiment": ["positive", "negative", "neutral"],
    "urgency": ["critical", "high", "medium", "low"],
    "language": ["english", "spanish", "french", "german", "chinese", "japanese", "other"],
    "topic": ["technology", "business", "science", "politics", "sports", "entertainment", "health", "other"],
}

# Brief classification hints per preset (appended to the system instruction)
PRESET_HINTS = {
    "sentiment": "Errors, crashes, failures, and bugs are negative. Mixed with more negatives than positives is negative. Only use neutral for purely factual statements.",
    "urgency": "critical=immediate danger/outage/safety, high=blocking release or urgent deadline within a week, medium=routine sprint work/no hard deadline soon, low=cosmetic/nice-to-have/no deadline.",
}


def _build_prompt(text: str, labels: list[str], label_str: str, preset: str | None) -> str:
    """Build classification prompt with targeted hints."""
    hint = ""
    if preset and preset in PRESET_HINTS:
        hint = " " + PRESET_HINTS[preset]
    elif not preset:
        hint = " Classify by the speaker's underlying intent and purpose, not by the grammatical form of the sentence. A demand phrased as a question is still a demand."

    return f"Classify into exactly one of: {label_str}.{hint}\n\nText: {text}\n\nCategory:"


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


def fuzzy_match(output: str, labels: list[str]) -> str:
    """Match model output against valid labels."""
    output_lower = output.strip().lower()
    # Exact match first
    for label in labels:
        if label.lower() == output_lower:
            return label
    # Check if output contains any label as substring
    for label in labels:
        if label.lower() in output_lower:
            return label

    # Fuzzy match
    matches = difflib.get_close_matches(output_lower, [l.lower() for l in labels], n=1, cutoff=0.4)
    if matches:
        idx = [l.lower() for l in labels].index(matches[0])
        return labels[idx]

    # Fallback: return first label rather than leaking raw model output
    return labels[0]


def main():
    parser = argparse.ArgumentParser(
        prog="hone-classify",
        description="Classify text into categories using a local AI model",
        epilog="Example: echo 'Great product!' | hone-classify --preset sentiment",
    )
    parser.add_argument(
        "--labels", "-l",
        help="Comma-separated labels (e.g., 'spam,ham')",
    )
    parser.add_argument(
        "--preset", "-p", choices=list(PRESETS.keys()),
        help="Use preset label set",
    )
    parser.add_argument("--input", "-i", help="Input file (default: stdin)")
    args = parser.parse_args()

    if not args.labels and not args.preset:
        parser.error("--labels or --preset is required")

    labels = PRESETS[args.preset] if args.preset else [l.strip() for l in args.labels.split(",")]

    text = get_input(args).strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    label_str = ", ".join(labels)
    prompt = _build_prompt(text, labels, label_str, args.preset)
    result = complete(prompt, max_tokens=10, stop_at="\n")
    matched = fuzzy_match(result, labels)
    print(matched)


if __name__ == "__main__":
    main()
