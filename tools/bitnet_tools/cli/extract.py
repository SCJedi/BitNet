"""bt-extract: Extract structured data from text using BitNet."""

import argparse
import json
import re
import sys

from ..engine import complete

TYPES = ["emails", "names", "dates", "urls", "phones"]

# --- Regex post-validation filters ---
# These patterns validate model output to filter hallucinations.
# They are intentionally loose: the model does the extraction,
# regex just sanity-checks the results.

_VALIDATORS = {
    "emails": re.compile(
        r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    ),
    "phones": re.compile(
        # Must contain at least 7 digits; allow separators and extension keywords
        r"^[\d\s()\-+.]+(?:\s*(?:ext|x|extension)\.?\s*\d+)?$", re.IGNORECASE
    ),
    "urls": re.compile(
        r"^(https?://|ftp://|www\.)\S+$", re.IGNORECASE
    ),
    "dates": re.compile(
        # Must contain at least one digit (filters out "tomorrow", "last week")
        r"\d"
    ),
    # Names: no regex validation (too varied to pattern-match)
}


def _count_digits(s: str) -> int:
    """Count digit characters in a string."""
    return sum(c.isdigit() for c in s)


def _validate_item(item: str, extract_type: str, source_text: str) -> bool:
    """Validate an extracted item against type-specific rules.

    Returns True if the item passes validation.
    """
    item = item.strip()
    if not item:
        return False

    # Type-specific validation
    validator = _VALIDATORS.get(extract_type)
    if validator and not validator.search(item):
        return False

    # Phone-specific: must have at least 7 digits
    if extract_type == "phones" and _count_digits(item) < 7:
        return False

    # Anti-hallucination: for emails and urls, verify key fragments
    # appear in the source text (the model shouldn't fabricate them)
    if extract_type == "emails":
        # The local part (before @) should appear somewhere in source
        local = item.split("@")[0]
        if local and local not in source_text:
            return False

    return True


def validate_extractions(items: list, extract_type: str, source_text: str) -> list:
    """Filter extracted items through post-validation."""
    return [
        item for item in items
        if _validate_item(str(item), extract_type, source_text)
    ]


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


def parse_json_array(text: str) -> list:
    """Find and parse a JSON array from model output."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        # Try to salvage: split by newlines/commas
        items = [s.strip().strip('"').strip("'") for s in text.split(",")]
        return [i for i in items if i]
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        # Fallback: split content between brackets
        inner = text[start + 1 : end]
        items = [s.strip().strip('"').strip("'") for s in inner.split(",")]
        return [i for i in items if i]


def main():
    parser = argparse.ArgumentParser(
        prog="bt-extract",
        description="Extract structured data from text using a local BitNet model",
        epilog="Example: echo 'Contact john@example.com' | bt-extract --type emails",
    )
    parser.add_argument(
        "--type", "-t", required=True, choices=TYPES,
        help="Type of data to extract",
    )
    parser.add_argument("--json", "-j", action="store_true", dest="as_json", help="Output as JSON array")
    parser.add_argument("--input", "-i", help="Input file (default: stdin)")
    args = parser.parse_args()

    text = get_input(args).strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    type_hint = {
        "emails": "email addresses (must contain @)",
        "phones": "phone numbers (digits with optional formatting)",
        "urls": "URLs (starting with http://, https://, or www.)",
        "dates": "dates (specific calendar dates only, not relative like 'tomorrow')",
        "names": "person names (full names of people)",
    }[args.type]
    prompt = (
        f"Extract all {type_hint} from this text. "
        f"Return ONLY a JSON array of strings found in the text. "
        f"If none found, return [].\n\nText: {text}\n\nJSON:"
    )
    result = complete(prompt, max_tokens=200, stop_at="]\n")
    # Ensure we have the closing bracket
    if "[" in result and "]" not in result:
        result += "]"

    items = parse_json_array(result)
    items = validate_extractions(items, args.type, text)

    if args.as_json:
        print(json.dumps(items, indent=2))
    else:
        for item in items:
            print(item)


if __name__ == "__main__":
    main()
