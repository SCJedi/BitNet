"""bt-jsonify: Convert unstructured text to JSON using BitNet."""

import argparse
import json
import sys

from ..engine import complete


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


def parse_json_object(text: str) -> dict:
    """Find and parse a JSON object from model output."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"raw": text.strip()}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"raw": text[start : end + 1]}


def main():
    parser = argparse.ArgumentParser(
        prog="bt-jsonify",
        description="Convert unstructured text to JSON using a local BitNet model",
        epilog="Example: echo 'John Doe, age 30, NYC' | bt-jsonify --fields name,age,city",
    )
    parser.add_argument(
        "--fields", "-f", required=True,
        help="Comma-separated field names",
    )
    parser.add_argument("--input", "-i", help="Input file (default: stdin)")
    args = parser.parse_args()

    text = get_input(args).strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    fields = ", ".join(f.strip() for f in args.fields.split(","))
    prompt = f"Convert to JSON with fields: {fields}. Use null if not found.\n\nText: {text}\n\nJSON:"
    result = complete(prompt, max_tokens=200, stop_at="}\n")
    # Ensure closing brace
    if "{" in result and "}" not in result:
        result += "}"

    obj = parse_json_object(result)

    # Filter to only requested fields (defense against prompt injection)
    requested = [f.strip() for f in args.fields.split(",")]
    if "raw" not in obj:  # only filter if we got structured output
        filtered = {}
        for field in requested:
            filtered[field] = obj.get(field, None)
        obj = filtered

    print(json.dumps(obj, indent=2))


if __name__ == "__main__":
    main()
