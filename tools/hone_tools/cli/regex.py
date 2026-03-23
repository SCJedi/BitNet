"""hone-regex: Generate regex patterns from natural language using local AI."""

import argparse
import re
import sys

from ..engine import complete


FLAVOR_HINTS = {
    "python": "Python re module syntax",
    "javascript": "JavaScript RegExp syntax (no lookbehind, no named groups with P)",
    "grep": "POSIX Extended Regular Expression (ERE) for grep -E",
    "pcre": "PCRE (Perl-Compatible Regular Expression) syntax",
}

PROMPT_TEMPLATE = (
    "Write ONLY a regex pattern for: {description}\n"
    "Flavor: {flavor_hint}\n"
    "Reply with ONLY the raw pattern. No explanation. No code fences. Just the pattern."
)

RETRY_TEMPLATE = (
    "The regex pattern you gave was invalid. Error: {error}\n"
    "Write a VALID regular expression that matches: {description}\n"
    "Flavor: {flavor_hint}\n"
    "Output ONLY the regex pattern, nothing else.\n\n"
    "Pattern:"
)

EXPLAIN_TEMPLATE = (
    "Explain this regex pattern in one short sentence: {pattern}\n"
    "Explanation:"
)


def _extract_from_code_block(raw: str) -> str | None:
    """Extract content from markdown code blocks."""
    # ```regex\n...\n``` or ```\n...\n```
    m = re.search(r"```(?:regex|re|python|js|javascript|pcre|text)?\s*\n(.+?)\n```", raw, re.DOTALL)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    return None


def _extract_from_backticks(raw: str) -> str | None:
    """Extract content from inline backticks."""
    # Find all backtick-quoted strings and pick the one that looks most like a regex
    matches = re.findall(r"`([^`]+)`", raw)
    if not matches:
        return None
    # Prefer the one with regex-like characters
    regex_chars = set(r"[]\(){}+*?|^$.\d\w\s")
    scored = []
    for m in matches:
        score = sum(1 for c in m if c in regex_chars)
        scored.append((score, m))
    scored.sort(reverse=True)
    return scored[0][1] if scored else None


def _looks_like_regex(s: str) -> bool:
    """Heuristic: does this string look like a regex pattern?"""
    if not s or len(s) < 2:
        return False
    # Contains common regex metacharacters
    regex_indicators = set(r"[]+*?(){}|\^$")
    indicator_count = sum(1 for c in s if c in regex_indicators)
    # Must have at least some regex syntax
    if indicator_count >= 1:
        return True
    # Or contains \d, \w, \s, etc.
    if re.search(r"\\[dwsWDS]", s):
        return True
    return False


def _clean_pattern(raw: str) -> str:
    """Extract regex pattern from potentially verbose model output."""
    text = raw.strip()

    if not text:
        return ""

    # Strategy 1: Try extracting from code blocks
    from_block = _extract_from_code_block(text)
    if from_block and _looks_like_regex(from_block):
        return from_block.strip("`").strip()

    # Strategy 2: Try extracting from inline backticks
    from_bt = _extract_from_backticks(text)
    if from_bt and _looks_like_regex(from_bt):
        return from_bt.strip()

    # Strategy 3: Look for "Pattern:" or similar labels
    for label in ("Pattern:", "Regex:", "pattern:", "regex:"):
        idx = text.find(label)
        if idx >= 0:
            after = text[idx + len(label):].strip()
            # Take the first line or backtick-delimited portion
            line = after.split("\n")[0].strip().strip("`").strip("'\"")
            if _looks_like_regex(line):
                return line

    # Strategy 4: Try each line, pick the first that looks like a regex
    lines = text.split("\n")
    for line in lines:
        line = line.strip().strip("`").strip("'\"")
        # Skip empty and obviously not-regex lines
        if not line or line.startswith(("#", "Here", "This", "The ", "To ", "A ", "Note")):
            continue
        # Strip JS /pattern/flags
        js_match = re.match(r"^/(.+)/[gimsuy]*$", line)
        if js_match:
            line = js_match.group(1)
        if _looks_like_regex(line):
            # Validate it compiles before accepting
            try:
                re.compile(line)
                return line
            except re.error:
                continue

    # Strategy 5: Last resort - strip everything and try first line
    text = re.sub(r"```[\w]*\n?", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip().strip("`").strip("'\"")
    first_line = text.split("\n")[0].strip()

    # Strip known prefixes
    for prefix in ("Pattern:", "Regex:", "Output:", "Result:", "Answer:"):
        if first_line.startswith(prefix):
            first_line = first_line[len(prefix):].strip()

    return first_line.strip("`").strip()


def _validate_regex(pattern: str) -> tuple[bool, str]:
    """Try to compile the pattern. Returns (ok, error_message)."""
    try:
        re.compile(pattern)
        return True, ""
    except re.error as e:
        return False, str(e)


def _generate_pattern(description: str, flavor: str) -> str:
    """Generate a regex pattern, with one retry on failure."""
    flavor_hint = FLAVOR_HINTS[flavor]

    prompt = PROMPT_TEMPLATE.format(
        description=description,
        flavor_hint=flavor_hint,
    )
    raw = complete(prompt, max_tokens=200, stop_at=None)
    pattern = _clean_pattern(raw)

    # Validate
    ok, error = _validate_regex(pattern)
    if ok and pattern:
        return pattern

    # Retry once with error feedback
    retry_prompt = RETRY_TEMPLATE.format(
        error=error if error else "empty output",
        description=description,
        flavor_hint=flavor_hint,
    )
    raw2 = complete(retry_prompt, max_tokens=200, stop_at=None)
    pattern2 = _clean_pattern(raw2)

    ok2, error2 = _validate_regex(pattern2)
    if ok2 and pattern2:
        return pattern2

    # If both fail, return what we have (or first attempt)
    if pattern:
        return pattern
    if pattern2:
        return pattern2

    print("Error: model failed to generate a valid regex pattern", file=sys.stderr)
    sys.exit(1)


def _test_pattern(pattern: str, test_string: str) -> list[str]:
    """Run re.findall with the pattern on the test string."""
    try:
        matches = re.findall(pattern, test_string)
        # Flatten tuples from groups
        flat = []
        for m in matches:
            if isinstance(m, tuple):
                flat.append(m[0] if m[0] else "".join(m))
            else:
                flat.append(m)
        return flat
    except re.error as e:
        print(f"Error testing regex: {e}", file=sys.stderr)
        return []


def _get_explanation(pattern: str) -> str:
    """Get a brief explanation of the pattern from the model."""
    prompt = EXPLAIN_TEMPLATE.format(pattern=pattern)
    return complete(prompt, max_tokens=80, stop_at=None).strip()


def get_input(args):
    """Get description from file, stdin, or positional arg."""
    if args.input_file:
        try:
            with open(args.input_file) as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: file not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)
        except PermissionError:
            print(f"Error: permission denied: {args.input_file}", file=sys.stderr)
            sys.exit(1)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("Error: no input (pipe text or use -i FILE)", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="hone-regex",
        description="Generate regex patterns from natural language",
        epilog='Example: echo "email addresses" | hone-regex --test "hi john@x.com"',
    )
    parser.add_argument(
        "--flavor", "-f", default="python",
        choices=list(FLAVOR_HINTS.keys()),
        help="Regex flavor/dialect (default: python)",
    )
    parser.add_argument(
        "--test", "-t", metavar="STRING",
        help="Test string to validate the regex against",
    )
    parser.add_argument(
        "--explain", "-e", action="store_true",
        help="Also output a brief explanation of the pattern",
    )
    parser.add_argument(
        "-i", dest="input_file", metavar="FILE",
        help="Read description from file (default: stdin)",
    )
    args = parser.parse_args()

    description = get_input(args).strip()
    if not description:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    pattern = _generate_pattern(description, args.flavor)
    print(pattern)

    if args.explain:
        explanation = _get_explanation(pattern)
        print(f"\nExplanation: {explanation}")

    if args.test:
        matches = _test_pattern(pattern, args.test)
        if matches:
            print(f"\nMatches in test string:")
            for m in matches:
                print(f"  {m}")
        else:
            print(f"\nNo matches found in test string.")


if __name__ == "__main__":
    main()
