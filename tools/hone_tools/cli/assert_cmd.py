"""hone-assert: Natural language test assertions with binary pass/fail output.

Evaluates whether a value satisfies a natural-language assertion.
Exit code 0 = PASS, exit code 1 = FAIL.

Pre-processes obvious cases (numeric comparisons, contains checks, regex)
without invoking the model. Falls back to the model for ambiguous assertions.
"""

import argparse
import json
import re
import sys

from ..engine import complete


# ---------------------------------------------------------------------------
# Pre-processing: fast deterministic evaluation for common assertion patterns
# ---------------------------------------------------------------------------

def _try_numeric(val_str: str):
    """Try to parse a numeric value, return float or None."""
    try:
        return float(val_str.replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _parse_number_from_text(text: str):
    """Extract a number mentioned in assertion text."""
    m = re.search(r'(?:^|[\s(])(-?\d+(?:\.\d+)?)', text)
    if m:
        return float(m.group(1))
    return None


def _is_valid_email(val: str) -> bool:
    """Basic email validation."""
    return bool(re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', val.strip()))


def _is_valid_url(val: str) -> bool:
    """Basic URL validation."""
    return bool(re.match(r'^https?://[^\s]+\.[^\s]+', val.strip()))


def _try_json_parse(val: str):
    """Try to parse value as JSON, return parsed or None."""
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None


def _count_items(val: str) -> int | None:
    """Try to determine item count from a value (JSON array, CSV, etc.)."""
    parsed = _try_json_parse(val)
    if isinstance(parsed, list):
        return len(parsed)
    # Try comma-separated
    if "," in val and not val.strip().startswith("{"):
        return len([x.strip() for x in val.split(",") if x.strip()])
    return None


def _preprocess(assertion: str, value: str) -> bool | None:
    """Try to evaluate the assertion deterministically.

    Returns True (PASS), False (FAIL), or None (needs model).
    """
    a = assertion.strip().lower()
    v = value.strip()
    v_lower = v.lower()

    # --- Status code / equality checks ---
    # "status code is 200", "value is 200", "equals 200"
    for pattern in [
        r'(?:status\s+code|value|result|output|response)\s+(?:is|equals?|==)\s+(\S+)',
        r'(?:is|equals?|==)\s+(\S+)$',
        r'^(\S+)$',  # bare assertion like "200" — skip, too ambiguous
    ]:
        m = re.search(pattern, a)
        if m and pattern != r'^(\S+)$':
            expected = m.group(1).strip('"\'')
            if v.strip('"\'') == expected:
                return True
            # Numeric comparison
            ev = _try_numeric(expected)
            vv = _try_numeric(v)
            if ev is not None and vv is not None and ev == vv:
                return True

    # --- Numeric comparisons ---
    # "greater than 5", "less than 100", "at least 3", "at most 10"
    # "more than 5", "fewer than 10"
    num_val = _try_numeric(v)
    if num_val is not None:
        patterns = [
            (r'(?:greater|more|bigger|larger|higher|above)\s+than\s+(-?\d+(?:\.\d+)?)', lambda n: num_val > n),
            (r'(?:less|fewer|smaller|lower|below)\s+than\s+(-?\d+(?:\.\d+)?)', lambda n: num_val < n),
            (r'(?:at\s+least|>=|no\s+less\s+than|minimum)\s+(-?\d+(?:\.\d+)?)', lambda n: num_val >= n),
            (r'(?:at\s+most|<=|no\s+more\s+than|maximum)\s+(-?\d+(?:\.\d+)?)', lambda n: num_val <= n),
            (r'(?:equals?|==|is\s+exactly|is)\s+(-?\d+(?:\.\d+)?)', lambda n: num_val == n),
            (r'(?:not\s+equal|!=|is\s+not)\s+(-?\d+(?:\.\d+)?)', lambda n: num_val != n),
            (r'(?:between)\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)', None),  # handled below
            (r'(?:is\s+)?(?:positive)', lambda n: num_val > 0),
            (r'(?:is\s+)?(?:negative)', lambda n: num_val < 0),
            (r'(?:is\s+)?(?:zero)', lambda n: num_val == 0),
            (r'(?:is\s+)?(?:even)', lambda n: num_val == int(num_val) and int(num_val) % 2 == 0),
            (r'(?:is\s+)?(?:odd)', lambda n: num_val == int(num_val) and int(num_val) % 2 != 0),
        ]
        for pat, fn in patterns:
            m = re.search(pat, a)
            if m:
                if 'between' in pat and m:
                    lo, hi = float(m.group(1)), float(m.group(2))
                    return lo <= num_val <= hi
                if fn is not None:
                    try:
                        n = float(m.group(1)) if m.lastindex and m.lastindex >= 1 else 0
                        return fn(n)
                    except (ValueError, IndexError):
                        # For patterns like "positive" that don't capture a number
                        return fn(0)

    # --- Contains / does not contain ---
    # Check negation first
    neg_contains = re.search(
        r'(?:does\s+not|doesn\'t|does\s+not|should\s+not|shouldn\'t|must\s+not)\s+contain[s]?\s+["\']?(.+?)["\']?\s*$',
        a
    )
    if neg_contains:
        needle = neg_contains.group(1).strip('"\'')
        return needle.lower() not in v_lower

    pos_contains = re.search(r'contains?\s+["\']?(.+?)["\']?\s*$', a)
    if pos_contains:
        needle = pos_contains.group(1).strip('"\'')
        return needle.lower() in v_lower

    # --- Starts with / ends with ---
    starts = re.search(r'starts?\s+with\s+["\']?(.+?)["\']?\s*$', a)
    if starts:
        prefix = starts.group(1).strip('"\'')
        return v_lower.startswith(prefix.lower())

    ends = re.search(r'ends?\s+with\s+["\']?(.+?)["\']?\s*$', a)
    if ends:
        suffix = ends.group(1).strip('"\'')
        return v_lower.endswith(suffix.lower())

    # --- Empty / not empty ---
    if re.search(r'(?:is\s+)?(?:empty|blank|null|nil|none)', a):
        if re.search(r'(?:not|non|isn\'t|is\s+not)\s+(?:empty|blank|null|nil|none)', a):
            return bool(v) and v_lower not in ('null', 'nil', 'none', '[]', '{}', '')
        return not v or v_lower in ('null', 'nil', 'none', '[]', '{}', '')

    # not empty (standalone)
    if re.search(r'(?:is\s+)?not\s+empty', a):
        return bool(v) and v_lower not in ('null', 'nil', 'none', '[]', '{}', '')

    # --- Type checks ---
    if re.search(r'(?:is\s+(?:a\s+)?(?:valid\s+)?)?(?:number|numeric|integer|int|float)', a):
        if re.search(r'(?:not|isn\'t)', a):
            return _try_numeric(v) is None
        if re.search(r'integer|int\b', a):
            n = _try_numeric(v)
            return n is not None and n == int(n)
        return _try_numeric(v) is not None

    if re.search(r'(?:is\s+(?:a\s+)?(?:valid\s+)?)?(?:email)', a):
        return _is_valid_email(v)

    if re.search(r'(?:is\s+(?:a\s+)?(?:valid\s+)?)?(?:url)', a):
        return _is_valid_url(v)

    if re.search(r'(?:is\s+(?:a\s+)?(?:valid\s+)?)?(?:json)', a):
        return _try_json_parse(v) is not None

    if re.search(r'(?:is\s+(?:a\s+)?)?(?:boolean|bool)', a):
        return v_lower in ('true', 'false', '0', '1', 'yes', 'no')

    if re.search(r'(?:is\s+(?:a\s+)?)?string', a) and not re.search(r'contains|starts|ends|empty', a):
        return isinstance(v, str) and _try_numeric(v) is None

    # --- List/array length checks ---
    # "list has more than 5 items", "array length is 3", "has at least 2 elements"
    count = _count_items(v)
    if count is not None:
        len_patterns = [
            (r'(?:has|have|with|length\s+(?:is\s+)?)?(?:more|greater)\s+than\s+(\d+)\s+(?:items?|elements?|entries?)', lambda n: count > n),
            (r'(?:has|have|with|length\s+(?:is\s+)?)?(?:fewer|less)\s+than\s+(\d+)\s+(?:items?|elements?|entries?)', lambda n: count < n),
            (r'(?:has|have|with|length\s+(?:is\s+)?)?(?:at\s+least|>=)\s+(\d+)\s+(?:items?|elements?|entries?)', lambda n: count >= n),
            (r'(?:has|have|with|length\s+(?:is\s+)?)?(?:at\s+most|<=)\s+(\d+)\s+(?:items?|elements?|entries?)', lambda n: count <= n),
            (r'(?:has|have|with|length\s+(?:is\s+)?)?(?:exactly\s+)?(\d+)\s+(?:items?|elements?|entries?)', lambda n: count == n),
            (r'(?:length|size|count)\s+(?:is|==|equals?)\s+(\d+)', lambda n: count == n),
            (r'(?:has|have|with)\s+(\d+)\s+(?:items?|elements?|entries?)', lambda n: count == n),
            (r'(?:is\s+)?empty', lambda n: count == 0),
        ]
        for pat, fn in len_patterns:
            m = re.search(pat, a)
            if m:
                try:
                    n = int(m.group(1))
                except (IndexError, AttributeError):
                    n = 0
                return fn(n)

    # --- True / false checks ---
    if re.search(r'^(?:is\s+)?true$', a):
        return v_lower in ('true', '1', 'yes')
    if re.search(r'^(?:is\s+)?false$', a):
        return v_lower in ('false', '0', 'no')

    # --- Matches regex ---
    regex_match = re.search(r'(?:matches?\s+(?:regex|pattern)\s+)(.+)$', a)
    if regex_match:
        pattern = regex_match.group(1).strip('"\'/ ')
        try:
            return bool(re.search(pattern, v))
        except re.error:
            return None  # invalid regex, let model handle it

    # --- Response contains error (common CI pattern) ---
    if re.search(r'response\s+contains?\s+error', a):
        return 'error' in v_lower

    # Can't determine — fall back to model
    return None


# ---------------------------------------------------------------------------
# Model-based evaluation
# ---------------------------------------------------------------------------

def _model_evaluate(assertion: str, value: str) -> bool:
    """Use the model to evaluate the assertion."""
    prompt = (
        f"Does the following value satisfy the assertion?\n\n"
        f"Assertion: {assertion}\n"
        f"Value: {value}\n\n"
        f"Reply with ONLY one word: PASS or FAIL."
    )
    result = complete(prompt, max_tokens=10, temperature=0.0, stop_at="\n")
    result_upper = result.strip().upper()

    if "PASS" in result_upper:
        return True
    if "FAIL" in result_upper:
        return False

    # Ambiguous model output — default to FAIL (conservative)
    print(f"Warning: ambiguous model output: {result!r}, defaulting to FAIL", file=sys.stderr)
    return False


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _get_assertion(args) -> str:
    """Get assertion text from file or stdin."""
    if args.input:
        try:
            with open(args.input) as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Error: file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        except PermissionError:
            print(f"Error: permission denied: {args.input}", file=sys.stderr)
            sys.exit(1)
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print("Error: no assertion (pipe text or use -i FILE)", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="hone-assert",
        description="Natural language test assertions with binary pass/fail output",
        epilog='Example: echo "status code is 200" | hone-assert --val "200"',
    )
    parser.add_argument(
        "--val", "-v", required=True,
        help="The value to check the assertion against",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print PASS/FAIL with explanation instead of just exit code",
    )
    parser.add_argument(
        "--input", "-i",
        help="Read assertion from file (default: stdin)",
    )
    args = parser.parse_args()

    assertion = _get_assertion(args)
    if not assertion:
        print("Error: empty assertion", file=sys.stderr)
        sys.exit(1)

    value = args.val

    # Try fast pre-processing first
    result = _preprocess(assertion, value)

    if result is not None:
        passed = result
        method = "pre-processed"
    else:
        passed = _model_evaluate(assertion, value)
        method = "model"

    if args.verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status} ({method}): assertion={assertion!r} value={value!r}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
