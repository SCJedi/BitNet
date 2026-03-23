"""hone-cron: Natural language to cron expression converter using local AI.

Bidirectional:
  echo "every weekday at 9am" | hone-cron          → 0 9 * * 1-5
  echo "0 9 * * 1-5" | hone-cron --explain         → "Every weekday at 9:00 AM"
  echo "every day at 3pm" | hone-cron --validate    → expression + next 3 run times
"""

import argparse
import re
import sys
from datetime import datetime, timedelta

from ..engine import complete


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GENERATE_TEMPLATE = (
    "Convert this to a cron expression (5 fields: minute hour day-of-month month day-of-week).\n"
    "Reply with ONLY the 5-field cron expression. No explanation. No code fences.\n\n"
    "{description}\n\n"
    "Cron:"
)

RETRY_TEMPLATE = (
    "Your previous answer was not a valid 5-field cron expression.\n"
    "Convert this to a cron expression (5 fields: minute hour day-of-month month day-of-week).\n"
    "Reply with ONLY the 5-field cron expression, like: 0 9 * * 1-5\n\n"
    "{description}\n\n"
    "Cron:"
)

EXPLAIN_TEMPLATE = (
    "Explain this cron expression in plain English in one short sentence.\n"
    "Cron: {expression}\n"
    "Explanation:"
)


# ---------------------------------------------------------------------------
# Cron field validation (no external deps)
# ---------------------------------------------------------------------------

# Month and day-of-week name mappings
MONTH_NAMES = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
DOW_NAMES = {
    "SUN": 0, "MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6,
}

# Field specs: (name, min_val, max_val, name_map_or_None)
FIELD_SPECS = [
    ("minute", 0, 59, None),
    ("hour", 0, 23, None),
    ("day_of_month", 1, 31, None),
    ("month", 1, 12, MONTH_NAMES),
    ("day_of_week", 0, 7, DOW_NAMES),  # 0 and 7 both = Sunday
]


def _resolve_name(token: str, name_map: dict | None) -> str:
    """Replace month/day names with numbers."""
    if name_map is None:
        return token
    upper = token.upper()
    if upper in name_map:
        return str(name_map[upper])
    return token


def _validate_field(field: str, min_val: int, max_val: int, name_map: dict | None) -> bool:
    """Validate a single cron field."""
    if field == "*":
        return True

    # Handle lists: 1,3,5
    parts = field.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            return False

        # Handle step: */5 or 1-10/2
        if "/" in part:
            base, step_str = part.split("/", 1)
            try:
                step = int(step_str)
                if step < 1:
                    return False
            except ValueError:
                return False
            if base == "*":
                continue
            # base could be a range
            part = base

        # Handle range: 1-5
        if "-" in part:
            range_parts = part.split("-", 1)
            if len(range_parts) != 2:
                return False
            low = _resolve_name(range_parts[0].strip(), name_map)
            high = _resolve_name(range_parts[1].strip(), name_map)
            try:
                low_int = int(low)
                high_int = int(high)
                if not (min_val <= low_int <= max_val and min_val <= high_int <= max_val):
                    return False
            except ValueError:
                return False
            continue

        # Single value
        resolved = _resolve_name(part, name_map)
        try:
            val = int(resolved)
            if not (min_val <= val <= max_val):
                return False
        except ValueError:
            return False

    return True


def validate_cron(expression: str) -> tuple[bool, str]:
    """Validate a 5-field cron expression. Returns (ok, error_message)."""
    fields = expression.strip().split()
    if len(fields) != 5:
        return False, f"Expected 5 fields, got {len(fields)}"

    for i, (name, min_val, max_val, name_map) in enumerate(FIELD_SPECS):
        if not _validate_field(fields[i], min_val, max_val, name_map):
            return False, f"Invalid {name} field: {fields[i]}"

    return True, ""


def is_cron_expression(text: str) -> bool:
    """Check if text looks like a cron expression (5 whitespace-separated tokens)."""
    text = text.strip()
    parts = text.split()
    if len(parts) != 5:
        return False
    # Quick heuristic: each part should be cron-like tokens
    cron_token = re.compile(r'^[\d\*\-,/A-Za-z]+$')
    return all(cron_token.match(p) for p in parts)


# ---------------------------------------------------------------------------
# Next run time calculation (no external deps)
# ---------------------------------------------------------------------------

def _expand_field(field: str, min_val: int, max_val: int, name_map: dict | None) -> set[int]:
    """Expand a cron field into a set of matching integer values."""
    result = set()

    for part in field.split(","):
        part = part.strip()
        step = 1

        if "/" in part:
            base, step_str = part.split("/", 1)
            step = int(step_str)
        else:
            base = part

        if base == "*":
            for v in range(min_val, max_val + 1, step):
                result.add(v)
        elif "-" in base:
            low_s, high_s = base.split("-", 1)
            low = int(_resolve_name(low_s.strip(), name_map))
            high = int(_resolve_name(high_s.strip(), name_map))
            for v in range(low, high + 1, step):
                result.add(v)
        else:
            val = int(_resolve_name(base, name_map))
            if step > 1:
                # e.g., 5/10 means 5,15,25,...
                v = val
                while v <= max_val:
                    result.add(v)
                    v += step
            else:
                result.add(val)

    return result


def next_run_times(expression: str, count: int = 3, from_time: datetime | None = None) -> list[datetime]:
    """Calculate the next `count` run times for a cron expression."""
    fields = expression.strip().split()
    if len(fields) != 5:
        return []

    minutes = _expand_field(fields[0], 0, 59, None)
    hours = _expand_field(fields[1], 0, 23, None)
    days_of_month = _expand_field(fields[2], 1, 31, None)
    months = _expand_field(fields[3], 1, 12, MONTH_NAMES)
    days_of_week = _expand_field(fields[4], 0, 7, DOW_NAMES)

    # Normalize: 7 -> 0 (both mean Sunday)
    if 7 in days_of_week:
        days_of_week.add(0)
        days_of_week.discard(7)

    now = from_time or datetime.now()
    # Start from the next minute
    current = now.replace(second=0, microsecond=0) + timedelta(minutes=1)

    results = []
    # Search up to 2 years ahead (enough for any cron pattern)
    max_iterations = 366 * 24 * 60  # ~1 year in minutes
    iterations = 0

    while len(results) < count and iterations < max_iterations:
        # Python weekday: Monday=0 ... Sunday=6
        # Cron weekday: Sunday=0, Monday=1 ... Saturday=6
        py_dow = current.weekday()
        cron_dow = (py_dow + 1) % 7

        if (current.minute in minutes and
            current.hour in hours and
            current.month in months):

            # Day matching: if both day-of-month and day-of-week are restricted
            # (not *), match either (OR). If only one is restricted, match that one.
            dom_is_star = fields[2] == "*"
            dow_is_star = fields[4] == "*"

            day_match = False
            if dom_is_star and dow_is_star:
                day_match = True
            elif dom_is_star:
                day_match = cron_dow in days_of_week
            elif dow_is_star:
                day_match = current.day in days_of_month
            else:
                # Both specified: OR logic (standard cron behavior)
                day_match = current.day in days_of_month or cron_dow in days_of_week

            if day_match:
                results.append(current)

        # Advance: skip efficiently
        if current.month not in months:
            # Jump to start of next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1, day=1, hour=0, minute=0)
            else:
                current = current.replace(month=current.month + 1, day=1, hour=0, minute=0)
            iterations += 1
            continue

        if current.hour not in hours:
            # Jump to next hour
            current = current.replace(minute=0) + timedelta(hours=1)
            iterations += 1
            continue

        current += timedelta(minutes=1)
        iterations += 1

    return results


# ---------------------------------------------------------------------------
# Output extraction / cleaning
# ---------------------------------------------------------------------------

def _extract_cron(raw: str) -> str:
    """Extract a 5-field cron expression from model output."""
    text = raw.strip()

    if not text:
        return ""

    # Try to find a line that looks like a cron expression
    for line in text.split("\n"):
        line = line.strip().strip("`").strip("'\"")
        # Remove common prefixes
        for prefix in ("Cron:", "cron:", "Expression:", "Output:", "Result:", "Answer:"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()

        if is_cron_expression(line):
            return line

    # Fallback: try the whole text stripped
    stripped = text.strip().strip("`").strip("'\"")
    for prefix in ("Cron:", "cron:", "Expression:", "Output:", "Result:", "Answer:"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):].strip()
    if is_cron_expression(stripped):
        return stripped

    # Last resort: find 5 cron-like tokens anywhere
    tokens = re.findall(r'[\d\*\-,/]+', text)
    if len(tokens) >= 5:
        candidate = " ".join(tokens[:5])
        if is_cron_expression(candidate):
            return candidate

    return text.strip().split("\n")[0].strip()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def generate_cron(description: str) -> str:
    """Generate a cron expression from natural language, with one retry."""
    prompt = GENERATE_TEMPLATE.format(description=description)
    raw = complete(prompt, max_tokens=50, stop_at="\n")
    expression = _extract_cron(raw)

    ok, error = validate_cron(expression)
    if ok:
        return expression

    # Retry once
    retry_prompt = RETRY_TEMPLATE.format(description=description)
    raw2 = complete(retry_prompt, max_tokens=50, stop_at="\n")
    expression2 = _extract_cron(raw2)

    ok2, error2 = validate_cron(expression2)
    if ok2:
        return expression2

    # Return best effort
    if expression:
        return expression
    if expression2:
        return expression2

    print("Error: model failed to generate a valid cron expression", file=sys.stderr)
    sys.exit(1)


def explain_cron(expression: str) -> str:
    """Explain a cron expression in plain English."""
    ok, error = validate_cron(expression)
    if not ok:
        print(f"Error: invalid cron expression: {error}", file=sys.stderr)
        sys.exit(1)

    prompt = EXPLAIN_TEMPLATE.format(expression=expression)
    return complete(prompt, max_tokens=100, stop_at=None).strip()


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def get_input(args) -> str:
    """Get input from file, stdin, or error."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="hone-cron",
        description="Convert natural language to cron expressions (and back)",
        epilog='Example: echo "every weekday at 9am" | hone-cron',
    )
    parser.add_argument(
        "--explain", "-e", action="store_true",
        help="Reverse mode: input a cron expression, output plain English",
    )
    parser.add_argument(
        "--validate", "-v", action="store_true",
        help="Validate the expression and show next 3 run times",
    )
    parser.add_argument(
        "-i", dest="input_file", metavar="FILE",
        help="Read input from file (default: stdin)",
    )
    args = parser.parse_args()

    text = get_input(args).strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    if args.explain:
        # Reverse mode: cron -> English
        if not is_cron_expression(text):
            # Maybe they gave us something close; try to validate
            ok, error = validate_cron(text)
            if not ok:
                print(f"Error: not a valid cron expression: {error}", file=sys.stderr)
                sys.exit(1)
        explanation = explain_cron(text)
        print(explanation)
    else:
        # Forward mode: English -> cron
        expression = generate_cron(text)
        print(expression)

        if args.validate:
            ok, error = validate_cron(expression)
            if not ok:
                print(f"\nWarning: generated expression may be invalid: {error}", file=sys.stderr)
            else:
                print(f"\nValid cron expression: {expression}")
                times = next_run_times(expression)
                if times:
                    print("Next 3 run times:")
                    for t in times:
                        print(f"  {t.strftime('%Y-%m-%d %H:%M (%A)')}")
                else:
                    print("Could not calculate next run times.")


if __name__ == "__main__":
    main()
