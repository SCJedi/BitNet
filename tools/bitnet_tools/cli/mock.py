"""bt-mock: Generate mock/fake data from type descriptions using BitNet."""

import argparse
import copy
import io
import json
import random
import re
import sys

from ..engine import complete


def get_input(args):
    """Read input from file or stdin."""
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


def parse_type_description(text: str) -> tuple[str, list[str]]:
    """Parse 'TypeName(field1, field2, ...)' into (type_name, [fields]).

    Returns (type_name, fields) or ("Record", []) if parsing fails.
    """
    text = text.strip()
    if not text:
        return ("Record", [])

    m = re.match(r"(\w+)\s*\(([^)]*)\)", text)
    if not m:
        return ("Record", [])

    type_name = m.group(1).strip()
    fields_str = m.group(2).strip()
    if not fields_str:
        return (type_name, [])

    fields = [f.strip() for f in fields_str.split(",") if f.strip()]
    return (type_name, fields)


def generate_seed_records(type_name: str, fields: list[str], count: int) -> list[dict]:
    """Use the model to generate seed records (up to 3) as JSON."""
    n = min(count, 3)
    fields_str = ", ".join(fields)

    if n == 1:
        prompt = (
            f"Generate 1 realistic fake {type_name} record as a JSON object "
            f"with these fields: {fields_str}.\n"
            f"Output ONLY valid JSON, no explanation.\n\nJSON:"
        )
    else:
        prompt = (
            f"Generate {n} realistic fake {type_name} records as a JSON array "
            f"with these fields: {fields_str}.\n"
            f"Output ONLY a valid JSON array, no explanation.\n\nJSON:"
        )

    # Allow enough tokens for the response
    max_tokens = 150 * n
    raw = complete(prompt, max_tokens=max_tokens, stop_at=None)

    records = parse_json_records(raw, fields)

    # If parsing returned nothing, build a minimal fallback
    if not records:
        records = [_fallback_record(fields)]

    return records[:n]


def parse_json_records(text: str, fields: list[str]) -> list[dict]:
    """Extract JSON records (object or array) from model output.

    Always returns a list of dicts. Each dict is validated to have all fields.
    """
    text = text.strip()

    # Try to find a JSON array first
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start != -1 and arr_end > arr_start:
        try:
            data = json.loads(text[arr_start:arr_end + 1])
            if isinstance(data, list):
                return [_ensure_fields(r, fields) for r in data if isinstance(r, dict)]
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        fragment = text[obj_start:obj_end + 1]
        # Could be multiple objects (not in an array)
        # Try wrapping in array brackets
        try:
            data = json.loads(fragment)
            if isinstance(data, dict):
                return [_ensure_fields(data, fields)]
        except json.JSONDecodeError:
            pass

        # Try wrapping separated objects as array
        try:
            wrapped = "[" + fragment + "]"
            # Fix missing commas between objects: }{ -> },{
            wrapped = re.sub(r"\}\s*\{", "},{", wrapped)
            data = json.loads(wrapped)
            if isinstance(data, list):
                return [_ensure_fields(r, fields) for r in data if isinstance(r, dict)]
        except json.JSONDecodeError:
            pass

    return []


def _ensure_fields(record: dict, fields: list[str]) -> dict:
    """Ensure record has all required fields, adding None for missing ones."""
    result = {}
    for f in fields:
        # Try case-insensitive match
        matched = False
        for k, v in record.items():
            if k.lower() == f.lower():
                result[f] = v
                matched = True
                break
        if not matched:
            result[f] = None
    return result


def _fallback_record(fields: list[str]) -> dict:
    """Generate a minimal fallback record with placeholder values."""
    record = {}
    for f in fields:
        fl = f.lower()
        if "name" in fl:
            record[f] = "Alice Smith"
        elif "email" in fl:
            record[f] = "alice@example.com"
        elif "age" in fl:
            record[f] = 30
        elif "price" in fl or "cost" in fl or "amount" in fl:
            record[f] = 29.99
        elif "stock" in fl or "active" in fl or "available" in fl or "enabled" in fl:
            record[f] = True
        elif "date" in fl or "created" in fl or "updated" in fl:
            record[f] = "2024-01-15"
        elif "id" in fl:
            record[f] = 1
        elif "phone" in fl:
            record[f] = "555-0100"
        elif "address" in fl or "street" in fl:
            record[f] = "123 Main St"
        elif "city" in fl:
            record[f] = "Springfield"
        elif "country" in fl:
            record[f] = "US"
        elif "zip" in fl or "postal" in fl:
            record[f] = "12345"
        elif "url" in fl or "website" in fl:
            record[f] = "https://example.com"
        elif "description" in fl or "bio" in fl or "summary" in fl:
            record[f] = "A sample description"
        elif "category" in fl or "type" in fl:
            record[f] = "general"
        elif "count" in fl or "quantity" in fl:
            record[f] = 10
        elif "rating" in fl or "score" in fl:
            record[f] = 4.5
        else:
            record[f] = f"sample_{f}"
    return record


# --- Programmatic variation for count > 3 ---

_FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Noah", "Olivia", "Peter",
    "Quinn", "Ruby", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zane",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
]

_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "proton.me",
    "company.com", "work.org", "mail.io", "fastmail.com", "icloud.com",
]

_CATEGORIES = [
    "Electronics", "Books", "Clothing", "Home", "Sports", "Toys",
    "Food", "Beauty", "Health", "Automotive", "Garden", "Music",
]

_CITIES = [
    "New York", "London", "Tokyo", "Paris", "Berlin", "Sydney",
    "Toronto", "Mumbai", "Seoul", "Chicago", "Austin", "Denver",
]


def vary_record(record: dict, index: int) -> dict:
    """Programmatically vary a seed record to create a new unique record."""
    varied = copy.deepcopy(record)
    rng = random.Random(index * 7 + 42)

    for key, val in varied.items():
        kl = key.lower()

        if isinstance(val, bool):
            varied[key] = rng.choice([True, False])

        elif isinstance(val, str):
            if "email" in kl:
                first = rng.choice(_FIRST_NAMES).lower()
                last = rng.choice(_LAST_NAMES).lower()
                domain = rng.choice(_DOMAINS)
                varied[key] = f"{first}.{last}@{domain}"
            elif "name" in kl:
                varied[key] = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
            elif "city" in kl:
                varied[key] = rng.choice(_CITIES)
            elif "category" in kl or "type" in kl:
                varied[key] = rng.choice(_CATEGORIES)
            elif "phone" in kl:
                varied[key] = f"555-{rng.randint(1000, 9999)}"
            elif "date" in kl or "created" in kl or "updated" in kl:
                y = rng.randint(2020, 2025)
                m = rng.randint(1, 12)
                d = rng.randint(1, 28)
                varied[key] = f"{y}-{m:02d}-{d:02d}"
            elif "address" in kl or "street" in kl:
                varied[key] = f"{rng.randint(100, 9999)} {rng.choice(['Main', 'Oak', 'Elm', 'Pine', 'Cedar', 'Maple'])} St"
            elif "url" in kl or "website" in kl:
                varied[key] = f"https://{rng.choice(_LAST_NAMES).lower()}.com"
            elif "country" in kl:
                varied[key] = rng.choice(["US", "UK", "CA", "AU", "DE", "FR", "JP", "BR", "IN", "KR"])
            elif "zip" in kl or "postal" in kl:
                varied[key] = f"{rng.randint(10000, 99999)}"
            elif "description" in kl or "bio" in kl or "summary" in kl:
                # Keep the original but add minor variation
                varied[key] = val
            else:
                # Generic string: keep as-is (better than mangling)
                pass

        elif isinstance(val, (int, float)):
            if "id" in kl:
                varied[key] = index + 1
            elif "age" in kl:
                varied[key] = rng.randint(18, 80)
            elif "price" in kl or "cost" in kl or "amount" in kl or "salary" in kl:
                # Vary by +/- 30%
                base = float(val) if val else 10.0
                factor = rng.uniform(0.7, 1.3)
                varied[key] = round(base * factor, 2)
            elif "count" in kl or "quantity" in kl:
                varied[key] = rng.randint(1, 100)
            elif "rating" in kl or "score" in kl:
                varied[key] = round(rng.uniform(1.0, 5.0), 1)
            else:
                # Generic number: vary by +/- 30%
                if isinstance(val, int):
                    varied[key] = max(0, val + rng.randint(-max(1, abs(val) // 3), max(1, abs(val) // 3)))
                else:
                    factor = rng.uniform(0.7, 1.3)
                    varied[key] = round(val * factor, 2)

    return varied


def generate_records(type_name: str, fields: list[str], count: int) -> list[dict]:
    """Generate the requested number of records.

    Uses the model for up to 3 seed records, then varies programmatically.
    """
    seeds = generate_seed_records(type_name, fields, count)

    if count <= len(seeds):
        return seeds[:count]

    # Expand programmatically
    records = list(seeds)
    for i in range(len(seeds), count):
        base = seeds[i % len(seeds)]
        records.append(vary_record(base, i))

    return records


# --- Output formatting ---

def format_json(records: list[dict]) -> str:
    """Format records as JSON (object if 1, array if many)."""
    if len(records) == 1:
        return json.dumps(records[0], indent=2)
    return json.dumps(records, indent=2)


def format_csv(records: list[dict]) -> str:
    """Format records as CSV with headers."""
    if not records:
        return ""

    import csv

    fields = list(records[0].keys())
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(fields)
    for r in records:
        writer.writerow([r.get(f, "") for f in fields])
    return output.getvalue().strip()


def format_sql_insert(records: list[dict], table_name: str) -> str:
    """Format records as SQL INSERT statements."""
    if not records:
        return ""

    fields = list(records[0].keys())
    cols = ", ".join(fields)
    lines = []
    for r in records:
        vals = []
        for f in fields:
            v = r.get(f)
            if v is None:
                vals.append("NULL")
            elif isinstance(v, bool):
                vals.append("TRUE" if v else "FALSE")
            elif isinstance(v, (int, float)):
                vals.append(str(v))
            else:
                escaped = str(v).replace("'", "''")
                vals.append(f"'{escaped}'")
        vals_str = ", ".join(vals)
        lines.append(f"INSERT INTO {table_name} ({cols}) VALUES ({vals_str});")

    return "\n".join(lines)


FORMATTERS = {
    "json": lambda records, table: format_json(records),
    "csv": lambda records, table: format_csv(records),
    "sql-insert": lambda records, table: format_sql_insert(records, table),
}


def main():
    parser = argparse.ArgumentParser(
        prog="bt-mock",
        description="Generate mock/fake data from type descriptions using a local BitNet model",
        epilog='Example: echo "User(name, email, age)" | bt-mock --count 5',
    )
    parser.add_argument(
        "--count", "-n", type=int, default=1,
        help="Number of records to generate (default: 1)",
    )
    parser.add_argument(
        "--format", "-f", dest="fmt", choices=list(FORMATTERS.keys()),
        default="json", help="Output format (default: json)",
    )
    parser.add_argument(
        "--table", "-t", default=None,
        help="Table name for sql-insert format (default: derived from type name)",
    )
    parser.add_argument("--input", "-i", help="Input file (default: stdin)")
    args = parser.parse_args()

    if args.count < 1:
        print("Error: --count must be at least 1", file=sys.stderr)
        sys.exit(1)

    text = get_input(args).strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    type_name, fields = parse_type_description(text)
    if not fields:
        print(f"Error: could not parse fields from: {text!r}", file=sys.stderr)
        print("Expected format: TypeName(field1, field2, ...)", file=sys.stderr)
        sys.exit(1)

    # Determine table name for SQL format
    table_name = args.table or type_name.lower() + "s"

    records = generate_records(type_name, fields, args.count)

    formatter = FORMATTERS[args.fmt]
    output = formatter(records, table_name)
    print(output)


if __name__ == "__main__":
    main()
