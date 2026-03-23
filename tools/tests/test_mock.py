"""Tests for bt-mock CLI tool.

Two test phases:
1. Unit tests: type parsing, JSON parsing, variation, formatting (no model needed)
2. Integration tests: full pipeline through the model (requires BitNet)
"""

import json
import subprocess
import sys

# Add tools dir to path so we can import directly
sys.path.insert(0, r"C:\Users\ericl\Documents\Projects\BitNet\tools")
from bitnet_tools.cli.mock import (
    parse_type_description,
    parse_json_records,
    _ensure_fields,
    _fallback_record,
    vary_record,
    format_json,
    format_csv,
    format_sql_insert,
)

PYTHON = sys.executable
CMD_PREFIX = [PYTHON, "-m", "bitnet_tools.cli.mock"]
TOOLS_DIR = r"C:\Users\ericl\Documents\Projects\BitNet\tools"

passed = 0
failed = 0
errors = []


def check(name, ok, detail=""):
    global passed, failed, errors
    if ok:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        msg = f"FAIL: {name}" + (f" -- {detail}" if detail else "")
        errors.append(msg)
        print(f"  {msg}")


def run_mock(input_text, extra_args=None, timeout=120):
    """Run bt-mock and return (exit_code, stdout, stderr)."""
    cmd = CMD_PREFIX + (extra_args or [])
    result = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=TOOLS_DIR,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# ============================================================
# PHASE 1: Unit tests (no model)
# ============================================================
print("=" * 60)
print("PHASE 1: Unit tests (no model needed)")
print("=" * 60)

# --- Type description parsing ---
print("\n--- Type Description Parsing ---")

name1, fields1 = parse_type_description("User(name, email, age)")
check("parse basic type", name1 == "User" and fields1 == ["name", "email", "age"])

name2, fields2 = parse_type_description("Product(name, price, category, in_stock)")
check("parse product type",
      name2 == "Product" and fields2 == ["name", "price", "category", "in_stock"])

name3, fields3 = parse_type_description("  Order ( id , user_id , total )  ")
check("parse with whitespace",
      name3 == "Order" and fields3 == ["id", "user_id", "total"])

name4, fields4 = parse_type_description("")
check("parse empty string", name4 == "Record" and fields4 == [])

name5, fields5 = parse_type_description("no_parens")
check("parse no parens", fields5 == [])

name6, fields6 = parse_type_description("Employee()")
check("parse empty parens", name6 == "Employee" and fields6 == [])

name7, fields7 = parse_type_description("X(a)")
check("parse single field", name7 == "X" and fields7 == ["a"])

# --- JSON record parsing ---
print("\n--- JSON Record Parsing ---")

fields = ["name", "email", "age"]

r1 = parse_json_records('{"name": "Alice", "email": "a@b.com", "age": 30}', fields)
check("parse single object", len(r1) == 1 and r1[0]["name"] == "Alice")

r2 = parse_json_records(
    '[{"name": "Alice", "email": "a@b.com", "age": 30}, '
    '{"name": "Bob", "email": "b@b.com", "age": 25}]',
    fields,
)
check("parse array of objects", len(r2) == 2 and r2[1]["name"] == "Bob")

r3 = parse_json_records("no json here", fields)
check("parse no JSON", r3 == [])

r4 = parse_json_records("", fields)
check("parse empty string", r4 == [])

r5 = parse_json_records(
    'Here is the data: {"name": "Charlie", "email": "c@c.com", "age": 40} Done.',
    fields,
)
check("parse JSON embedded in text", len(r5) == 1 and r5[0]["name"] == "Charlie")

# Case-insensitive field matching
r6 = parse_json_records('{"Name": "Diana", "EMAIL": "d@d.com", "Age": 35}', fields)
check("parse case-insensitive fields",
      len(r6) == 1 and r6[0]["name"] == "Diana" and r6[0]["email"] == "d@d.com")

# --- Ensure fields ---
print("\n--- Field Validation ---")

e1 = _ensure_fields({"name": "Alice", "extra": "junk"}, ["name", "email"])
check("ensure adds missing fields", e1 == {"name": "Alice", "email": None})

e2 = _ensure_fields({}, ["x", "y"])
check("ensure all missing", e2 == {"x": None, "y": None})

# --- Fallback record ---
print("\n--- Fallback Records ---")

fb1 = _fallback_record(["name", "email", "age"])
check("fallback has name", isinstance(fb1["name"], str) and len(fb1["name"]) > 0)
check("fallback has email", "@" in fb1["email"])
check("fallback has age", isinstance(fb1["age"], int))

fb2 = _fallback_record(["price", "in_stock", "created_at"])
check("fallback price is number", isinstance(fb2["price"], (int, float)))
check("fallback in_stock is bool", isinstance(fb2["in_stock"], bool))
check("fallback date is string", isinstance(fb2["created_at"], str) and "-" in fb2["created_at"])

fb3 = _fallback_record(["unknown_field"])
check("fallback unknown field", isinstance(fb3["unknown_field"], str))

# --- Record variation ---
print("\n--- Record Variation ---")

seed = {"name": "Alice Smith", "email": "alice@example.com", "age": 30, "active": True}
v1 = vary_record(seed, 1)
v2 = vary_record(seed, 2)
check("vary changes name", v1["name"] != seed["name"] or v2["name"] != seed["name"],
      f"v1={v1['name']}, v2={v2['name']}")
check("vary changes email", "@" in v1["email"] and v1["email"] != seed["email"])
check("vary age is int", isinstance(v1["age"], int) and 18 <= v1["age"] <= 80)
check("vary active is bool", isinstance(v1["active"], bool))

# Deterministic: same index -> same result
v3a = vary_record(seed, 5)
v3b = vary_record(seed, 5)
check("vary deterministic", v3a == v3b)

# Different indices -> different results (with high probability)
v4a = vary_record(seed, 10)
v4b = vary_record(seed, 11)
check("vary different indices differ",
      v4a["name"] != v4b["name"] or v4a["email"] != v4b["email"])

# Price variation
price_seed = {"name": "Widget", "price": 100.0}
pv = vary_record(price_seed, 1)
check("vary price is float", isinstance(pv["price"], float))
check("vary price in range", 50 <= pv["price"] <= 150,
      f"price={pv['price']}")

# --- Output formatting: JSON ---
print("\n--- JSON Formatting ---")

records_1 = [{"name": "Alice", "age": 30}]
j1 = format_json(records_1)
check("json single is object", j1.startswith("{"))
parsed_j1 = json.loads(j1)
check("json single valid", parsed_j1["name"] == "Alice")

records_2 = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
j2 = format_json(records_2)
check("json multiple is array", j2.startswith("["))
parsed_j2 = json.loads(j2)
check("json multiple valid", len(parsed_j2) == 2)

# --- Output formatting: CSV ---
print("\n--- CSV Formatting ---")

c1 = format_csv([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
lines = c1.replace("\r\n", "\n").split("\n")
check("csv has header", lines[0] == "name,age")
check("csv has rows", len(lines) == 3)
check("csv first row", "Alice" in lines[1])

c2 = format_csv([])
check("csv empty", c2 == "")

# --- Output formatting: SQL INSERT ---
print("\n--- SQL INSERT Formatting ---")

s1 = format_sql_insert(
    [{"name": "Alice", "age": 30}],
    "users",
)
check("sql has INSERT INTO", "INSERT INTO users" in s1)
check("sql has values", "'Alice'" in s1 and "30" in s1)

s2 = format_sql_insert(
    [{"name": "Bob", "active": True}, {"name": "Carol", "active": False}],
    "people",
)
check("sql multiple inserts", s2.count("INSERT INTO") == 2)
check("sql bool TRUE", "TRUE" in s2)
check("sql bool FALSE", "FALSE" in s2)

s3 = format_sql_insert([{"val": None}], "t")
check("sql NULL", "NULL" in s3)

s4 = format_sql_insert([], "t")
check("sql empty", s4 == "")

# SQL injection defense: single quotes escaped
s5 = format_sql_insert([{"name": "O'Brien"}], "users")
check("sql escapes quotes", "O''Brien" in s5)

print()

# ============================================================
# PHASE 2: Integration tests (requires model)
# ============================================================
print("=" * 60)
print("PHASE 2: Integration tests (requires model)")
print("=" * 60)


def check_integration(name, input_text, extra_args=None,
                       expect_fail=False, check_output=None,
                       check_fields=None, check_count=None,
                       check_format=None):
    """Run a single integration test."""
    global passed, failed, errors

    print(f"\n  TEST: {name}")
    print(f"    Input: {input_text!r}")
    if extra_args:
        print(f"    Args: {extra_args}")

    try:
        code, stdout, stderr = run_mock(input_text, extra_args)
    except subprocess.TimeoutExpired:
        failed += 1
        msg = f"FAIL [{name}]: Timed out"
        errors.append(msg)
        print(f"    {msg}")
        return
    except Exception as e:
        failed += 1
        msg = f"FAIL [{name}]: Exception: {e}"
        errors.append(msg)
        print(f"    {msg}")
        return

    if expect_fail:
        if code != 0:
            passed += 1
            print(f"    PASS (expected failure, got exit code {code})")
        else:
            failed += 1
            msg = f"FAIL [{name}]: Expected failure but got exit code 0, output: {stdout[:200]}"
            errors.append(msg)
            print(f"    {msg}")
        return

    if code != 0:
        failed += 1
        msg = f"FAIL [{name}]: Exit code {code}. Stderr: {stderr}"
        errors.append(msg)
        print(f"    {msg}")
        return

    output = stdout.strip()
    print(f"    Output: {output[:300]}{'...' if len(output) > 300 else ''}")

    # Parse output as JSON for validation (if JSON format)
    if not extra_args or not any(a in ["csv", "sql-insert"] for a in (extra_args or [])):
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            failed += 1
            msg = f"FAIL [{name}]: Output is not valid JSON: {output[:200]}"
            errors.append(msg)
            print(f"    {msg}")
            return

        records = data if isinstance(data, list) else [data]

        if check_fields:
            for i, rec in enumerate(records):
                missing = [f for f in check_fields if f not in rec]
                if missing:
                    failed += 1
                    msg = f"FAIL [{name}]: Record {i} missing fields: {missing}"
                    errors.append(msg)
                    print(f"    {msg}")
                    return

        if check_count is not None:
            actual = len(records)
            if actual != check_count:
                failed += 1
                msg = f"FAIL [{name}]: Expected {check_count} records, got {actual}"
                errors.append(msg)
                print(f"    {msg}")
                return

    if check_format:
        if not check_format(output):
            failed += 1
            msg = f"FAIL [{name}]: Format check failed"
            errors.append(msg)
            print(f"    {msg}")
            return

    if check_output:
        if not check_output(output):
            failed += 1
            msg = f"FAIL [{name}]: Output check failed"
            errors.append(msg)
            print(f"    {msg}")
            return

    passed += 1
    print(f"    PASS")


# --- Single record ---
print("\n--- Single Record ---")

check_integration(
    "1. single user",
    "User(name, email, age)",
    check_fields=["name", "email", "age"],
    check_count=1,
)

check_integration(
    "2. single product",
    "Product(name, price, category)",
    check_fields=["name", "price", "category"],
    check_count=1,
)

# --- Multiple records ---
print("\n--- Multiple Records ---")

check_integration(
    "3. three users",
    "User(name, email, age)",
    extra_args=["--count", "3"],
    check_fields=["name", "email", "age"],
    check_count=3,
)

check_integration(
    "4. five products",
    "Product(name, price, category, in_stock)",
    extra_args=["-n", "5"],
    check_fields=["name", "price", "category", "in_stock"],
    check_count=5,
)

check_integration(
    "5. ten records (programmatic expansion)",
    "Employee(name, email, department, salary)",
    extra_args=["--count", "10"],
    check_fields=["name", "email", "department", "salary"],
    check_count=10,
)

# --- Various field types ---
print("\n--- Various Field Types ---")

check_integration(
    "6. dates and booleans",
    "Event(title, date, location, is_public)",
    check_fields=["title", "date", "location", "is_public"],
    check_count=1,
)

check_integration(
    "7. addresses",
    "Address(street, city, state, zip_code, country)",
    check_fields=["street", "city", "state", "zip_code", "country"],
    check_count=1,
)

check_integration(
    "8. financial data",
    "Transaction(id, amount, currency, timestamp, status)",
    check_fields=["id", "amount", "currency", "timestamp", "status"],
    check_count=1,
)

# --- Output formats ---
print("\n--- Output Formats ---")

check_integration(
    "9. CSV format",
    "User(name, email, age)",
    extra_args=["--format", "csv", "--count", "3"],
    check_format=lambda s: s.count("\n") >= 3 and "name" in s.split("\n")[0],
)

check_integration(
    "10. SQL INSERT format",
    "User(name, email, age)",
    extra_args=["--format", "sql-insert", "--count", "2"],
    check_format=lambda s: "INSERT INTO" in s and s.count("INSERT INTO") == 2,
)

check_integration(
    "11. SQL INSERT with custom table",
    "User(name, email, age)",
    extra_args=["--format", "sql-insert", "--table", "app_users", "--count", "1"],
    check_format=lambda s: "INSERT INTO app_users" in s,
)

check_integration(
    "12. JSON format explicit",
    "User(name, email)",
    extra_args=["--format", "json"],
    check_fields=["name", "email"],
    check_count=1,
)

# --- Edge cases ---
print("\n--- Edge Cases ---")

check_integration(
    "13. empty input",
    "",
    expect_fail=True,
)

check_integration(
    "14. no parens",
    "just a random string",
    expect_fail=True,
)

check_integration(
    "15. single field",
    "Color(hex_code)",
    check_fields=["hex_code"],
    check_count=1,
)

check_integration(
    "16. many fields",
    "Person(first_name, last_name, email, phone, address, city, state, zip_code, country, birthday)",
    check_fields=["first_name", "last_name", "email", "phone", "address",
                   "city", "state", "zip_code", "country", "birthday"],
    check_count=1,
)

check_integration(
    "17. count=1 explicit",
    "Item(name, weight)",
    extra_args=["--count", "1"],
    check_fields=["name", "weight"],
    check_count=1,
)

# --- Large count ---
print("\n--- Large Count ---")

check_integration(
    "18. 20 records",
    "User(name, email, age)",
    extra_args=["--count", "20"],
    check_fields=["name", "email", "age"],
    check_count=20,
)


# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'='*60}")

if errors:
    print("\nFAILURES:")
    for e in errors:
        print(f"  - {e}")

sys.exit(0 if failed == 0 else 1)
