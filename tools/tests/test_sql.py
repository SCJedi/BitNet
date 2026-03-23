"""Tests for hone-sql CLI tool.

Two test phases:
1. Unit tests: schema parsing, SQL validation, extraction (no model needed)
2. Integration tests: full pipeline through the model (requires model)
"""

import re
import subprocess
import sys

# Add tools dir to path so we can import directly
sys.path.insert(0, r"C:\Users\ericl\Documents\Projects\BitNet\tools")
from hone_tools.cli.sql import (
    parse_schema,
    format_schema_for_prompt,
    validate_sql,
    validate_against_schema,
    is_sql_query,
    _extract_sql,
    VALID_SQL_STARTS,
)

PYTHON = sys.executable
CMD_PREFIX = [PYTHON, "-m", "hone_tools.cli.sql"]
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


def run_sql(input_text, extra_args=None, timeout=120):
    """Run hone-sql and return (exit_code, stdout, stderr)."""
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

# --- Schema parsing ---
print("\n--- Schema Parsing ---")

schema1 = parse_schema("users(id, name, email, country, created_at)")
check("parse single table", schema1 == {"users": ["id", "name", "email", "country", "created_at"]})

schema2 = parse_schema("users(id, name, email), orders(id, user_id, amount, date)")
check("parse multi table", schema2 == {
    "users": ["id", "name", "email"],
    "orders": ["id", "user_id", "amount", "date"],
})

schema3 = parse_schema("  Users( ID , Name )  ")
check("parse with whitespace", schema3 == {"users": ["id", "name"]})

schema4 = parse_schema("")
check("parse empty string", schema4 == {})

schema5 = parse_schema("no_parens_here")
check("parse no parens", schema5 == {})

schema6 = parse_schema("products(id, name, price, category), inventory(id, product_id, quantity)")
check("parse products+inventory", len(schema6) == 2 and "products" in schema6 and "inventory" in schema6)

# --- format_schema_for_prompt ---
print("\n--- Schema Formatting ---")

formatted = format_schema_for_prompt({"users": ["id", "name"]})
check("format schema has table", "users" in formatted)
check("format schema has cols", "id" in formatted and "name" in formatted)
check("format empty schema", format_schema_for_prompt({}) == "")

# --- SQL validation ---
print("\n--- SQL Validation ---")

# Valid SQL
valid_sqls = [
    ("SELECT * FROM users;", "basic select"),
    ("SELECT id, name FROM users WHERE active = 1;", "select with where"),
    ("INSERT INTO users (name) VALUES ('test');", "insert"),
    ("UPDATE users SET name = 'test' WHERE id = 1;", "update"),
    ("DELETE FROM users WHERE id = 1;", "delete"),
    ("CREATE TABLE test (id INT);", "create table"),
    ("ALTER TABLE users ADD COLUMN email VARCHAR(255);", "alter table"),
    ("DROP TABLE test;", "drop table"),
    ("WITH cte AS (SELECT * FROM users) SELECT * FROM cte;", "with CTE"),
    ("SELECT COUNT(*) FROM users GROUP BY country;", "group by"),
    ("SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id;", "join"),
    ("SELECT * FROM users ORDER BY created_at DESC LIMIT 10;", "order limit"),
    ("SELECT country, COUNT(*) as cnt FROM users GROUP BY country HAVING cnt > 5;", "having"),
    ("SELECT * FROM users WHERE email LIKE '%@gmail.com';", "like"),
    ("SELECT AVG(amount) FROM orders;", "aggregate"),
    ("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);", "subquery"),
]

for sql, desc in valid_sqls:
    ok, err = validate_sql(sql)
    check(f"Valid SQL: {desc}", ok, f"Error: {err}")

# Invalid SQL
invalid_sqls = [
    ("", "empty"),
    ("hello world", "not SQL"),
    ("SELEC * FROM users;", "typo in keyword"),
    ("SELECT * FROM users WHERE (a = 1;", "unbalanced parens"),
    ("SELECT * FROM users WHERE name = 'test;", "unbalanced quotes"),
]

for sql, desc in invalid_sqls:
    ok, err = validate_sql(sql)
    check(f"Invalid SQL: {desc}", not ok, "Should have failed")

# --- Schema validation ---
print("\n--- Schema Validation ---")

test_schema = {"users": ["id", "name", "email"], "orders": ["id", "user_id", "amount"]}

# Table exists
ok1, err1 = validate_sql("SELECT * FROM users;", test_schema)
check("schema: known table", ok1, err1)

# Table doesn't exist
ok2, err2 = validate_sql("SELECT * FROM products;", test_schema)
check("schema: unknown table rejected", not ok2, "Should detect unknown table")
check("schema: error mentions table", "products" in err2.lower() if err2 else False)

# JOIN with known tables
ok3, err3 = validate_sql(
    "SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id;",
    test_schema,
)
check("schema: join known tables", ok3, err3)

# JOIN with unknown table
ok4, err4 = validate_sql(
    "SELECT * FROM users u JOIN payments p ON u.id = p.user_id;",
    test_schema,
)
check("schema: join unknown table", not ok4, "Should detect unknown 'payments'")

# --- is_sql_query ---
print("\n--- is_sql_query ---")

check("is_sql: SELECT", is_sql_query("SELECT * FROM users"))
check("is_sql: INSERT", is_sql_query("INSERT INTO users VALUES (1)"))
check("is_sql: UPDATE", is_sql_query("UPDATE users SET x = 1"))
check("is_sql: DELETE", is_sql_query("DELETE FROM users"))
check("is_sql: WITH", is_sql_query("WITH x AS (SELECT 1) SELECT * FROM x"))
check("is_sql: CREATE", is_sql_query("CREATE TABLE test (id INT)"))
check("not_sql: plain text", not is_sql_query("all users"))
check("not_sql: empty", not is_sql_query(""))
check("not_sql: number", not is_sql_query("42"))

# --- _extract_sql ---
print("\n--- SQL Extraction ---")

# Plain SQL
check("extract plain", _extract_sql("SELECT * FROM users;") == "SELECT * FROM users;")

# With code fences
check("extract code fence",
      _extract_sql("```sql\nSELECT * FROM users;\n```") == "SELECT * FROM users;")

# With prefix
check("extract prefix SQL:",
      _extract_sql("SQL: SELECT * FROM users;") == "SELECT * FROM users;")
check("extract prefix Query:",
      _extract_sql("Query: SELECT * FROM users;") == "SELECT * FROM users;")

# With explanation after
result = _extract_sql("SELECT * FROM users;\n\nThis selects all users from the table.")
check("extract with explanation", result == "SELECT * FROM users;")

# Multi-line SQL
multi = _extract_sql("SELECT u.name, COUNT(o.id)\nFROM users u\nJOIN orders o ON u.id = o.user_id\nGROUP BY u.name;")
check("extract multi-line", "SELECT" in multi and "GROUP BY" in multi)

# Inline backticks
check("extract backticks",
      _extract_sql("`SELECT * FROM users;`") == "SELECT * FROM users;")

# Missing semicolon gets added
check("extract adds semicolon",
      _extract_sql("SELECT * FROM users").endswith(";"))

# Noisy output with SQL embedded
noisy = "Here's the query you requested:\n\nSELECT * FROM users WHERE active = 1;\n\nThis will return all active users."
check("extract from noisy", "SELECT * FROM users WHERE active = 1;" in _extract_sql(noisy))

# Empty
check("extract empty", _extract_sql("") == "")

print()

# ============================================================
# PHASE 2: Integration tests (requires model)
# ============================================================
print("=" * 60)
print("PHASE 2: Integration tests (requires model)")
print("=" * 60)


def check_integration(name, description, extra_args=None,
                       must_be_valid=True, expect_fail=False,
                       check_output=None, check_keywords=None,
                       must_contain_any=None):
    """Run a single integration test."""
    global passed, failed, errors

    print(f"\n  TEST: {name}")
    print(f"    Input: {description!r}")
    if extra_args:
        print(f"    Args: {extra_args}")

    try:
        code, stdout, stderr = run_sql(description, extra_args)
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
            msg = f"FAIL [{name}]: Expected failure but got exit code 0, output: {stdout}"
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
    # For non-explain mode, take first SQL statement
    if not extra_args or "--explain" not in extra_args:
        # Get the SQL part (before any validation output)
        output_lines = output.split("\n")
        output = output_lines[0].strip() if output_lines else output

    print(f"    Output: {output[:200]}{'...' if len(output) > 200 else ''}")

    if must_be_valid and (not extra_args or "--explain" not in extra_args):
        ok, err = validate_sql(output)
        if not ok:
            failed += 1
            msg = f"FAIL [{name}]: Invalid SQL: {err} (output: {output})"
            errors.append(msg)
            print(f"    {msg}")
            return

    if check_keywords:
        output_upper = output.upper()
        missing = [kw for kw in check_keywords if kw.upper() not in output_upper]
        if missing:
            failed += 1
            msg = f"FAIL [{name}]: Missing keywords: {missing}"
            errors.append(msg)
            print(f"    {msg}")
            return

    if must_contain_any:
        output_upper = output.upper()
        found = any(kw.upper() in output_upper for kw in must_contain_any)
        if not found:
            failed += 1
            msg = f"FAIL [{name}]: Must contain one of {must_contain_any}"
            errors.append(msg)
            print(f"    {msg}")
            return

    if check_output:
        if not check_output(stdout):
            failed += 1
            msg = f"FAIL [{name}]: Output check failed"
            errors.append(msg)
            print(f"    {msg}")
            return

    passed += 1
    print(f"    PASS")


# --- BASIC SELECT (1-6) ---
print("\n--- Basic SELECT ---")

check_integration(
    "1. all users",
    "all users",
    check_keywords=["SELECT", "FROM", "users"],
)

check_integration(
    "2. top 10 orders by amount",
    "top 10 orders by amount",
    check_keywords=["SELECT", "FROM", "orders", "ORDER BY", "LIMIT"],
    must_contain_any=["DESC", "amount"],
)

check_integration(
    "3. users who signed up this week",
    "users who signed up this week",
    check_keywords=["SELECT", "FROM", "users"],
    must_contain_any=["DATE", "INTERVAL", "WEEK", "created_at", "signup", "sign_up", "registered", "date_sub", "now", "CURRENT", "7 DAY"],
)

check_integration(
    "4. count of users by country",
    "count of users by country",
    check_keywords=["SELECT", "COUNT", "GROUP BY", "country"],
)

check_integration(
    "5. average order amount",
    "average order amount",
    check_keywords=["SELECT", "AVG"],
    must_contain_any=["amount", "total", "price"],
)

check_integration(
    "6. users with no orders",
    "users with no orders",
    check_keywords=["SELECT", "FROM", "users"],
    must_contain_any=["LEFT JOIN", "NOT IN", "NOT EXISTS"],
)

# --- WITH SCHEMA (7-9) ---
print("\n--- With Schema ---")

check_integration(
    "7. active premium users with schema",
    "active premium users",
    extra_args=["--schema", "users(id, name, plan, is_active)"],
    check_keywords=["SELECT", "FROM", "users"],
    must_contain_any=["is_active", "plan", "premium", "active"],
)

check_integration(
    "8. total revenue per month with schema",
    "total revenue per month",
    extra_args=["--schema", "orders(id, user_id, amount, created_at)"],
    check_keywords=["SELECT", "FROM", "orders"],
    must_contain_any=["SUM", "amount", "GROUP BY", "MONTH", "revenue"],
)

check_integration(
    "9. users who ordered more than 5 times",
    "users who ordered more than 5 times",
    extra_args=["--schema", "users(id, name), orders(id, user_id, amount)"],
    check_keywords=["SELECT", "FROM"],
    must_contain_any=["JOIN", "HAVING", "COUNT", "> 5", ">= 5", ">= 6"],
)

# --- JOINS (10-11) ---
print("\n--- Joins ---")

check_integration(
    "10. users and their orders",
    "users and their orders",
    check_keywords=["SELECT", "FROM"],
    must_contain_any=["JOIN"],
)

check_integration(
    "11. products never ordered",
    "products never ordered",
    check_keywords=["SELECT", "FROM", "products"],
    must_contain_any=["LEFT JOIN", "NOT IN", "NOT EXISTS", "IS NULL"],
)

# --- MUTATIONS (12-14) ---
print("\n--- Mutations ---")

check_integration(
    "12. add column email to users table",
    "add column email to users table",
    check_keywords=["ALTER TABLE", "users"],
    must_contain_any=["ADD", "email"],
)

check_integration(
    "13. delete inactive users",
    "delete inactive users",
    check_keywords=["DELETE", "FROM", "users"],
    must_contain_any=["inactive", "active", "is_active", "status", "DATE", "WHERE"],
)

check_integration(
    "14. update price of product 42 to 19.99",
    "update price of product 42 to 19.99",
    check_keywords=["UPDATE"],
    must_contain_any=["19.99", "42", "price"],
)

# --- EXPLAIN MODE (15) ---
print("\n--- Explain Mode ---")

check_integration(
    "15. explain complex query",
    "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name HAVING COUNT(o.id) > 5",
    extra_args=["--explain"],
    must_be_valid=False,
    check_output=lambda s: len(s) > 10,  # got some English text back
)

# --- DIALECT (16-18) ---
print("\n--- Dialect ---")

check_integration(
    "16. top 10 users (postgres)",
    "top 10 users",
    extra_args=["--dialect", "postgres"],
    check_keywords=["SELECT", "LIMIT"],
)

check_integration(
    "17. top 10 users (mysql)",
    "top 10 users",
    extra_args=["--dialect", "mysql"],
    check_keywords=["SELECT", "LIMIT"],
)

check_integration(
    "18. random user (sqlite)",
    "random user",
    extra_args=["--dialect", "sqlite"],
    check_keywords=["SELECT", "FROM"],
    must_contain_any=["RANDOM", "LIMIT 1", "limit 1"],
)

# --- EDGE CASES (19-22) ---
print("\n--- Edge Cases ---")

check_integration(
    "19. empty input",
    "",
    expect_fail=True,
)

check_integration(
    "20. drop all tables",
    "drop all tables",
    must_be_valid=True,  # tool generates SQL, doesn't execute it
    check_keywords=["DROP"],
)

check_integration(
    "21. vague input - stuff about users",
    "stuff about users",
    must_be_valid=True,  # should still produce valid SQL
    check_keywords=["SELECT", "FROM", "users"],
)

check_integration(
    "22. already SQL - pass through",
    "SELECT * FROM users WHERE active = 1",
    must_be_valid=True,
    check_keywords=["SELECT", "FROM", "users"],
)

# --- VALIDATE FLAG (23) ---
print("\n--- Validate Flag ---")

check_integration(
    "23. generate with validate",
    "all users",
    extra_args=["--validate"],
    check_keywords=["SELECT", "FROM", "users"],
)

# --- ADDITIONAL (24-26) ---
print("\n--- Additional ---")

check_integration(
    "24. complex aggregation",
    "top 5 customers by total spending",
    check_keywords=["SELECT", "FROM", "ORDER BY"],
    must_contain_any=["SUM", "LIMIT", "amount", "spending", "total"],
)

check_integration(
    "25. create table",
    "create a users table with id, name, email, and created_at",
    check_keywords=["CREATE TABLE"],
    must_contain_any=["users", "id", "name", "email"],
)

check_integration(
    "26. insert row",
    "insert a new user named Alice with email alice@example.com",
    check_keywords=["INSERT"],
    must_contain_any=["Alice", "alice@example.com"],
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
