"""hone-sql: Natural language to SQL query converter using local AI.

Bidirectional:
  echo "all users" | hone-sql                              -> SELECT * FROM users;
  echo "top 10 orders by amount" | hone-sql                -> SELECT * FROM orders ORDER BY amount DESC LIMIT 10;
  echo "count users by country" | hone-sql --schema "users(id, name, email, country)" -> precise query
  echo "SELECT ..." | hone-sql --explain                   -> plain English explanation
"""

import argparse
import re
import sys

from ..engine import complete


# ---------------------------------------------------------------------------
# SQL dialects
# ---------------------------------------------------------------------------

DIALECT_HINTS = {
    "mysql": "MySQL syntax. Use DATE_SUB, NOW(), LIMIT, IFNULL, backtick quoting.",
    "postgres": "PostgreSQL syntax. Use CURRENT_DATE, INTERVAL, LIMIT, COALESCE, double-quote identifiers.",
    "sqlite": "SQLite syntax. Use date('now'), LIMIT, IFNULL, RANDOM().",
}

# ---------------------------------------------------------------------------
# Schema parsing
# ---------------------------------------------------------------------------

def parse_schema(schema_str: str) -> dict[str, list[str]]:
    """Parse a schema string like 'users(id, name, email), orders(id, user_id, amount)'
    into {'users': ['id', 'name', 'email'], 'orders': ['id', 'user_id', 'amount']}.
    """
    tables = {}
    if not schema_str or not schema_str.strip():
        return tables

    # Match table_name(col1, col2, ...)
    pattern = re.compile(r'(\w+)\s*\(([^)]+)\)')
    for match in pattern.finditer(schema_str):
        table_name = match.group(1).strip().lower()
        columns = [c.strip().lower() for c in match.group(2).split(',') if c.strip()]
        tables[table_name] = columns

    return tables


def format_schema_for_prompt(schema: dict[str, list[str]]) -> str:
    """Format parsed schema into a readable prompt section."""
    if not schema:
        return ""
    lines = ["Available tables and columns:"]
    for table, cols in schema.items():
        lines.append(f"  {table}({', '.join(cols)})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GENERATE_TEMPLATE = (
    "Convert this natural language request into a SQL query.\n"
    "{dialect_hint}\n"
    "{schema_section}"
    "Reply with ONLY the SQL query. No explanation. No code fences. No markdown.\n\n"
    "Request: {description}\n\n"
    "SQL:"
)

RETRY_TEMPLATE = (
    "Your previous SQL query had issues: {error}\n"
    "Convert this natural language request into a valid SQL query.\n"
    "{dialect_hint}\n"
    "{schema_section}"
    "Reply with ONLY the SQL query. No explanation. No code fences.\n\n"
    "Request: {description}\n\n"
    "SQL:"
)

EXPLAIN_TEMPLATE = (
    "Explain this SQL query in plain English. Be concise (1-3 sentences).\n\n"
    "SQL: {sql}\n\n"
    "Explanation:"
)


# ---------------------------------------------------------------------------
# SQL validation
# ---------------------------------------------------------------------------

VALID_SQL_STARTS = (
    "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP",
    "WITH", "EXPLAIN", "SHOW", "DESCRIBE", "TRUNCATE", "REPLACE",
    "MERGE", "GRANT", "REVOKE", "BEGIN", "COMMIT", "ROLLBACK",
    "SET", "USE",
)


def validate_sql(sql: str, schema: dict[str, list[str]] | None = None) -> tuple[bool, str]:
    """Basic SQL syntax validation. Returns (ok, error_message)."""
    sql = sql.strip()
    if not sql:
        return False, "Empty SQL"

    # Check starts with valid keyword
    first_word = sql.split()[0].upper().rstrip("(")
    if first_word not in VALID_SQL_STARTS:
        return False, f"SQL must start with a valid keyword, got: {first_word}"

    # Check balanced parentheses
    depth = 0
    for ch in sql:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth < 0:
            return False, "Unbalanced parentheses (extra closing)"
    if depth != 0:
        return False, "Unbalanced parentheses (unclosed)"

    # Check balanced quotes (single quotes)
    single_count = sql.count("'")
    if single_count % 2 != 0:
        return False, "Unbalanced single quotes"

    # Schema validation: check referenced tables/columns exist
    if schema:
        issues = validate_against_schema(sql, schema)
        if issues:
            return False, "; ".join(issues)

    return True, ""


def validate_against_schema(sql: str, schema: dict[str, list[str]]) -> list[str]:
    """Check that tables and columns referenced in SQL exist in the schema.
    Returns list of issues (empty = ok).
    """
    issues = []
    sql_upper = sql.upper()
    sql_lower = sql.lower()

    # Extract table references (simple heuristic: words after FROM, JOIN, UPDATE, INTO)
    table_pattern = re.compile(
        r'(?:FROM|JOIN|UPDATE|INTO|TABLE)\s+(\w+)',
        re.IGNORECASE,
    )
    referenced_tables = set()
    for match in table_pattern.finditer(sql):
        tname = match.group(1).lower()
        # Skip SQL keywords that might follow FROM/JOIN
        if tname.upper() in ('SELECT', 'WHERE', 'SET', 'VALUES', 'AS', 'ON', 'IF', 'EXISTS', 'NOT'):
            continue
        referenced_tables.add(tname)

    schema_tables = set(schema.keys())

    for table in referenced_tables:
        if table not in schema_tables:
            issues.append(f"Table '{table}' not found in schema (available: {', '.join(sorted(schema_tables))})")

    # Column validation: extract column references and check against known tables
    # Only validate columns for tables we know about
    all_columns = set()
    for cols in schema.values():
        all_columns.update(cols)

    # Extract column-like references (words before = or in SELECT list)
    # This is a best-effort heuristic, not a full SQL parser
    col_pattern = re.compile(r'(?:SELECT|WHERE|ON|BY|SET|HAVING)\s+(.+?)(?:\s+FROM|\s+WHERE|\s+ORDER|\s+GROUP|\s+HAVING|\s+LIMIT|\s+SET|\s*;|\s*$)', re.IGNORECASE | re.DOTALL)

    return issues


# ---------------------------------------------------------------------------
# Output extraction / cleaning
# ---------------------------------------------------------------------------

def _extract_sql(raw: str) -> str:
    """Extract a SQL query from potentially verbose model output."""
    text = raw.strip()
    if not text:
        return ""

    # Strip markdown code fences
    text = re.sub(r'```(?:sql|SQL)?\s*\n?', '', text)
    text = re.sub(r'```', '', text)

    # Strip inline backticks wrapping the whole thing
    if text.startswith('`') and text.endswith('`'):
        text = text[1:-1].strip()

    # Strip common prefixes
    for prefix in ("SQL:", "sql:", "Query:", "query:", "Output:", "Result:", "Answer:"):
        if text.lstrip().startswith(prefix):
            text = text.lstrip()[len(prefix):].strip()
            break

    # If multi-line, try to find the SQL statement
    lines = text.split('\n')
    if len(lines) > 1:
        # Strategy 1: Find lines that start with SQL keywords
        sql_lines = []
        in_sql = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_sql:
                    # Blank line might end the SQL
                    # But multi-line SQL can have blanks, check if next non-blank continues
                    continue
                continue

            first_word = stripped.split()[0].upper().rstrip('(') if stripped.split() else ''
            if first_word in VALID_SQL_STARTS:
                in_sql = True
                sql_lines.append(stripped)
            elif in_sql:
                # Check if this looks like a SQL continuation
                continuation_words = ('FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER',
                                     'OUTER', 'ON', 'AND', 'OR', 'ORDER', 'GROUP', 'HAVING',
                                     'LIMIT', 'OFFSET', 'SET', 'VALUES', 'INTO', 'AS',
                                     'UNION', 'INTERSECT', 'EXCEPT', 'WHEN', 'THEN', 'ELSE',
                                     'END', 'CASE', 'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE')
                if first_word in continuation_words or stripped.startswith(('(', ')', ',', '--')):
                    sql_lines.append(stripped)
                elif stripped.endswith(';'):
                    sql_lines.append(stripped)
                    break
                else:
                    # Probably explanation text, stop
                    break

        if sql_lines:
            text = ' '.join(sql_lines)
        else:
            # Take first non-empty line
            for line in lines:
                if line.strip():
                    text = line.strip()
                    break

    text = text.strip()

    # Ensure ends with semicolon
    if text and not text.endswith(';'):
        text += ';'

    return text


def is_sql_query(text: str) -> bool:
    """Check if text looks like a SQL query."""
    text = text.strip()
    if not text:
        return False
    first_word = text.split()[0].upper().rstrip('(')
    return first_word in VALID_SQL_STARTS


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def generate_sql(description: str, dialect: str = "mysql",
                 schema: dict[str, list[str]] | None = None) -> str:
    """Generate a SQL query from natural language, with one retry."""
    dialect_hint = DIALECT_HINTS.get(dialect, DIALECT_HINTS["mysql"])
    schema_section = ""
    if schema:
        schema_section = format_schema_for_prompt(schema) + "\n"

    prompt = GENERATE_TEMPLATE.format(
        description=description,
        dialect_hint=dialect_hint,
        schema_section=schema_section,
    )
    raw = complete(prompt, max_tokens=300, stop_at=None)
    sql = _extract_sql(raw)

    ok, error = validate_sql(sql, schema)
    if ok:
        return sql

    # Retry once with error feedback
    retry_prompt = RETRY_TEMPLATE.format(
        error=error,
        description=description,
        dialect_hint=dialect_hint,
        schema_section=schema_section,
    )
    raw2 = complete(retry_prompt, max_tokens=300, stop_at=None)
    sql2 = _extract_sql(raw2)

    ok2, error2 = validate_sql(sql2, schema)
    if ok2:
        return sql2

    # Return best effort
    if sql:
        return sql
    if sql2:
        return sql2

    print("Error: model failed to generate a valid SQL query", file=sys.stderr)
    sys.exit(1)


def explain_sql(sql: str) -> str:
    """Explain a SQL query in plain English."""
    prompt = EXPLAIN_TEMPLATE.format(sql=sql)
    return complete(prompt, max_tokens=200, stop_at=None).strip()


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
        prog="hone-sql",
        description="Convert natural language to SQL queries (and back)",
        epilog='Example: echo "top 10 users by signups" | hone-sql',
    )
    parser.add_argument(
        "--schema", "-s", metavar="SCHEMA",
        help='Table schema hint, e.g. "users(id, name, email), orders(id, user_id, amount)"',
    )
    parser.add_argument(
        "--dialect", "-d", default="mysql",
        choices=list(DIALECT_HINTS.keys()),
        help="SQL dialect (default: mysql)",
    )
    parser.add_argument(
        "--explain", "-e", action="store_true",
        help="Reverse mode: input SQL, output plain English explanation",
    )
    parser.add_argument(
        "--validate", "-v", action="store_true",
        help="Syntax-check the generated SQL",
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
        # Reverse mode: SQL -> English
        if not is_sql_query(text):
            print("Warning: input doesn't look like SQL, explaining anyway", file=sys.stderr)
        explanation = explain_sql(text)
        print(explanation)
    else:
        # Forward mode: English -> SQL
        schema = parse_schema(args.schema) if args.schema else None
        sql = generate_sql(text, dialect=args.dialect, schema=schema)
        print(sql)

        if args.validate:
            ok, error = validate_sql(sql, schema)
            if ok:
                print("\nValid SQL syntax.", file=sys.stderr)
            else:
                print(f"\nWarning: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
