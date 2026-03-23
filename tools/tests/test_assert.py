"""Tests for bt-assert CLI tool.

Tests the pre-processing logic (deterministic, no model needed).
Each test pipes an assertion via stdin and checks the exit code.
Exit 0 = PASS, Exit 1 = FAIL.
"""

import subprocess
import sys

PYTHON = sys.executable
CMD_PREFIX = [PYTHON, "-m", "bitnet_tools.cli.assert_cmd"]
TOOLS_DIR = r"C:\Users\ericl\Documents\Projects\BitNet\tools"

passed = 0
failed = 0
errors = []


def run_assert(assertion: str, value: str, verbose: bool = False, timeout: int = 10):
    """Run bt-assert and return (exit_code, stdout, stderr)."""
    cmd = CMD_PREFIX + ["--val", value]
    if verbose:
        cmd.append("--verbose")
    result = subprocess.run(
        cmd,
        input=assertion,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=TOOLS_DIR,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def check(name: str, assertion: str, value: str, expect_pass: bool):
    """Run a single test case and verify exit code."""
    global passed, failed, errors

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Assertion: {assertion!r}")
    print(f"  Value:     {value!r}")
    print(f"  Expect:    {'PASS (exit 0)' if expect_pass else 'FAIL (exit 1)'}")

    try:
        code, stdout, stderr = run_assert(assertion, value, verbose=True)
        expected_code = 0 if expect_pass else 1

        if code == expected_code:
            print(f"  Result:    OK (exit {code})")
            if stdout:
                print(f"  Output:    {stdout}")
            passed += 1
        else:
            msg = f"FAILED: expected exit {expected_code}, got {code}"
            print(f"  Result:    {msg}")
            if stdout:
                print(f"  Stdout:    {stdout}")
            if stderr:
                print(f"  Stderr:    {stderr}")
            failed += 1
            errors.append(f"{name}: {msg}")

    except subprocess.TimeoutExpired:
        msg = "TIMEOUT"
        print(f"  Result:    {msg}")
        failed += 1
        errors.append(f"{name}: {msg}")
    except Exception as e:
        msg = f"ERROR: {e}"
        print(f"  Result:    {msg}")
        failed += 1
        errors.append(f"{name}: {msg}")


# =========================================================================
# Test cases
# =========================================================================

# --- Numeric comparisons ---
check("numeric_equals",
      "status code is 200", "200", expect_pass=True)

check("numeric_equals_fail",
      "status code is 200", "404", expect_pass=False)

check("greater_than_pass",
      "greater than 5", "10", expect_pass=True)

check("greater_than_fail",
      "greater than 5", "3", expect_pass=False)

check("less_than_pass",
      "less than 100", "50", expect_pass=True)

check("less_than_fail",
      "less than 100", "200", expect_pass=False)

check("at_least_pass",
      "at least 3", "3", expect_pass=True)

check("at_least_fail",
      "at least 3", "2", expect_pass=False)

check("at_most_pass",
      "at most 10", "10", expect_pass=True)

check("at_most_fail",
      "at most 10", "15", expect_pass=False)

check("between_pass",
      "between 1 and 10", "5", expect_pass=True)

check("between_fail",
      "between 1 and 10", "15", expect_pass=False)

check("is_positive_pass",
      "is positive", "42", expect_pass=True)

check("is_positive_fail",
      "is positive", "-3", expect_pass=False)

check("is_negative_pass",
      "is negative", "-7", expect_pass=True)

check("is_even_pass",
      "is even", "4", expect_pass=True)

check("is_even_fail",
      "is even", "7", expect_pass=False)

check("is_odd_pass",
      "is odd", "7", expect_pass=True)

# --- String contains ---
check("contains_pass",
      "contains hello", "hello world", expect_pass=True)

check("contains_fail",
      "contains error", "success: true", expect_pass=False)

check("not_contains_pass",
      "does not contain error", "success: true", expect_pass=True)

check("not_contains_fail",
      "does not contain error", "error: bad request", expect_pass=False)

# --- Starts with / ends with ---
check("starts_with_pass",
      "starts with http", "https://example.com", expect_pass=True)

check("starts_with_fail",
      "starts with http", "ftp://example.com", expect_pass=False)

check("ends_with_pass",
      "ends with .json", "data.json", expect_pass=True)

check("ends_with_fail",
      "ends with .json", "data.xml", expect_pass=False)

# --- Empty checks ---
check("is_empty_pass",
      "is empty", "", expect_pass=True)

check("is_empty_fail",
      "is empty", "something", expect_pass=False)

check("not_empty_pass",
      "is not empty", "something", expect_pass=True)

check("not_empty_fail",
      "is not empty", "", expect_pass=False)

# --- Type checks ---
check("is_number_pass",
      "is a number", "42.5", expect_pass=True)

check("is_number_fail",
      "is a number", "hello", expect_pass=False)

check("is_integer_pass",
      "is an integer", "42", expect_pass=True)

check("is_integer_fail",
      "is an integer", "42.5", expect_pass=False)

check("is_email_pass",
      "is a valid email", "john@test.com", expect_pass=True)

check("is_email_fail",
      "is a valid email", "not-an-email", expect_pass=False)

check("is_url_pass",
      "is a valid url", "https://example.com", expect_pass=True)

check("is_url_fail",
      "is a valid url", "not a url", expect_pass=False)

check("is_json_pass",
      "is valid json", '{"key": "value"}', expect_pass=True)

check("is_json_fail",
      "is valid json", "{broken json", expect_pass=False)

check("is_boolean_pass",
      "is a boolean", "true", expect_pass=True)

check("is_boolean_fail",
      "is a boolean", "maybe", expect_pass=False)

# --- List/array length ---
check("list_more_than_pass",
      "has more than 2 items", "[1,2,3,4]", expect_pass=True)

check("list_more_than_fail",
      "has more than 5 items", "[1,2,3]", expect_pass=False)

check("list_exactly_pass",
      "has 3 items", "[1,2,3]", expect_pass=True)

check("list_exactly_fail",
      "has 3 items", "[1,2]", expect_pass=False)

check("list_at_least_pass",
      "has at least 2 elements", "[1,2,3]", expect_pass=True)

check("list_fewer_than_pass",
      "has fewer than 5 items", "[1,2]", expect_pass=True)

# --- True/false checks ---
check("is_true_pass",
      "is true", "true", expect_pass=True)

check("is_true_fail",
      "is true", "false", expect_pass=False)

check("is_false_pass",
      "is false", "false", expect_pass=True)

# --- Regex match ---
check("regex_match_pass",
      "matches regex ^\\d{3}$", "200", expect_pass=True)

check("regex_match_fail",
      "matches regex ^\\d{3}$", "20", expect_pass=False)

# --- Null checks ---
check("null_empty_pass",
      "is null", "null", expect_pass=True)

check("null_empty_fail",
      "is null", "something", expect_pass=False)

# --- Edge cases ---
check("value_equals_pass",
      "value is ok", "ok", expect_pass=True)

check("result_equals_pass",
      "result equals success", "success", expect_pass=True)

check("result_equals_fail",
      "result equals success", "failure", expect_pass=False)

check("is_zero_pass",
      "is zero", "0", expect_pass=True)

check("is_zero_fail",
      "is zero", "5", expect_pass=False)

# =========================================================================
# Summary
# =========================================================================
print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
if errors:
    print(f"\nFAILURES:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
