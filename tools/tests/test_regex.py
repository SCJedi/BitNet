"""Tests for bt-regex CLI tool.

Tests the full pipeline: natural language -> regex pattern -> validation.
Each test verifies:
1. The command exits successfully (exit code 0)
2. The output is a compilable regex (re.compile succeeds)
3. Where --test is used, expected matches are found
"""

import re
import subprocess
import sys

PYTHON = sys.executable
CMD_PREFIX = [PYTHON, "-m", "bitnet_tools.cli.regex"]
TOOLS_DIR = r"C:\Users\ericl\Documents\Projects\BitNet\tools"

passed = 0
failed = 0
errors = []


def run_regex(description, extra_args=None, timeout=120):
    """Run bt-regex with the given description and return (exit_code, stdout, stderr)."""
    cmd = CMD_PREFIX + (extra_args or [])
    result = subprocess.run(
        cmd,
        input=description,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=TOOLS_DIR,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def check(name, description, extra_args=None, must_match=None, must_not_match=None,
          expect_fail=False, test_string=None, expected_in_output=None):
    """Run a single test case."""
    global passed, failed, errors

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Description: {description!r}")
    if extra_args:
        print(f"  Args: {extra_args}")

    try:
        args = list(extra_args) if extra_args else []
        code, stdout, stderr = run_regex(description, args)
    except subprocess.TimeoutExpired:
        failed += 1
        msg = f"FAIL [{name}]: Timed out"
        errors.append(msg)
        print(f"  {msg}")
        return
    except Exception as e:
        failed += 1
        msg = f"FAIL [{name}]: Exception: {e}"
        errors.append(msg)
        print(f"  {msg}")
        return

    if expect_fail:
        if code != 0:
            passed += 1
            print(f"  PASS (expected failure, got exit code {code})")
        else:
            failed += 1
            msg = f"FAIL [{name}]: Expected failure but got success"
            errors.append(msg)
            print(f"  {msg}")
        return

    # Check exit code
    if code != 0:
        failed += 1
        msg = f"FAIL [{name}]: Exit code {code}. Stderr: {stderr}"
        errors.append(msg)
        print(f"  {msg}")
        return

    # Extract just the pattern (first line of output)
    pattern = stdout.split("\n")[0].strip()
    print(f"  Pattern: {pattern}")

    if not pattern:
        failed += 1
        msg = f"FAIL [{name}]: Empty pattern output"
        errors.append(msg)
        print(f"  {msg}")
        return

    # Check it compiles
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        failed += 1
        msg = f"FAIL [{name}]: Pattern doesn't compile: {e}"
        errors.append(msg)
        print(f"  {msg}")
        return

    # Check must_match strings
    if must_match:
        for s in must_match:
            if not compiled.search(s):
                failed += 1
                msg = f"FAIL [{name}]: Pattern should match {s!r} but doesn't"
                errors.append(msg)
                print(f"  {msg}")
                return

    # Check must_not_match strings
    if must_not_match:
        for s in must_not_match:
            if compiled.search(s):
                failed += 1
                msg = f"FAIL [{name}]: Pattern should NOT match {s!r} but does"
                errors.append(msg)
                print(f"  {msg}")
                return

    # Check expected strings in full output
    if expected_in_output:
        for s in expected_in_output:
            if s not in stdout:
                failed += 1
                msg = f"FAIL [{name}]: Expected {s!r} in output but not found"
                errors.append(msg)
                print(f"  {msg}")
                return

    passed += 1
    print(f"  PASS")


# ============================================================
# BASIC PATTERNS (1-6)
# ============================================================

check(
    "1. Email addresses",
    "email addresses",
    must_match=["john@example.com", "user.name+tag@domain.co"],
    must_not_match=["plaintext", "@@"],
)

check(
    "2. US phone numbers",
    "US phone numbers like (555) 123-4567 or 555-123-4567",
    must_match=["555-123-4567"],
)

check(
    "3. URLs",
    "URLs starting with http or https",
    must_match=["https://example.com", "http://test.org/path"],
)

check(
    "4. ISO dates",
    "dates in YYYY-MM-DD format",
    must_match=["2024-03-15", "1999-12-31"],
    must_not_match=["03-15-2024"],
)

check(
    "5. IPv4 addresses",
    "IPv4 addresses with four groups of digits separated by dots, like 192.168.0.1",
    must_match=["192.168.0.1"],
)

check(
    "6. Hex color codes",
    "6-digit hex color codes like #ff0000 or #00AABB",
    must_match=["#ff0000"],
)

# ============================================================
# SPECIFIC PATTERNS (7-11)
# ============================================================

check(
    "7. Positive integers",
    "positive integers (one or more digits)",
    must_match=["42", "7", "100"],
)

check(
    "8. Capitalized words",
    "words starting with a capital letter",
    must_match=["Hello", "World"],
)

check(
    "9. Comment lines",
    "lines that start with #",
    must_match=["# this is a comment", "#comment"],
)

check(
    "10. JSON keys",
    'double-quoted strings followed by a colon like "key":',
    must_match=['"name":'],
)

check(
    "11. Python function defs",
    "Python function definitions starting with def",
    must_match=["def foo()", "def my_func(x, y)"],
)

# ============================================================
# WITH --test FLAG (12-14)
# ============================================================

check(
    "12. Email with --test",
    "email addresses",
    extra_args=["--test", "contact john@test.com or visit our site"],
    expected_in_output=["john@test.com"],
)

check(
    "13. Numbers with --test",
    "numbers (sequences of digits)",
    extra_args=["--test", "I have 3 cats and 12 dogs"],
    expected_in_output=["3", "12"],
)

check(
    "14. Dates with --test",
    "dates in YYYY-MM-DD format",
    extra_args=["--test", "deadline is 2024-03-15 ok"],
    expected_in_output=["2024-03-15"],
)

# ============================================================
# EDGE CASES (15-18)
# ============================================================

check(
    "15. Empty input",
    "",
    expect_fail=True,
)

check(
    "16. Vague input",
    "match stuff",
    # Should still produce a compilable pattern
)

check(
    "17. Impossible (prime numbers)",
    "match prime numbers only",
    # Can't do this with regex but should still produce something compilable
)

check(
    "18. With --explain flag",
    "email addresses",
    extra_args=["--explain"],
    expected_in_output=["Explanation:"],
)

# ============================================================
# ADDITIONAL TESTS (19-23)
# ============================================================

check(
    "19. Flavor: javascript",
    "email addresses",
    extra_args=["--flavor", "javascript"],
    must_match=["test@example.com"],
)

check(
    "20. Flavor: grep",
    "words with digits",
    extra_args=["--flavor", "grep"],
    must_match=["abc123"],
)

check(
    "21. SSN pattern",
    "US Social Security numbers like 123-45-6789",
    must_match=["123-45-6789"],
    must_not_match=["12-345-6789"],
)

check(
    "22. Whitespace",
    "one or more whitespace characters",
    must_match=["  ", "\t"],
    must_not_match=["abc"],
)

check(
    "23. File extensions",
    "filenames ending in .py or .js",
    must_match=["script.py", "app.js"],
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
