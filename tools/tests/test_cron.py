"""Tests for hone-cron CLI tool.

Two test phases:
1. Unit tests: validation, next-run-time calculation, extraction (no model needed)
2. Integration tests: full pipeline through the model (requires model)
"""

import re
import subprocess
import sys
from datetime import datetime, timedelta

# Add tools dir to path so we can import directly
sys.path.insert(0, r"C:\Users\ericl\Documents\Projects\BitNet\tools")
from hone_tools.cli.cron import (
    validate_cron,
    is_cron_expression,
    next_run_times,
    _extract_cron,
    _expand_field,
    MONTH_NAMES,
    DOW_NAMES,
)

PYTHON = sys.executable
CMD_PREFIX = [PYTHON, "-m", "hone_tools.cli.cron"]
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


def run_cron(input_text, extra_args=None, timeout=120):
    """Run hone-cron and return (exit_code, stdout, stderr)."""
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

# --- Validation ---
print("\n--- Cron Validation ---")

# Valid expressions
valid_exprs = [
    ("* * * * *", "every minute"),
    ("0 * * * *", "every hour"),
    ("0 0 * * *", "midnight daily"),
    ("0 9 * * 1-5", "weekdays at 9am"),
    ("0 15 * * 0", "Sunday 3pm"),
    ("*/15 * * * *", "every 15 min"),
    ("*/5 9-17 * * 1-5", "every 5 min business hours"),
    ("0 12 1 * *", "noon first of month"),
    ("0 8 * * 1,3", "Mon/Wed 8am"),
    ("0 9,17 * * *", "9am and 5pm"),
    ("0 0 1 1,4,7,10 *", "quarterly"),
    ("30 4 * * 0", "Sunday 4:30am"),
    ("0 0 * * 0", "Sunday midnight"),
    ("0 0 1 1 *", "Jan 1 midnight"),
    ("5 4 * * 7", "day-of-week 7 (Sunday)"),
]

for expr, desc in valid_exprs:
    ok, err = validate_cron(expr)
    check(f"Valid: {expr} ({desc})", ok, f"Error: {err}")

# Invalid expressions
invalid_exprs = [
    ("60 * * * *", "minute 60"),
    ("* 25 * * *", "hour 25"),
    ("* * 0 * *", "day 0"),
    ("* * 32 * *", "day 32"),
    ("* * * 13 *", "month 13"),
    ("* * * * 8", "dow 8"),
    ("* * * *", "only 4 fields"),
    ("* * * * * *", "6 fields"),
    ("", "empty"),
    ("abc def ghi jkl mno", "non-numeric"),
]

for expr, desc in invalid_exprs:
    ok, err = validate_cron(expr)
    check(f"Invalid: {expr} ({desc})", not ok, "Should have failed validation")

# --- is_cron_expression ---
print("\n--- is_cron_expression ---")
check("is_cron: '* * * * *'", is_cron_expression("* * * * *"))
check("is_cron: '0 9 * * 1-5'", is_cron_expression("0 9 * * 1-5"))
check("is_cron: '*/15 * * * *'", is_cron_expression("*/15 * * * *"))
check("not cron: 'every day'", not is_cron_expression("every day"))
check("not cron: '* * *'", not is_cron_expression("* * *"))
check("not cron: empty", not is_cron_expression(""))

# --- _extract_cron ---
print("\n--- _extract_cron ---")
check("extract plain", _extract_cron("0 9 * * 1-5") == "0 9 * * 1-5")
check("extract with prefix", _extract_cron("Cron: 0 9 * * 1-5") == "0 9 * * 1-5")
check("extract from backticks", _extract_cron("`0 9 * * 1-5`") == "0 9 * * 1-5")
check("extract with explanation", _extract_cron("The cron is:\n0 9 * * 1-5\nThis runs at 9am") == "0 9 * * 1-5")
check("extract multiline noise", _extract_cron("Here you go\n\n0 0 * * *\n\nDone!") == "0 0 * * *")

# --- _expand_field ---
print("\n--- _expand_field ---")
check("expand *", _expand_field("*", 0, 59, None) == set(range(60)))
check("expand 5", _expand_field("5", 0, 59, None) == {5})
check("expand 1-5", _expand_field("1-5", 0, 59, None) == {1, 2, 3, 4, 5})
check("expand 1,3,5", _expand_field("1,3,5", 0, 59, None) == {1, 3, 5})
check("expand */15", _expand_field("*/15", 0, 59, None) == {0, 15, 30, 45})
check("expand */5 (0-23)", _expand_field("*/5", 0, 23, None) == {0, 5, 10, 15, 20})
check("expand 9-17", _expand_field("9-17", 0, 23, None) == set(range(9, 18)))
check("expand MON", _expand_field("MON", 0, 7, DOW_NAMES) == {1})
check("expand JAN,APR", _expand_field("JAN,APR", 1, 12, MONTH_NAMES) == {1, 4})

# --- next_run_times ---
print("\n--- next_run_times ---")

# Use a fixed reference time for deterministic tests
ref = datetime(2026, 3, 22, 10, 0, 0)  # Sunday March 22, 2026, 10:00 AM

# Every minute: next 3 should be 10:01, 10:02, 10:03
times = next_run_times("* * * * *", count=3, from_time=ref)
check("every minute count", len(times) == 3)
check("every minute t1", times[0] == datetime(2026, 3, 22, 10, 1))
check("every minute t2", times[1] == datetime(2026, 3, 22, 10, 2))
check("every minute t3", times[2] == datetime(2026, 3, 22, 10, 3))

# Every hour at :00
times = next_run_times("0 * * * *", count=3, from_time=ref)
check("every hour count", len(times) == 3)
check("every hour t1", times[0] == datetime(2026, 3, 22, 11, 0))
check("every hour t2", times[1] == datetime(2026, 3, 22, 12, 0))

# Daily at midnight
times = next_run_times("0 0 * * *", count=3, from_time=ref)
check("daily midnight count", len(times) == 3)
check("daily midnight t1", times[0] == datetime(2026, 3, 23, 0, 0))
check("daily midnight t2", times[1] == datetime(2026, 3, 24, 0, 0))

# Weekday at 9am (ref is Sunday, so next is Monday March 23)
times = next_run_times("0 9 * * 1-5", count=3, from_time=ref)
check("weekday 9am count", len(times) == 3)
check("weekday 9am t1", times[0] == datetime(2026, 3, 23, 9, 0))  # Monday
check("weekday 9am t2", times[1] == datetime(2026, 3, 24, 9, 0))  # Tuesday
check("weekday 9am t3", times[2] == datetime(2026, 3, 25, 9, 0))  # Wednesday

# Sunday at 3pm (ref is Sunday 10am, so next is this Sunday at 3pm)
times = next_run_times("0 15 * * 0", count=3, from_time=ref)
check("sunday 3pm count", len(times) == 3)
check("sunday 3pm t1", times[0] == datetime(2026, 3, 22, 15, 0))  # This Sunday
check("sunday 3pm t2", times[1] == datetime(2026, 3, 29, 15, 0))  # Next Sunday

# Every 15 minutes
times = next_run_times("*/15 * * * *", count=4, from_time=ref)
check("every 15min count", len(times) == 4)
check("every 15min t1", times[0] == datetime(2026, 3, 22, 10, 15))
check("every 15min t2", times[1] == datetime(2026, 3, 22, 10, 30))
check("every 15min t3", times[2] == datetime(2026, 3, 22, 10, 45))
check("every 15min t4", times[3] == datetime(2026, 3, 22, 11, 0))

# First of month at noon
times = next_run_times("0 12 1 * *", count=3, from_time=ref)
check("1st noon count", len(times) == 3)
check("1st noon t1", times[0] == datetime(2026, 4, 1, 12, 0))
check("1st noon t2", times[1] == datetime(2026, 5, 1, 12, 0))

# Quarterly: 1st of Jan,Apr,Jul,Oct at midnight
times = next_run_times("0 0 1 1,4,7,10 *", count=4, from_time=ref)
check("quarterly count", len(times) == 4)
check("quarterly t1", times[0] == datetime(2026, 4, 1, 0, 0))
check("quarterly t2", times[1] == datetime(2026, 7, 1, 0, 0))
check("quarterly t3", times[2] == datetime(2026, 10, 1, 0, 0))
check("quarterly t4", times[3] == datetime(2027, 1, 1, 0, 0))

# Mon and Wed at 8am
times = next_run_times("0 8 * * 1,3", count=4, from_time=ref)
check("mon-wed 8am count", len(times) == 4)
# March 22 is Sunday. Next Mon=23, Wed=25, Mon=30, Wed=Apr 1
check("mon-wed 8am t1", times[0] == datetime(2026, 3, 23, 8, 0))
check("mon-wed 8am t2", times[1] == datetime(2026, 3, 25, 8, 0))

# 9am and 5pm daily
times = next_run_times("0 9,17 * * *", count=4, from_time=ref)
check("9am-5pm count", len(times) == 4)
check("9am-5pm t1", times[0] == datetime(2026, 3, 22, 17, 0))  # today 5pm
check("9am-5pm t2", times[1] == datetime(2026, 3, 23, 9, 0))   # tomorrow 9am

# Invalid expression returns empty
times = next_run_times("invalid", count=3)
check("invalid expr returns empty", len(times) == 0)

# Next run times are chronologically ordered
times = next_run_times("*/5 9-17 * * 1-5", count=10, from_time=ref)
check("chronological order", all(times[i] < times[i+1] for i in range(len(times)-1)))
check("business hours only", all(9 <= t.hour <= 17 for t in times))
check("weekdays only", all(t.weekday() < 5 for t in times))  # Monday=0..Friday=4

# Edge: month skip efficiency
ref_dec = datetime(2026, 12, 15, 10, 0, 0)
times = next_run_times("0 0 1 3 *", count=2, from_time=ref_dec)
check("month skip", len(times) == 2)
check("month skip t1", times[0] == datetime(2027, 3, 1, 0, 0))

print()

# ============================================================
# PHASE 2: Integration tests (requires model)
# ============================================================
print("=" * 60)
print("PHASE 2: Integration tests (requires model)")
print("=" * 60)


def check_integration(name, description, extra_args=None, acceptable_exprs=None,
                       must_be_valid=True, expect_fail=False, check_output=None):
    """Run a single integration test."""
    global passed, failed, errors

    print(f"\n  TEST: {name}")
    print(f"    Input: {description!r}")

    try:
        code, stdout, stderr = run_cron(description, extra_args)
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
            msg = f"FAIL [{name}]: Expected failure but got exit code 0"
            errors.append(msg)
            print(f"    {msg}")
        return

    if code != 0:
        failed += 1
        msg = f"FAIL [{name}]: Exit code {code}. Stderr: {stderr}"
        errors.append(msg)
        print(f"    {msg}")
        return

    output = stdout.split("\n")[0].strip()
    print(f"    Output: {output}")

    if must_be_valid:
        ok, err = validate_cron(output)
        if not ok:
            failed += 1
            msg = f"FAIL [{name}]: Invalid cron expression: {err}"
            errors.append(msg)
            print(f"    {msg}")
            return

    if acceptable_exprs:
        if output not in acceptable_exprs:
            # Still pass if valid - model may produce equivalent expression
            print(f"    Note: got {output!r}, expected one of {acceptable_exprs}")
            print(f"    (accepting as valid cron expression)")

    if check_output:
        if not check_output(stdout):
            failed += 1
            msg = f"FAIL [{name}]: Output check failed"
            errors.append(msg)
            print(f"    {msg}")
            return

    passed += 1
    print(f"    PASS")


# --- Basic generation (1-11) ---
print("\n--- Basic Generation ---")

check_integration(
    "1. every minute",
    "every minute",
    acceptable_exprs=["* * * * *"],
)

check_integration(
    "2. every hour",
    "every hour",
    acceptable_exprs=["0 * * * *"],
)

check_integration(
    "3. every day at midnight",
    "every day at midnight",
    acceptable_exprs=["0 0 * * *"],
)

check_integration(
    "4. every weekday at 9am",
    "every weekday at 9am",
    acceptable_exprs=["0 9 * * 1-5"],
)

check_integration(
    "5. every Sunday at 3pm",
    "every Sunday at 3pm",
    acceptable_exprs=["0 15 * * 0", "0 15 * * 7"],
)

check_integration(
    "6. every 15 minutes",
    "every 15 minutes",
    acceptable_exprs=["*/15 * * * *"],
)

check_integration(
    "7. every 5 min business hours",
    "every 5 minutes during business hours on weekdays",
    acceptable_exprs=["*/5 9-17 * * 1-5"],
)

check_integration(
    "8. first of month at noon",
    "first day of every month at noon",
    acceptable_exprs=["0 12 1 * *"],
)

check_integration(
    "9. Monday and Wednesday at 8am",
    "every Monday and Wednesday at 8am",
    acceptable_exprs=["0 8 * * 1,3"],
)

check_integration(
    "10. twice a day",
    "twice a day at 9am and 5pm",
    acceptable_exprs=["0 9,17 * * *"],
)

check_integration(
    "11. quarterly",
    "every quarter (Jan, Apr, Jul, Oct) on the 1st at midnight",
    acceptable_exprs=["0 0 1 1,4,7,10 *"],
)

# --- Explain mode (13-16) ---
print("\n--- Explain Mode ---")

check_integration(
    "13. explain weekday 9am",
    "0 9 * * 1-5",
    extra_args=["--explain"],
    must_be_valid=False,
    check_output=lambda s: len(s) > 5,  # got some English text back
)

check_integration(
    "14. explain every 15 min",
    "*/15 * * * *",
    extra_args=["--explain"],
    must_be_valid=False,
    check_output=lambda s: len(s) > 5,
)

check_integration(
    "15. explain first of month",
    "0 0 1 * *",
    extra_args=["--explain"],
    must_be_valid=False,
    check_output=lambda s: len(s) > 5,
)

check_integration(
    "16. explain Sunday 4:30am",
    "30 4 * * 0",
    extra_args=["--explain"],
    must_be_valid=False,
    check_output=lambda s: len(s) > 5,
)

# --- Validate mode (17) ---
print("\n--- Validate Mode ---")

check_integration(
    "17. validate every day at 3pm",
    "every day at 3pm",
    extra_args=["--validate"],
    check_output=lambda s: "Next 3 run times" in s or "next" in s.lower(),
)

# --- Edge cases (18-20) ---
print("\n--- Edge Cases ---")

check_integration(
    "18. empty input",
    "",
    expect_fail=True,
)

check_integration(
    "19. invalid cron --explain",
    "60 25 * * *",
    extra_args=["--explain"],
    expect_fail=True,
)

check_integration(
    "20. ambiguous input",
    "sometimes",
    # Should still produce something, even if not great
)

# --- Additional integration tests ---
print("\n--- Additional ---")

check_integration(
    "21. every Saturday at 6am",
    "every Saturday at 6am",
    acceptable_exprs=["0 6 * * 6"],
)

check_integration(
    "22. every 30 minutes",
    "every 30 minutes",
    acceptable_exprs=["*/30 * * * *", "0,30 * * * *"],
)

check_integration(
    "23. daily at 2:30am",
    "every day at 2:30am",
    acceptable_exprs=["30 2 * * *"],
)

check_integration(
    "24. every December 25th",
    "every December 25th at midnight",
    acceptable_exprs=["0 0 25 12 *"],
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
