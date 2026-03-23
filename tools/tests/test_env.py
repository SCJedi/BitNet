"""Tests for hone-env CLI tool.

Tests the regex extraction pipeline and output formatting.
No model calls needed for the core tests (--comments flag excluded).
"""

import json
import subprocess
import sys

PYTHON = sys.executable
CMD_PREFIX = [PYTHON, "-m", "hone_tools.cli.env"]
TOOLS_DIR = r"C:\Users\ericl\Documents\Projects\BitNet\tools"

passed = 0
failed = 0
errors = []


def run_env(input_text, extra_args=None, timeout=30):
    """Run hone-env with the given input and return (exit_code, stdout, stderr)."""
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


def check(name, input_text, extra_args=None, expect_vars=None, expect_not_vars=None,
          expect_exit=0, expect_in_output=None, expect_not_in_output=None):
    """Run a single test case."""
    global passed, failed, errors
    try:
        rc, stdout, stderr = run_env(input_text, extra_args)

        # Check exit code
        if rc != expect_exit:
            msg = f"FAIL [{name}]: expected exit {expect_exit}, got {rc}. stderr: {stderr}"
            print(msg)
            errors.append(msg)
            failed += 1
            return

        # Check expected vars appear in output
        if expect_vars:
            for var in expect_vars:
                if var not in stdout:
                    msg = f"FAIL [{name}]: expected '{var}' in output, got:\n{stdout}"
                    print(msg)
                    errors.append(msg)
                    failed += 1
                    return

        # Check vars that should NOT appear
        if expect_not_vars:
            for var in expect_not_vars:
                if var in stdout:
                    msg = f"FAIL [{name}]: '{var}' should NOT be in output, got:\n{stdout}"
                    print(msg)
                    errors.append(msg)
                    failed += 1
                    return

        # Check substring in output
        if expect_in_output:
            for s in expect_in_output:
                if s not in stdout:
                    msg = f"FAIL [{name}]: expected '{s}' in output, got:\n{stdout}"
                    print(msg)
                    errors.append(msg)
                    failed += 1
                    return

        # Check substring NOT in output
        if expect_not_in_output:
            for s in expect_not_in_output:
                if s in stdout:
                    msg = f"FAIL [{name}]: '{s}' should NOT be in output, got:\n{stdout}"
                    print(msg)
                    errors.append(msg)
                    failed += 1
                    return

        print(f"PASS [{name}]")
        passed += 1

    except subprocess.TimeoutExpired:
        msg = f"FAIL [{name}]: timed out"
        print(msg)
        errors.append(msg)
        failed += 1
    except Exception as e:
        msg = f"FAIL [{name}]: exception: {e}"
        print(msg)
        errors.append(msg)
        failed += 1


# ---- Test cases ----

# 1. Python os.getenv
check(
    "python_getenv",
    'import os\ndb = os.getenv("DATABASE_URL")\nkey = os.environ["API_KEY"]',
    expect_vars=["DATABASE_URL", "API_KEY"],
)

# 2. Python os.environ.get
check(
    "python_environ_get",
    'secret = os.environ.get("JWT_SECRET", "default")\nport = os.environ.get("APP_PORT", "8080")',
    expect_vars=["JWT_SECRET", "APP_PORT"],
)

# 3. JavaScript process.env.KEY
check(
    "js_process_env_dot",
    'const url = process.env.DATABASE_URL;\nconst key = process.env.STRIPE_API_KEY;',
    expect_vars=["DATABASE_URL", "STRIPE_API_KEY"],
)

# 4. JavaScript process.env["KEY"]
check(
    "js_process_env_bracket",
    'const secret = process.env["AUTH_SECRET"];\nconst mode = process.env["NODE_ENV"];',
    expect_vars=["AUTH_SECRET", "NODE_ENV"],
)

# 5. Docker compose with ${KEY}
check(
    "docker_compose_interp",
    '''version: "3"
services:
  web:
    image: myapp
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - SECRET_KEY=${SECRET_KEY}
''',
    expect_vars=["DATABASE_URL", "REDIS_URL", "SECRET_KEY"],
)

# 6. Dockerfile ENV directive
check(
    "dockerfile_env",
    '''FROM python:3.11
ENV APP_PORT=8080
ENV DATABASE_HOST=localhost
RUN pip install -r requirements.txt
''',
    expect_vars=["APP_PORT", "DATABASE_HOST"],
)

# 7. Shell script
check(
    "shell_vars",
    '''#!/bin/bash
echo $DATABASE_URL
curl -H "Authorization: Bearer $API_TOKEN" $API_BASE_URL/endpoint
''',
    expect_vars=["DATABASE_URL", "API_TOKEN", "API_BASE_URL"],
)

# 8. Mixed languages
check(
    "mixed_languages",
    '''# Python
db = os.getenv("DATABASE_URL")
# JavaScript
const key = process.env.API_KEY;
# Docker
ENV REDIS_HOST=localhost
# Shell
echo $SMTP_PASSWORD
''',
    expect_vars=["DATABASE_URL", "API_KEY", "REDIS_HOST", "SMTP_PASSWORD"],
)

# 9. No vars found
check(
    "no_vars_found",
    'def hello():\n    print("Hello, world!")\n    return 42',
    expect_exit=0,  # exits 0 with stderr message
)

# 10. JSON format output
check(
    "json_format",
    'import os\ndb = os.getenv("DATABASE_URL")\nkey = os.environ["API_KEY"]',
    extra_args=["--format", "json"],
    expect_in_output=['"DATABASE_URL"', '"API_KEY"'],
)

# 11. YAML format output
check(
    "yaml_format",
    'import os\ndb = os.getenv("DATABASE_URL")\nkey = os.environ["API_KEY"]',
    extra_args=["--format", "yaml"],
    expect_in_output=["DATABASE_URL:", "API_KEY:"],
)

# 12. Shell format output
check(
    "shell_format",
    'import os\ndb = os.getenv("DATABASE_URL")\nkey = os.environ["API_KEY"]',
    extra_args=["--format", "shell"],
    expect_in_output=["export DATABASE_URL=", "export API_KEY="],
)

# 13. Deduplication - same var referenced multiple times
check(
    "dedup",
    '''import os
db1 = os.getenv("DATABASE_URL")
db2 = os.environ["DATABASE_URL"]
db3 = os.environ.get("DATABASE_URL")
''',
    expect_vars=["DATABASE_URL"],
)

# 14. Placeholder values are sensible
check(
    "placeholder_url",
    'db = os.getenv("DATABASE_URL")',
    expect_in_output=["DATABASE_URL=https://example.com"],
)

# 15. Placeholder for secret
check(
    "placeholder_secret",
    'key = os.environ["SECRET_KEY"]',
    expect_in_output=["SECRET_KEY=changeme"],
)

# 16. Placeholder for port
check(
    "placeholder_port",
    'port = os.environ.get("APP_PORT")',
    expect_in_output=["APP_PORT=3000"],
)

# 17. docker-compose environment list style
check(
    "compose_env_list",
    '''services:
  app:
    environment:
      - DATABASE_URL=postgres://localhost/db
      - REDIS_URL=redis://localhost:6379
      - DEBUG_MODE
''',
    expect_vars=["DATABASE_URL", "REDIS_URL", "DEBUG_MODE"],
)

# 18. .env file lines
check(
    "dotenv_file_input",
    '''DATABASE_URL=postgres://localhost/mydb
API_KEY=sk-abc123
REDIS_URL=redis://localhost
''',
    expect_vars=["DATABASE_URL", "API_KEY", "REDIS_URL"],
)

# 19. JSON format is valid JSON
def test_json_valid():
    global passed, failed, errors
    name = "json_valid_parse"
    try:
        rc, stdout, stderr = run_env(
            'db = os.getenv("DATABASE_URL")\nkey = os.environ["API_KEY"]',
            extra_args=["--format", "json"],
        )
        if rc != 0:
            msg = f"FAIL [{name}]: exit code {rc}"
            print(msg)
            errors.append(msg)
            failed += 1
            return
        obj = json.loads(stdout)
        if "DATABASE_URL" not in obj or "API_KEY" not in obj:
            msg = f"FAIL [{name}]: missing keys in JSON: {obj}"
            print(msg)
            errors.append(msg)
            failed += 1
            return
        print(f"PASS [{name}]")
        passed += 1
    except json.JSONDecodeError as e:
        msg = f"FAIL [{name}]: invalid JSON: {e}\nOutput: {stdout}"
        print(msg)
        errors.append(msg)
        failed += 1

test_json_valid()

# 20. Noise filtering - common system vars should not appear from generic context
check(
    "noise_filter",
    'path = os.environ.get("APP_SECRET")\n# also uses PATH variable',
    expect_vars=["APP_SECRET"],
    expect_not_vars=["PATH"],
)

# 21. Empty input
check(
    "empty_input",
    "",
    expect_exit=1,
)

# ---- Summary ----
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
if errors:
    print(f"\nFailures:")
    for e in errors:
        print(f"  {e}")
print(f"{'='*50}")

sys.exit(1 if failed else 0)
