"""Tests for hone-explain CLI tool.

Two test phases:
1. Unit tests: detect_mode() and postprocess() — no model needed
2. Integration tests: full pipeline through the model (requires model)
"""

import subprocess
import sys

sys.path.insert(0, r"C:\Users\ericl\Documents\Projects\BitNet\tools")
from hone_tools.cli.explain import detect_mode, postprocess, MODES

PYTHON = sys.executable
CMD_PREFIX = [PYTHON, "-m", "hone_tools.cli.explain"]
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


def run_explain(input_text, extra_args=None, timeout=120):
    """Run hone-explain and return (exit_code, stdout, stderr)."""
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
print("PHASE 1: Unit tests — detect_mode() and postprocess()")
print("=" * 60)

# --- Error detection ---
print("\n--- Error Detection ---")

check("detect Python traceback (ImportError)", detect_mode(
    'Traceback (most recent call last):\n  File "app.py", line 3, in <module>\n    import pandas\nModuleNotFoundError: No module named \'pandas\''
) == "error")

check("detect Python traceback (KeyError)", detect_mode(
    'Traceback (most recent call last):\n  File "app.py", line 5, in <module>\n    x = d["missing"]\nKeyError: \'missing\''
) == "error")

check("detect JS TypeError", detect_mode(
    'TypeError: Cannot read properties of undefined (reading \'map\')\n    at Object.<anonymous> (/app/index.js:15:20)'
) == "error")

check("detect C segfault", detect_mode(
    'Segmentation fault (core dumped)'
) == "error")

check("detect gcc error", detect_mode(
    'main.c:10:5: error: expected \';\' after expression\n    printf("hello")\n    ^\n1 error generated.'
) == "error")

check("detect generic ERROR keyword", detect_mode(
    'ERROR: Failed to connect to database at localhost:5432'
) == "error")

check("detect FATAL keyword", detect_mode(
    'FATAL: password authentication failed for user "admin"'
) == "error")

check("detect rust panic", detect_mode(
    'thread \'main\' panicked at \'index out of bounds: the len is 3 but the index is 5\', src/main.rs:4:5'
) == "error")

# --- Code detection ---
print("\n--- Code Detection ---")

check("detect Python function", detect_mode(
    'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)'
) == "code")

check("detect Python class", detect_mode(
    'class UserProfile:\n    def __init__(self, name, email):\n        self.name = name\n        self.email = email\n    def greet(self):\n        return f"Hello, {self.name}"'
) == "code")

check("detect JavaScript function", detect_mode(
    'const fetchData = async (url) => {\n  const response = await fetch(url);\n  return response.json();\n};'
) == "code")

check("detect SQL query", detect_mode(
    'SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nJOIN orders o ON u.id = o.user_id\nWHERE o.created_at > \'2024-01-01\'\nGROUP BY u.name\nHAVING COUNT(o.id) > 5;'
) == "code")

check("detect JSON", detect_mode(
    '{"name": "hone-cli", "version": "0.1.0", "dependencies": {"numpy": ">=1.20"}}'
) == "code")

# --- Config detection ---
print("\n--- Config Detection ---")

check("detect docker-compose", detect_mode(
    'services:\n  web:\n    image: nginx:alpine\n    ports:\n      - "8080:80"\n  db:\n    image: postgres:15'
) == "config")

check("detect nginx config", detect_mode(
    'server {\n    listen 80;\n    server_name example.com;\n    location / {\n        proxy_pass http://backend:3000;\n    }\n}'
) == "config")

check("detect cron expression", detect_mode(
    '0 9 * * 1-5'
) == "config")

check("detect Dockerfile", detect_mode(
    'FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD ["python", "app.py"]'
) == "config")

check("detect systemd unit", detect_mode(
    '[Unit]\nDescription=My App Service\nAfter=network.target\n\n[Service]\nExecStart=/usr/bin/myapp\nRestart=always\n\n[Install]\nWantedBy=multi-user.target'
) == "config")

check("detect kubernetes YAML", detect_mode(
    'apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: nginx-deployment\nspec:\n  replicas: 3'
) == "config")

check("detect INI config", detect_mode(
    '[database]\nhost = localhost\nport = 5432\nname = myapp\n\n[redis]\nhost = localhost\nport = 6379'
) == "config")

# --- Regex detection ---
print("\n--- Regex Detection ---")

check("detect email regex", detect_mode(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
) == "regex")

check("detect URL regex", detect_mode(
    r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
) == "regex")

check("detect IP regex", detect_mode(
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
) == "regex")

# --- CLI detection ---
print("\n--- CLI Detection ---")

check("detect curl command", detect_mode(
    'curl -X POST -H "Content-Type: application/json" -d \'{"key":"val"}\' https://api.example.com/data'
) == "cli")

check("detect git command", detect_mode(
    'git rebase -i HEAD~3'
) == "cli")

check("detect docker run", detect_mode(
    'docker run -d -p 8080:80 --name web -v /data:/app/data nginx:alpine'
) == "cli")

check("detect $ prompt command", detect_mode(
    '$ kubectl get pods -n production --sort-by=.metadata.creationTimestamp'
) == "cli")

check("detect > prompt command", detect_mode(
    '> npm install --save-dev typescript @types/node'
) == "cli")

# --- Log detection ---
print("\n--- Log Detection ---")

check("detect ISO timestamp logs", detect_mode(
    '2024-03-15 10:23:45 INFO  Starting application server\n2024-03-15 10:23:46 INFO  Connected to database\n2024-03-15 10:23:47 WARN  Cache miss rate above threshold'
) == "log")

check("detect syslog format", detect_mode(
    'Mar 15 10:23:45 server1 sshd[1234]: Accepted publickey for user from 10.0.0.1\nMar 15 10:23:46 server1 sshd[1234]: pam_unix(sshd:session): session opened'
) == "log")

# --- Generic / edge cases ---
print("\n--- Generic & Edge Cases ---")

check("generic plain text", detect_mode(
    'The quick brown fox jumps over the lazy dog.'
) == "generic")

check("single word", detect_mode("hello") == "generic")

check("all modes are valid", all(m in MODES for m in MODES))

# --- Bash one-liner detection ---
print("\n--- Bash One-liner Detection ---")

check("detect find|xargs|grep", detect_mode(
    'find . -name "*.py" | xargs grep -l "TODO" | head -20'
) == "cli")

check("detect piped awk", detect_mode(
    'awk \'{print $1}\' access.log | sort | uniq -c | sort -rn | head -10'
) == "cli")

# --- Postprocess tests ---
print("\n--- Postprocess ---")

check("strip markdown headings", postprocess("### Explanation\nThis is the answer") == "This is the answer")
check("strip bold", postprocess("This is **important** text") == "This is important text")
check("strip code fences", postprocess("```python\ncode\n```") == "code")
check("brief: first sentence", postprocess("First sentence. Second sentence. Third.", brief=True) == "First sentence.")
check("brief: no period takes first line", postprocess("First line\nSecond line", brief=True) == "First line")
check("empty returns empty", postprocess("") == "")
check("whitespace only", postprocess("   \n  ") == "")

# Brief with various sentence endings
check("brief: exclamation", postprocess("Watch out! Second.", brief=True) == "Watch out!")
check("brief: question", postprocess("What happened? Let me explain.", brief=True) == "What happened?")


# ============================================================
# PHASE 2: Integration tests (requires model)
# ============================================================
print(f"\n{'=' * 60}")
print("PHASE 2: Integration tests (requires model)")
print("=" * 60)


def check_integration(name, input_text, extra_args=None, expect_fail=False,
                       min_length=10, check_output=None):
    """Run a single integration test."""
    global passed, failed, errors

    print(f"\n  TEST: {name}")
    preview = input_text[:80].replace("\n", "\\n")
    print(f"    Input: {preview}...")

    try:
        code, stdout, stderr = run_explain(input_text, extra_args)
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
            print(f"    PASS (expected failure, exit code {code})")
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

    print(f"    Output ({len(stdout)} chars): {stdout[:120]}...")

    # Basic length check
    if len(stdout) < min_length:
        failed += 1
        msg = f"FAIL [{name}]: Output too short ({len(stdout)} chars)"
        errors.append(msg)
        print(f"    {msg}")
        return

    if check_output:
        ok = check_output(stdout)
        if not ok:
            failed += 1
            msg = f"FAIL [{name}]: Output check failed"
            errors.append(msg)
            print(f"    {msg}")
            return

    passed += 1
    print(f"    PASS")


# --- ERRORS (1-5) ---
print("\n--- Errors ---")

check_integration(
    "1. Python ImportError traceback",
    'Traceback (most recent call last):\n  File "app.py", line 3, in <module>\n    import pandas\nModuleNotFoundError: No module named \'pandas\'',
)

check_integration(
    "2. Python KeyError traceback",
    'Traceback (most recent call last):\n  File "app.py", line 5, in <module>\n    x = data["missing_key"]\nKeyError: \'missing_key\'',
)

check_integration(
    "3. JS TypeError",
    'TypeError: Cannot read properties of undefined (reading \'map\')\n    at Object.<anonymous> (/app/index.js:15:20)\n    at Module._compile (node:internal/modules/cjs/loader:1241:14)',
)

check_integration(
    "4. C segfault",
    'Program received signal SIGSEGV, Segmentation fault.\n0x00005555555551a9 in main () at test.c:8\n8\t    *ptr = 42;',
)

check_integration(
    "5. GCC compilation error",
    'main.c: In function \'main\':\nmain.c:10:5: error: expected \';\' after expression\n   10 |     printf("hello")\n      |     ^~~~~~\n      |          ;\n1 error generated.',
)

# --- CODE (6-9) ---
print("\n--- Code ---")

check_integration(
    "6. Python fibonacci",
    'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)',
)

check_integration(
    "7. Bash one-liner",
    'find . -name "*.py" -mtime -7 | xargs grep -l "TODO" | sort',
)

check_integration(
    "8. SQL with joins",
    'SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nJOIN orders o ON u.id = o.user_id\nWHERE o.created_at > \'2024-01-01\'\nGROUP BY u.name\nHAVING COUNT(o.id) > 5\nORDER BY order_count DESC;',
)

check_integration(
    "9. Python class",
    'class RateLimiter:\n    def __init__(self, max_calls, period):\n        self.max_calls = max_calls\n        self.period = period\n        self.calls = []\n\n    def allow(self):\n        now = time.time()\n        self.calls = [t for t in self.calls if now - t < self.period]\n        if len(self.calls) < self.max_calls:\n            self.calls.append(now)\n            return True\n        return False',
)

# --- CONFIG (10-12) ---
print("\n--- Config ---")

check_integration(
    "10. Docker compose",
    'services:\n  web:\n    image: nginx:alpine\n    ports:\n      - "8080:80"\n    volumes:\n      - ./html:/usr/share/nginx/html\n  db:\n    image: postgres:15\n    environment:\n      POSTGRES_PASSWORD: secret',
)

check_integration(
    "11. Nginx server block",
    'server {\n    listen 80;\n    server_name example.com;\n    location / {\n        proxy_pass http://backend:3000;\n        proxy_set_header Host $host;\n    }\n    location /static {\n        root /var/www;\n        expires 30d;\n    }\n}',
)

check_integration(
    "12. Cron expression",
    '0 9 * * 1-5',
    extra_args=["-m", "config"],
)

# --- REGEX (13-14) ---
print("\n--- Regex ---")

check_integration(
    "13. Email regex",
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
)

check_integration(
    "14. URL regex",
    r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
)

# --- CLI (15-17) ---
print("\n--- CLI ---")

check_integration(
    "15. curl POST command",
    'curl -X POST -H "Content-Type: application/json" -d \'{"key":"val"}\' https://api.example.com/data',
)

check_integration(
    "16. git rebase",
    'git rebase -i HEAD~3',
)

check_integration(
    "17. docker run",
    'docker run -d -p 8080:80 --name web -v /data:/app/data nginx:alpine',
)

# --- EDGE CASES (18-22) ---
print("\n--- Edge Cases ---")

check_integration(
    "18. Empty input",
    "",
    expect_fail=True,
)

check_integration(
    "19. Single word",
    "kubernetes",
    min_length=5,
)

check_integration(
    "20. --brief flag",
    'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)',
    extra_args=["--brief"],
    check_output=lambda s: len(s.split(".")) <= 3 and len(s) < 200,
)

check_integration(
    "21. --fix flag on error",
    'Traceback (most recent call last):\n  File "app.py", line 3, in <module>\n    import pandas\nModuleNotFoundError: No module named \'pandas\'',
    extra_args=["--fix"],
)

check_integration(
    "22. --mode override",
    'some random text that is not code',
    extra_args=["-m", "code"],
    min_length=5,
)


# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'=' * 60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'=' * 60}")

if errors:
    print("\nFAILURES:")
    for e in errors:
        print(f"  - {e}")

sys.exit(0 if failed == 0 else 1)
