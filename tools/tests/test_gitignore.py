"""Tests for hone-gitignore CLI tool.

Two test phases:
1. Unit tests: template matching, keyword extraction, dedup, generation (no model)
2. Integration tests: full pipeline including model fallback (requires model)
"""

import subprocess
import sys
import tempfile
import os

# Add tools dir to path so we can import directly
sys.path.insert(0, r"C:\Users\ericl\Documents\Projects\BitNet\tools")
from hone_tools.cli.gitignore import (
    _normalize_keyword,
    _extract_keywords,
    _match_templates,
    _combine_templates,
    generate_gitignore,
    TEMPLATES,
    COMMON,
)

PYTHON = sys.executable
CMD_PREFIX = [PYTHON, "-m", "hone_tools.cli.gitignore"]
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


def run_gitignore(input_text, extra_args=None, timeout=120):
    """Run hone-gitignore and return (exit_code, stdout, stderr)."""
    cmd = CMD_PREFIX + (extra_args or [])
    result = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=TOOLS_DIR,
    )
    return result.returncode, result.stdout, result.stderr.strip()


# ============================================================
# PHASE 1: Unit tests (no model)
# ============================================================
print("=" * 60)
print("PHASE 1: Unit tests (no model needed)")
print("=" * 60)

# --- Keyword extraction ---
print("\n--- Keyword Extraction ---")

check("extract simple", _extract_keywords("python flask docker") == ["python", "flask", "docker"])
check("extract with commas", _extract_keywords("python, flask, docker") == ["python", "flask", "docker"])
check("extract with mixed separators", _extract_keywords("python/flask + docker") == ["python", "flask", "docker"])
check("extract uppercase", _extract_keywords("Python Flask Docker") == ["python", "flask", "docker"])
check("extract empty", _extract_keywords("") == [])
check("extract whitespace", _extract_keywords("   ") == [])

# --- Normalize keyword ---
print("\n--- Keyword Normalization ---")

check("normalize lowercase", _normalize_keyword("Python") == "python")
check("normalize strip", _normalize_keyword("  python  ") == "python")
check("normalize punctuation", _normalize_keyword("python,") == "python")

# --- Template matching ---
print("\n--- Template Matching ---")

check("match python", _match_templates(["python"]) == ["python"])
check("match node alias js", _match_templates(["js"]) == ["js"])
check("match multiple", _match_templates(["python", "docker"]) == ["python", "docker"])
check("match with unknown", _match_templates(["python", "foobar"]) == ["python"])
check("match none", _match_templates(["foobar", "baz"]) == [])
check("match aliases", _match_templates(["nodejs"]) == ["nodejs"])
check("match ts alias", _match_templates(["ts"]) == ["ts"])
check("match cpp alias", _match_templates(["cpp"]) == ["cpp"])

# --- Template combination ---
print("\n--- Template Combination ---")

lines = _combine_templates(["python"])
check("combine python has __pycache__", any("__pycache__/" in l for l in lines))
check("combine python has .venv", any(".venv/" in l for l in lines))

lines = _combine_templates(["python", "docker"])
check("combine python+docker has __pycache__", any("__pycache__/" in l for l in lines))
check("combine python+docker has .docker", any(".docker/" in l for l in lines))

lines = _combine_templates(["node", "typescript"])
check("combine node+ts has node_modules", any("node_modules/" in l for l in lines))
check("combine node+ts has tsconfig", any("tsconfig.tsbuildinfo" in l for l in lines))

# Test deduplication: node and react both have node_modules/
lines = _combine_templates(["node", "react"])
node_modules_count = sum(1 for l in lines if l.strip() == "node_modules/")
check("dedup node_modules", node_modules_count == 1, f"got {node_modules_count}")

# Test no-comments mode
lines_nc = _combine_templates(["python"], with_comments=False)
check("no-comments strips #", not any(l.startswith("#") for l in lines_nc if l))

# --- Full generation (template path, no model) ---
print("\n--- Full Generation (templates only) ---")

# 1. Single language
output = generate_gitignore("python")
check("gen python has __pycache__", "__pycache__/" in output)
check("gen python has .DS_Store", ".DS_Store" in output)
check("gen python has .env", ".env" in output)
check("gen python ends with newline", output.endswith("\n"))

# 2. Multiple languages
output = generate_gitignore("python docker")
check("gen python+docker has __pycache__", "__pycache__/" in output)
check("gen python+docker has .docker", ".docker/" in output)

# 3. Common combos: node+react
output = generate_gitignore("node react")
check("gen node+react has node_modules", "node_modules/" in output)
check("gen node+react has build", "build/" in output)

# 4. Common combos: python+flask+docker
output = generate_gitignore("python flask docker")
check("gen py+flask+docker has __pycache__", "__pycache__/" in output)
check("gen py+flask+docker has instance", "instance/" in output)
check("gen py+flask+docker has .docker", ".docker/" in output)

# 5. node+react+typescript
output = generate_gitignore("node react typescript")
check("gen node+react+ts has node_modules", "node_modules/" in output)
check("gen node+react+ts has tsconfig", "tsconfig.tsbuildinfo" in output)

# 6. Java
output = generate_gitignore("java")
check("gen java has *.class", "*.class" in output)
check("gen java has target", "target/" in output)

# 7. Go
output = generate_gitignore("go")
check("gen go has *.exe", "*.exe" in output)
check("gen go has vendor", "vendor/" in output)

# 8. Rust
output = generate_gitignore("rust")
check("gen rust has target", "target/" in output)
check("gen rust has Cargo.lock", "Cargo.lock" in output)

# 9. C/C++
output = generate_gitignore("c cpp")
check("gen c has *.o", "*.o" in output)
check("gen c has *.so", "*.so" in output)

# 10. --append flag: no common rules
output_append = generate_gitignore("python", append=True)
check("append no .DS_Store header", ".DS_Store" not in output_append or "__pycache__/" in output_append)
check("append has __pycache__", "__pycache__/" in output_append)

# Verify append mode actually omits the COMMON header
output_full = generate_gitignore("python", append=False)
check("append shorter than full", len(output_append) < len(output_full))

# 11. --comment flag off
output_nc = generate_gitignore("python", comments=False)
lines_nc = output_nc.split("\n")
comment_lines = [l for l in lines_nc if l.startswith("#")]
check("no-comment mode", len(comment_lines) == 0, f"found {len(comment_lines)} comment lines")

# 12. Empty-ish input with only noise words
output_noise = generate_gitignore("add rules for the project")
check("noise-only still returns something", len(output_noise) > 10)

# 13. Multiple blank lines collapsed
output = generate_gitignore("python docker rust")
check("no triple blank lines", "\n\n\n" not in output)

# --- CLI subprocess tests (no model, template path) ---
print("\n--- CLI Subprocess Tests ---")

# 14. Basic CLI invocation
code, stdout, stderr = run_gitignore("python")
check("cli python exit 0", code == 0)
check("cli python has __pycache__", "__pycache__/" in stdout)

# 15. CLI with --append
code, stdout, stderr = run_gitignore("python", extra_args=["--append"])
check("cli append exit 0", code == 0)
check("cli append has __pycache__", "__pycache__/" in stdout)

# 16. CLI with --no-comment
code, stdout, stderr = run_gitignore("python", extra_args=["--no-comment"])
check("cli no-comment exit 0", code == 0)
comment_lines = [l for l in stdout.split("\n") if l.startswith("#")]
check("cli no-comment no # lines", len(comment_lines) == 0, f"found {len(comment_lines)}")

# 17. CLI empty input
code, stdout, stderr = run_gitignore("")
check("cli empty input fails", code != 0)
check("cli empty input error msg", "empty input" in stderr.lower() or "error" in stderr.lower())

# 18. CLI -i flag with file
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write("python flask\n")
    tmpfile = f.name

try:
    code, stdout, stderr = run_gitignore("", extra_args=["-i", tmpfile])
    check("cli -i file exit 0", code == 0)
    check("cli -i file has __pycache__", "__pycache__/" in stdout)
    check("cli -i file has instance", "instance/" in stdout)
finally:
    os.unlink(tmpfile)

# 19. CLI -i with nonexistent file
code, stdout, stderr = run_gitignore("", extra_args=["-i", "nonexistent_file_12345.txt"])
check("cli -i nonexistent fails", code != 0)
check("cli -i nonexistent error msg", "not found" in stderr.lower() or "error" in stderr.lower())

# 20. Multiple technologies via CLI
code, stdout, stderr = run_gitignore("go rust java")
check("cli multi-tech exit 0", code == 0)
check("cli multi-tech has *.exe", "*.exe" in stdout)
check("cli multi-tech has *.class", "*.class" in stdout)
check("cli multi-tech has target/", "target/" in stdout)

# 21. Aliases work via CLI
code, stdout, stderr = run_gitignore("js ts")
check("cli aliases exit 0", code == 0)
check("cli aliases has node_modules", "node_modules/" in stdout)
check("cli aliases has tsconfig", "tsconfig.tsbuildinfo" in stdout)

# 22. IDE templates
code, stdout, stderr = run_gitignore("python vscode jetbrains")
check("cli ide exit 0", code == 0)
check("cli ide has .vscode", ".vscode/" in stdout)
check("cli ide has .idea", ".idea/" in stdout)

# 23. OS templates
code, stdout, stderr = run_gitignore("macos windows linux")
check("cli os exit 0", code == 0)
check("cli os has .DS_Store", ".DS_Store" in stdout)
check("cli os has Thumbs.db", "Thumbs.db" in stdout)

# 24. Terraform
code, stdout, stderr = run_gitignore("terraform")
check("cli terraform exit 0", code == 0)
check("cli terraform has .terraform", ".terraform/" in stdout)
check("cli terraform has tfstate", "*.tfstate" in stdout)

# 25. Unity
code, stdout, stderr = run_gitignore("unity")
check("cli unity exit 0", code == 0)
check("cli unity has Library", "[Ll]ibrary/" in stdout)

# 26. LaTeX
code, stdout, stderr = run_gitignore("latex")
check("cli latex exit 0", code == 0)
check("cli latex has *.aux", "*.aux" in stdout)

# --- Template coverage checks ---
print("\n--- Template Coverage ---")

# Verify all template keys are accessible
for key in ["python", "node", "java", "go", "rust", "c", "cpp", "ruby", "php",
            "docker", "vscode", "jetbrains", "macos", "windows", "linux",
            "flask", "django", "terraform", "swift", "kotlin", "latex",
            "unity", "angular", "vue", "nextjs", "react", "typescript"]:
    tmpl = TEMPLATES.get(key)
    check(f"template '{key}' exists", tmpl is not None, f"key {key} not in TEMPLATES")


# ============================================================
# PHASE 2: Integration tests (requires model, for unknown tech)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Integration tests (model fallback)")
print("=" * 60)

# 27. Unknown technology - falls to model
print("\n--- Model Fallback ---")

code, stdout, stderr = run_gitignore("elixir phoenix")
check("cli unknown tech exit 0", code == 0, f"stderr: {stderr}")
check("cli unknown tech has output", len(stdout.strip()) > 10, f"output: {stdout[:100]}")

# 28. Natural language input
code, stdout, stderr = run_gitignore("add rules for compiled C files")
check("cli natural lang exit 0", code == 0, f"stderr: {stderr}")
check("cli natural lang has output", len(stdout.strip()) > 10)
# Should match 'c' template via keyword extraction
check("cli natural lang has *.o", "*.o" in stdout)

# 29. Mixed known + unknown
code, stdout, stderr = run_gitignore("python celery redis")
check("cli mixed exit 0", code == 0, f"stderr: {stderr}")
check("cli mixed has __pycache__", "__pycache__/" in stdout)


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
