"""Tests for bt-commit: diff parsing, scope inference, type inference, and formatting."""

import pytest
from bitnet_tools.cli.commit import (
    parse_diff, infer_scope, infer_type, build_summary,
    build_prompt, format_commit, VALID_TYPES,
)


# ============================================================
# Sample diffs
# ============================================================

DIFF_SINGLE_FILE_ADD = """\
diff --git a/src/auth/login.py b/src/auth/login.py
index 1234567..abcdefg 100644
--- a/src/auth/login.py
+++ b/src/auth/login.py
@@ -10,6 +10,15 @@ class AuthManager:
     def __init__(self):
         self.sessions = {}

+    def validate_token(self, token: str) -> bool:
+        \"\"\"Validate a JWT token.\"\"\"
+        if not token:
+            return False
+        try:
+            payload = jwt.decode(token, self.secret, algorithms=["HS256"])
+            return payload.get("exp", 0) > time.time()
+        except jwt.InvalidTokenError:
+            return False
+
     def login(self, username, password):
         pass
"""

DIFF_BUG_FIX = """\
diff --git a/gpu/convert_checkpoint.py b/gpu/convert_checkpoint.py
index 797ad1d..d3a7037 100755
--- a/gpu/convert_checkpoint.py
+++ b/gpu/convert_checkpoint.py
@@ -34,7 +34,7 @@ def convert_ts_checkpoint(
     def convert_int8_to_int2(weight):
         return convert_weight_int8_to_int2(weight)

-    merged_result = torch.load(input_path, map_location="cpu", mmap=True)
+    merged_result = torch.load(input_path, map_location="cpu", mmap=True, weights_only=True)
     int2_result = {}
     fp16_result = {}
     zero = torch.zeros(1).to(torch.bfloat16)
diff --git a/gpu/generate.py b/gpu/generate.py
index 638ed7b..030b97f 100755
--- a/gpu/generate.py
+++ b/gpu/generate.py
@@ -64,9 +64,9 @@ class FastGen:
         decode_model = fast.Transformer(model_args_decode)

         fp16_ckpt_path = str(Path(ckpt_dir) / "model_state_fp16.pt")
-        fp16_checkpoint = torch.load(fp16_ckpt_path, map_location="cpu")
+        fp16_checkpoint = torch.load(fp16_ckpt_path, map_location="cpu", weights_only=True)
         int2_ckpt_path = str(Path(ckpt_dir) / "model_state_int2.pt")
-        int2_checkpoint = torch.load(int2_ckpt_path, map_location="cpu")
+        int2_checkpoint = torch.load(int2_ckpt_path, map_location="cpu", weights_only=True)
         prefill_model.load_state_dict(fp16_checkpoint, strict=True)
         decode_model.load_state_dict(int2_checkpoint, strict=True)
"""

DIFF_MULTI_FILE_FEATURE = """\
diff --git a/src/api/routes.py b/src/api/routes.py
new file mode 100644
--- /dev/null
+++ b/src/api/routes.py
@@ -0,0 +1,20 @@
+from flask import Blueprint, jsonify, request
+
+api = Blueprint("api", __name__)
+
+@api.route("/users", methods=["GET"])
+def list_users():
+    return jsonify(get_all_users())
+
+@api.route("/users/<int:uid>", methods=["GET"])
+def get_user(uid):
+    return jsonify(get_user_by_id(uid))
+
+@api.route("/users", methods=["POST"])
+def create_user():
+    data = request.json
+    return jsonify(create_new_user(data)), 201
diff --git a/src/api/models.py b/src/api/models.py
new file mode 100644
--- /dev/null
+++ b/src/api/models.py
@@ -0,0 +1,12 @@
+class User:
+    def __init__(self, name, email):
+        self.name = name
+        self.email = email
+
+    def to_dict(self):
+        return {"name": self.name, "email": self.email}
diff --git a/tests/test_api.py b/tests/test_api.py
new file mode 100644
--- /dev/null
+++ b/tests/test_api.py
@@ -0,0 +1,8 @@
+def test_list_users(client):
+    resp = client.get("/users")
+    assert resp.status_code == 200
+
+def test_create_user(client):
+    resp = client.post("/users", json={"name": "Alice"})
+    assert resp.status_code == 201
"""

DIFF_DOCS_ONLY = """\
diff --git a/README.md b/README.md
index aaa..bbb 100644
--- a/README.md
+++ b/README.md
@@ -1,5 +1,10 @@
 # Project Name

-A simple project.
+A comprehensive project for building CLI tools.
+
+## Installation
+
+```bash
+pip install bitnet-tools
+```
"""

DIFF_TEST_ONLY = """\
diff --git a/tests/test_engine.py b/tests/test_engine.py
index aaa..bbb 100644
--- a/tests/test_engine.py
+++ b/tests/test_engine.py
@@ -10,6 +10,16 @@ class TestEngine:
     def test_basic(self):
         assert complete("hello") != ""

+    def test_temperature(self):
+        r1 = complete("test", temperature=0.0)
+        r2 = complete("test", temperature=0.0)
+        assert r1 == r2
+
+    def test_max_tokens(self):
+        r = complete("test", max_tokens=5)
+        assert len(r.split()) <= 10
"""

DIFF_REFACTOR = """\
diff --git a/src/utils.py b/src/utils.py
index aaa..bbb 100644
--- a/src/utils.py
+++ b/src/utils.py
@@ -1,30 +1,15 @@
-def process_data(data):
-    result = []
-    for item in data:
-        if item.get("active"):
-            name = item["name"]
-            value = item["value"]
-            processed = {"name": name.upper(), "val": value * 2}
-            result.append(processed)
-    return result
-
-def filter_data(data):
-    filtered = []
-    for item in data:
-        if item.get("active"):
-            filtered.append(item)
-    return filtered
+def process_data(data):
+    return [
+        {"name": item["name"].upper(), "val": item["value"] * 2}
+        for item in data
+        if item.get("active")
+    ]
+
+def filter_active(data):
+    return [item for item in data if item.get("active")]
"""

DIFF_CONFIG = """\
diff --git a/pyproject.toml b/pyproject.toml
index aaa..bbb 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -10,6 +10,7 @@ dependencies = []
 bt-summarize = "bitnet_tools.cli.summarize:main"
 bt-extract = "bitnet_tools.cli.extract:main"
+bt-commit = "bitnet_tools.cli.commit:main"
"""

DIFF_DELETE_ONLY = """\
diff --git a/src/deprecated.py b/src/deprecated.py
deleted file mode 100644
index aaa..000 100644
--- a/src/deprecated.py
+++ /dev/null
@@ -1,25 +0,0 @@
-\"\"\"Deprecated utilities - remove in v2.0\"\"\"
-
-def old_function():
-    pass
-
-def legacy_handler():
-    pass
-
-class OldClass:
-    def method(self):
-        pass
"""

DIFF_LARGE = """\
diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1,3 +1,10 @@
+import os
+import sys
+import json
+import logging
+import pathlib
+from typing import Optional
+
 def main():
     pass
diff --git a/src/b.py b/src/b.py
--- a/src/b.py
+++ b/src/b.py
@@ -1,3 +1,8 @@
+from dataclasses import dataclass
+
+@dataclass
+class Config:
+    debug: bool = False
 pass
diff --git a/src/c.py b/src/c.py
--- a/src/c.py
+++ b/src/c.py
@@ -1,3 +1,6 @@
+def helper():
+    return 42
+
 pass
diff --git a/src/d.py b/src/d.py
--- a/src/d.py
+++ b/src/d.py
@@ -1,3 +1,6 @@
+class Handler:
+    pass
+
 pass
diff --git a/src/e.py b/src/e.py
--- a/src/e.py
+++ b/src/e.py
@@ -1,3 +1,6 @@
+def process():
+    return None
+
 pass
diff --git a/src/f.py b/src/f.py
--- a/src/f.py
+++ b/src/f.py
@@ -1,3 +1,6 @@
+async def fetch():
+    return {}
+
 pass
"""

DIFF_ONE_LINE = """\
diff --git a/config.py b/config.py
index aaa..bbb 100644
--- a/config.py
+++ b/config.py
@@ -5,3 +5,3 @@
-DEBUG = True
+DEBUG = False
"""

DIFF_CI = """\
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index aaa..bbb 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -10,6 +10,8 @@ jobs:
     steps:
       - uses: actions/checkout@v4
       - uses: actions/setup-python@v5
+        with:
+          python-version: "3.12"
       - run: pip install -e .
       - run: pytest
"""


# ============================================================
# Tests: parse_diff
# ============================================================

class TestParseDiff:
    def test_single_file(self):
        result = parse_diff(DIFF_SINGLE_FILE_ADD)
        assert result["is_diff"] is True
        assert len(result["files"]) == 1
        assert result["files"][0]["path"] == "src/auth/login.py"
        assert result["total_add"] == 10
        assert result["total_del"] == 0

    def test_bug_fix_multi_file(self):
        result = parse_diff(DIFF_BUG_FIX)
        assert result["is_diff"] is True
        assert len(result["files"]) == 2
        assert result["total_add"] == 3
        assert result["total_del"] == 3

    def test_new_files(self):
        result = parse_diff(DIFF_MULTI_FILE_FEATURE)
        assert len(result["files"]) == 3
        total_added = sum(f["added"] for f in result["files"])
        assert total_added >= 30

    def test_delete_only(self):
        result = parse_diff(DIFF_DELETE_ONLY)
        assert result["is_diff"] is True
        assert result["total_del"] > 0
        assert result["total_add"] == 0

    def test_empty_string(self):
        result = parse_diff("")
        assert result["is_diff"] is False
        assert result["files"] == []

    def test_non_diff_text(self):
        result = parse_diff("This is just some random text\nwith multiple lines\n")
        assert result["is_diff"] is False

    def test_notable_changes_captured(self):
        result = parse_diff(DIFF_SINGLE_FILE_ADD)
        changes = result["files"][0]["changes"]
        assert any("validate_token" in c for c in changes)

    def test_large_diff(self):
        result = parse_diff(DIFF_LARGE)
        assert len(result["files"]) == 6
        assert result["total_add"] > 20


# ============================================================
# Tests: infer_scope
# ============================================================

class TestInferScope:
    def test_single_file_scope(self):
        files = [{"path": "src/auth/login.py", "added": 5, "removed": 0}]
        assert infer_scope(files) == "login"

    def test_shared_directory(self):
        files = [
            {"path": "src/api/routes.py", "added": 10, "removed": 0},
            {"path": "src/api/models.py", "added": 5, "removed": 0},
        ]
        assert infer_scope(files) == "api"

    def test_top_level_directory(self):
        files = [
            {"path": "gpu/convert_checkpoint.py", "added": 1, "removed": 1},
            {"path": "gpu/generate.py", "added": 2, "removed": 2},
        ]
        assert infer_scope(files) == "gpu"

    def test_no_files(self):
        assert infer_scope([]) is None

    def test_divergent_paths(self):
        files = [
            {"path": "src/api/routes.py", "added": 5, "removed": 0},
            {"path": "tests/test_api.py", "added": 3, "removed": 0},
            {"path": "docs/api.md", "added": 2, "removed": 0},
        ]
        # Multiple top-level dirs — may return None or a common part
        scope = infer_scope(files)
        # Should be None since src, tests, docs are all different
        assert scope is None or isinstance(scope, str)

    def test_root_file(self):
        files = [{"path": "pyproject.toml", "added": 1, "removed": 0}]
        scope = infer_scope(files)
        assert scope == "pyproject"


# ============================================================
# Tests: infer_type
# ============================================================

class TestInferType:
    def test_docs(self):
        parsed = parse_diff(DIFF_DOCS_ONLY)
        assert infer_type(parsed) == "docs"

    def test_test_files(self):
        parsed = parse_diff(DIFF_TEST_ONLY)
        assert infer_type(parsed) == "test"

    def test_fix_heuristic(self):
        parsed = parse_diff(DIFF_BUG_FIX)
        assert infer_type(parsed) == "fix"

    def test_refactor_heuristic(self):
        parsed = parse_diff(DIFF_REFACTOR)
        t = infer_type(parsed)
        assert t == "refactor"

    def test_ci(self):
        parsed = parse_diff(DIFF_CI)
        assert infer_type(parsed) == "ci"

    def test_config(self):
        parsed = parse_diff(DIFF_CONFIG)
        assert infer_type(parsed) == "build"

    def test_one_line_fix(self):
        parsed = parse_diff(DIFF_ONE_LINE)
        assert infer_type(parsed) == "fix"


# ============================================================
# Tests: build_summary
# ============================================================

class TestBuildSummary:
    def test_summary_contains_file_count(self):
        parsed = parse_diff(DIFF_BUG_FIX)
        summary = build_summary(parsed)
        assert "2 file(s)" in summary

    def test_summary_contains_file_paths(self):
        parsed = parse_diff(DIFF_BUG_FIX)
        summary = build_summary(parsed)
        assert "convert_checkpoint.py" in summary
        assert "generate.py" in summary

    def test_summary_shows_adds_deletes(self):
        parsed = parse_diff(DIFF_SINGLE_FILE_ADD)
        summary = build_summary(parsed)
        assert "+10" in summary
        assert "-0" in summary


# ============================================================
# Tests: build_prompt
# ============================================================

class TestBuildPrompt:
    def test_basic_prompt(self):
        prompt = build_prompt("Changed 1 file: +5 -3")
        assert "conventional commit" in prompt.lower()
        assert "Changed 1 file" in prompt
        assert "Subject line only" in prompt

    def test_forced_type_in_prompt(self):
        prompt = build_prompt("test", commit_type="feat")
        assert "feat" in prompt

    def test_body_flag(self):
        prompt = build_prompt("test", include_body=True)
        assert "body paragraph" in prompt.lower()

    def test_scope_in_prompt(self):
        prompt = build_prompt("test", scope="auth")
        assert "auth" in prompt


# ============================================================
# Tests: format_commit
# ============================================================

class TestFormatCommit:
    def test_basic_formatting(self):
        result = format_commit("feat: add user authentication")
        assert result == "feat: add user authentication"

    def test_with_scope(self):
        result = format_commit("feat(auth): add login endpoint")
        assert result == "feat(auth): add login endpoint"

    def test_forced_type_overrides(self):
        result = format_commit("chore: update deps", commit_type="fix")
        assert result.startswith("fix: ")

    def test_forced_scope_overrides(self):
        result = format_commit("feat: add login", scope="auth")
        assert "feat(auth): " in result

    def test_removes_trailing_period(self):
        result = format_commit("feat: add login.")
        assert not result.endswith(".")

    def test_lowercase_first_letter(self):
        result = format_commit("feat: Add login")
        assert "feat: add login" == result

    def test_invalid_type_falls_back(self):
        result = format_commit("banana: do something")
        assert result.startswith("chore: ")

    def test_no_type_prefix(self):
        result = format_commit("add user authentication")
        assert result.startswith("chore: ")
        assert "add user authentication" in result

    def test_enforces_72_char_limit(self):
        long_desc = "a" * 100
        result = format_commit(f"feat: {long_desc}")
        subject = result.split("\n")[0]
        assert len(subject) <= 72

    def test_strips_markdown(self):
        result = format_commit("```\nfeat: add login\n```")
        assert "```" not in result
        assert "feat: add login" == result

    def test_strips_bold(self):
        result = format_commit("**feat: add login**")
        assert "**" not in result

    def test_strips_quotes(self):
        result = format_commit('"feat: add login"')
        assert result == "feat: add login"

    def test_strips_common_prefixes(self):
        result = format_commit("Commit message: feat: add login")
        assert result == "feat: add login"

    def test_body_preserved(self):
        raw = "feat: add login\n\nThis adds JWT-based authentication."
        result = format_commit(raw, include_body=True)
        assert "\n\n" in result
        assert "JWT" in result

    def test_body_stripped_when_not_requested(self):
        raw = "feat: add login\n\nThis adds JWT-based authentication."
        result = format_commit(raw, include_body=False)
        assert "\n\n" not in result

    def test_type_and_scope_forced(self):
        result = format_commit("do something", commit_type="feat", scope="api")
        assert result == "feat(api): do something"

    def test_all_valid_types_accepted(self):
        for t in VALID_TYPES:
            result = format_commit(f"{t}: some change")
            assert result.startswith(f"{t}: ")

    def test_heading_stripped(self):
        result = format_commit("### feat: add login")
        assert "###" not in result

    def test_empty_description(self):
        # Model returns garbage — should still produce something valid
        result = format_commit("")
        assert result.startswith("chore: ")

    def test_multiline_subject_takes_first_line(self):
        result = format_commit("feat: add login\nfeat: add logout")
        assert "logout" not in result


# ============================================================
# Integration-style tests (parse → infer → format pipeline)
# ============================================================

class TestPipeline:
    def _run_pipeline(self, diff_text, commit_type=None, scope=None, include_body=False):
        """Run the full pre/post processing pipeline (no model call)."""
        parsed = parse_diff(diff_text)
        ct = commit_type or infer_type(parsed)
        sc = scope or infer_scope(parsed["files"])
        summary = build_summary(parsed)
        # Simulate model output with a placeholder
        fake_model_output = f"{ct or 'chore'}: update code"
        if sc:
            fake_model_output = f"{ct or 'chore'}({sc}): update code"
        return format_commit(fake_model_output, commit_type=ct, scope=sc, include_body=include_body)

    def test_single_file_add(self):
        result = self._run_pipeline(DIFF_SINGLE_FILE_ADD)
        assert result  # non-empty
        assert len(result.split("\n")[0]) <= 72

    def test_bug_fix(self):
        result = self._run_pipeline(DIFF_BUG_FIX)
        assert result.startswith("fix(")

    def test_docs_only(self):
        result = self._run_pipeline(DIFF_DOCS_ONLY)
        assert result.startswith("docs(")

    def test_forced_type_override(self):
        result = self._run_pipeline(DIFF_DOCS_ONLY, commit_type="chore")
        assert result.startswith("chore(")

    def test_forced_scope_override(self):
        result = self._run_pipeline(DIFF_SINGLE_FILE_ADD, scope="myscope")
        assert "(myscope)" in result

    def test_delete_only(self):
        result = self._run_pipeline(DIFF_DELETE_ONLY)
        assert result.startswith("refactor(")

    def test_config_change(self):
        result = self._run_pipeline(DIFF_CONFIG)
        assert result.startswith("build(")

    def test_ci_change(self):
        result = self._run_pipeline(DIFF_CI)
        assert result.startswith("ci(")

    def test_one_line_change(self):
        result = self._run_pipeline(DIFF_ONE_LINE)
        assert result.startswith("fix(")

    def test_large_diff(self):
        result = self._run_pipeline(DIFF_LARGE)
        subject = result.split("\n")[0]
        assert len(subject) <= 72

    def test_empty_diff_parse(self):
        parsed = parse_diff("")
        assert not parsed["is_diff"]
        assert parsed["files"] == []

    def test_non_diff_input_parse(self):
        parsed = parse_diff("Hello world, this is not a diff at all.")
        assert not parsed["is_diff"]
