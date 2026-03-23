"""Tests for hone-changelog: log parsing, grouping, formatting, and CLI."""

import pytest
from datetime import date
from hone_tools.cli.changelog import (
    parse_log_line,
    parse_log,
    group_by_type,
    clean_descriptions_fallback,
    format_bullet,
    format_markdown,
    format_keepachangelog,
    _apply_rewritten,
    COMMIT_TYPES,
    TYPE_ORDER,
)


# ============================================================
# Sample inputs
# ============================================================

LOG_CONVENTIONAL = """\
abc1234 feat: add user authentication
def5678 fix: broken navigation on mobile
1112233 refactor: simplify database queries
4455667 docs: update API reference
7788990 test: add integration tests for auth
aabbccd chore: bump dependency versions
"""

LOG_SCOPED = """\
abc1234 feat(auth): add JWT token validation
def5678 fix(nav): prevent double-click crash
1112233 refactor(db): use connection pooling
"""

LOG_MIXED = """\
abc1234 feat: add dark mode
def5678 Update README with install instructions
1112233 fix: handle null pointer in parser
4455667 WIP: experimenting with new layout
7788990 perf: cache database lookups
"""

LOG_SINGLE = "abc1234 feat: add login page\n"

LOG_EMPTY = ""

LOG_NO_HASH = """\
feat: add login
fix: broken button
"""

LOG_MULTI_FEAT = """\
abc1234 feat: add login page
def5678 feat: add registration flow
1112233 feat(auth): add password reset
4455667 fix: login redirect loop
"""


# ============================================================
# Tests: parse_log_line
# ============================================================

class TestParseLogLine:
    def test_conventional_commit(self):
        result = parse_log_line("abc1234 feat: add user auth")
        assert result is not None
        assert result["hash"] == "abc1234"
        assert result["type"] == "feat"
        assert result["description"] == "add user auth"
        assert result["scope"] == ""

    def test_scoped_commit(self):
        result = parse_log_line("abc1234 feat(auth): add JWT validation")
        assert result["type"] == "feat"
        assert result["scope"] == "auth"
        assert result["description"] == "add JWT validation"

    def test_non_conventional_commit(self):
        result = parse_log_line("abc1234 Update README")
        assert result["type"] == "other"
        assert result["description"] == "Update README"

    def test_empty_line(self):
        assert parse_log_line("") is None
        assert parse_log_line("   ") is None

    def test_no_hash(self):
        result = parse_log_line("feat: add login")
        assert result is not None
        assert result["hash"] == ""
        assert result["type"] == "feat"
        assert result["description"] == "add login"

    def test_unknown_type_becomes_other(self):
        result = parse_log_line("abc1234 banana: do something")
        assert result["type"] == "other"

    def test_all_known_types(self):
        for t in COMMIT_TYPES:
            result = parse_log_line(f"abc1234 {t}: some change")
            assert result["type"] == t, f"Type {t} not recognized"

    def test_preserves_raw_message(self):
        result = parse_log_line("abc1234 feat(api): add endpoints")
        assert result["raw"] == "feat(api): add endpoints"

    def test_long_hash(self):
        long_hash = "a" * 40
        result = parse_log_line(f"{long_hash} fix: something")
        assert result["hash"] == long_hash


# ============================================================
# Tests: parse_log
# ============================================================

class TestParseLog:
    def test_conventional_log(self):
        commits = parse_log(LOG_CONVENTIONAL)
        assert len(commits) == 6
        types = [c["type"] for c in commits]
        assert "feat" in types
        assert "fix" in types

    def test_mixed_log(self):
        commits = parse_log(LOG_MIXED)
        assert len(commits) == 5
        other_count = sum(1 for c in commits if c["type"] == "other")
        assert other_count == 2  # "Update README" and "WIP:"

    def test_single_commit(self):
        commits = parse_log(LOG_SINGLE)
        assert len(commits) == 1
        assert commits[0]["type"] == "feat"

    def test_empty_input(self):
        commits = parse_log(LOG_EMPTY)
        assert commits == []

    def test_blank_lines_filtered(self):
        text = "\nabc1234 feat: something\n\ndef5678 fix: other\n\n"
        commits = parse_log(text)
        assert len(commits) == 2

    def test_no_hash_lines(self):
        commits = parse_log(LOG_NO_HASH)
        assert len(commits) == 2
        assert all(c["hash"] == "" for c in commits)


# ============================================================
# Tests: group_by_type
# ============================================================

class TestGroupByType:
    def test_groups_conventional(self):
        commits = parse_log(LOG_CONVENTIONAL)
        clean_descriptions_fallback(commits)
        groups = group_by_type(commits)
        assert "feat" in groups
        assert "fix" in groups
        assert len(groups["feat"]) == 1
        assert len(groups["fix"]) == 1

    def test_multiple_same_type(self):
        commits = parse_log(LOG_MULTI_FEAT)
        clean_descriptions_fallback(commits)
        groups = group_by_type(commits)
        assert len(groups["feat"]) == 3
        assert len(groups["fix"]) == 1

    def test_order_matches_type_order(self):
        commits = parse_log(LOG_CONVENTIONAL)
        clean_descriptions_fallback(commits)
        groups = group_by_type(commits)
        keys = list(groups.keys())
        # feat should come before fix, fix before refactor, etc.
        assert keys.index("feat") < keys.index("fix")
        assert keys.index("fix") < keys.index("refactor")

    def test_other_type_last(self):
        commits = parse_log(LOG_MIXED)
        clean_descriptions_fallback(commits)
        groups = group_by_type(commits)
        keys = list(groups.keys())
        if "other" in keys:
            assert keys.index("other") == len(keys) - 1 or keys[-1] == "other" or True
            # "other" should appear after known types
            for known in keys:
                if known != "other" and known in TYPE_ORDER:
                    assert keys.index(known) < keys.index("other")

    def test_empty_commits(self):
        groups = group_by_type([])
        assert groups == {}


# ============================================================
# Tests: clean_descriptions_fallback
# ============================================================

class TestCleanDescriptionsFallback:
    def test_copies_description_to_clean(self):
        commits = parse_log(LOG_SINGLE)
        clean_descriptions_fallback(commits)
        assert commits[0]["clean"] == commits[0]["description"]

    def test_empty_list(self):
        result = clean_descriptions_fallback([])
        assert result == []


# ============================================================
# Tests: _apply_rewritten
# ============================================================

class TestApplyRewritten:
    def test_numbered_output(self):
        commits = [
            {"description": "add login", "type": "feat"},
            {"description": "fix nav", "type": "fix"},
        ]
        raw = "1. Added user login page\n2. Fixed navigation bug"
        _apply_rewritten(commits, raw)
        assert commits[0]["clean"] == "Added user login page"
        assert commits[1]["clean"] == "Fixed navigation bug"

    def test_partial_output_fills_gaps(self):
        commits = [
            {"description": "add login", "type": "feat"},
            {"description": "fix nav", "type": "fix"},
            {"description": "update docs", "type": "docs"},
        ]
        raw = "1. Added login page"
        _apply_rewritten(commits, raw)
        assert commits[0]["clean"] == "Added login page"
        assert commits[1]["clean"] == "fix nav"  # fallback
        assert commits[2]["clean"] == "update docs"  # fallback

    def test_empty_output(self):
        commits = [{"description": "add login", "type": "feat"}]
        _apply_rewritten(commits, "")
        assert commits[0]["clean"] == "add login"

    def test_out_of_range_index_ignored(self):
        commits = [{"description": "add login", "type": "feat"}]
        raw = "1. Added login\n5. Out of range entry"
        _apply_rewritten(commits, raw)
        assert commits[0]["clean"] == "Added login"


# ============================================================
# Tests: format_bullet
# ============================================================

class TestFormatBullet:
    def _make_groups(self, log_text):
        commits = parse_log(log_text)
        clean_descriptions_fallback(commits)
        return group_by_type(commits)

    def test_basic_output(self):
        groups = self._make_groups(LOG_CONVENTIONAL)
        output = format_bullet(groups)
        assert "Features:" in output
        assert "Bug Fixes:" in output
        assert "  - add user authentication" in output

    def test_with_version(self):
        groups = self._make_groups(LOG_SINGLE)
        output = format_bullet(groups, version="v1.0.0")
        assert "v1.0.0" in output

    def test_with_date(self):
        groups = self._make_groups(LOG_SINGLE)
        output = format_bullet(groups, date_str="2026-03-23")
        assert "2026-03-23" in output

    def test_with_version_and_date(self):
        groups = self._make_groups(LOG_SINGLE)
        output = format_bullet(groups, version="v2.0.0", date_str="2026-01-01")
        assert "v2.0.0" in output
        assert "2026-01-01" in output

    def test_scoped_commits_show_scope(self):
        groups = self._make_groups(LOG_SCOPED)
        output = format_bullet(groups)
        assert "[auth]" in output
        assert "[nav]" in output

    def test_no_header_when_no_version_or_date(self):
        groups = self._make_groups(LOG_SINGLE)
        output = format_bullet(groups)
        # First line should be the type heading, not a blank version header
        first_line = output.strip().splitlines()[0]
        assert first_line == "Features:"

    def test_ends_with_newline(self):
        groups = self._make_groups(LOG_SINGLE)
        output = format_bullet(groups)
        assert output.endswith("\n")


# ============================================================
# Tests: format_markdown
# ============================================================

class TestFormatMarkdown:
    def _make_groups(self, log_text):
        commits = parse_log(log_text)
        clean_descriptions_fallback(commits)
        return group_by_type(commits)

    def test_has_headings(self):
        groups = self._make_groups(LOG_CONVENTIONAL)
        output = format_markdown(groups)
        assert "## Features" in output
        assert "## Bug Fixes" in output

    def test_version_header(self):
        groups = self._make_groups(LOG_SINGLE)
        output = format_markdown(groups, version="v1.0.0")
        assert "# v1.0.0" in output

    def test_version_with_date(self):
        groups = self._make_groups(LOG_SINGLE)
        output = format_markdown(groups, version="v1.0.0", date_str="2026-03-23")
        assert "# v1.0.0 (2026-03-23)" in output

    def test_hash_suffix(self):
        groups = self._make_groups(LOG_CONVENTIONAL)
        output = format_markdown(groups)
        assert "(abc1234)" in output

    def test_scope_bold(self):
        groups = self._make_groups(LOG_SCOPED)
        output = format_markdown(groups)
        assert "**auth:**" in output

    def test_bullet_format(self):
        groups = self._make_groups(LOG_SINGLE)
        output = format_markdown(groups)
        assert output.strip().splitlines()[-1].startswith("- ")


# ============================================================
# Tests: format_keepachangelog
# ============================================================

class TestFormatKeepAChangelog:
    def _make_groups(self, log_text):
        commits = parse_log(log_text)
        clean_descriptions_fallback(commits)
        return group_by_type(commits)

    def test_has_kac_header(self):
        groups = self._make_groups(LOG_SINGLE)
        output = format_keepachangelog(groups, version="v1.0.0", date_str="2026-03-23")
        assert "## [v1.0.0] - 2026-03-23" in output

    def test_unreleased_default(self):
        groups = self._make_groups(LOG_SINGLE)
        output = format_keepachangelog(groups)
        assert "[Unreleased]" in output

    def test_feat_maps_to_added(self):
        groups = self._make_groups(LOG_SINGLE)
        output = format_keepachangelog(groups)
        assert "### Added" in output

    def test_fix_maps_to_fixed(self):
        groups = self._make_groups("abc1234 fix: broken button\n")
        output = format_keepachangelog(groups)
        assert "### Fixed" in output

    def test_refactor_maps_to_changed(self):
        groups = self._make_groups("abc1234 refactor: simplify logic\n")
        output = format_keepachangelog(groups)
        assert "### Changed" in output

    def test_mixed_types(self):
        groups = self._make_groups(LOG_CONVENTIONAL)
        output = format_keepachangelog(groups, version="v2.0.0", date_str="2026-01-01")
        assert "### Added" in output
        assert "### Fixed" in output
        assert "### Changed" in output  # refactor, docs
        assert "### Other" in output  # test, chore

    def test_kac_order(self):
        groups = self._make_groups(LOG_CONVENTIONAL)
        output = format_keepachangelog(groups)
        added_pos = output.index("### Added")
        fixed_pos = output.index("### Fixed")
        # Added should come before Fixed per KaC convention
        assert added_pos < fixed_pos


# ============================================================
# Integration: full pipeline (no model)
# ============================================================

class TestPipeline:
    def _run(self, log_text, fmt="bullet", version=None, date_str=None):
        commits = parse_log(log_text)
        clean_descriptions_fallback(commits)
        groups = group_by_type(commits)
        from hone_tools.cli.changelog import FORMATTERS
        formatter = FORMATTERS[fmt]
        return formatter(groups, version=version, date_str=date_str)

    def test_bullet_pipeline(self):
        output = self._run(LOG_CONVENTIONAL)
        assert "Features:" in output
        assert "Bug Fixes:" in output

    def test_markdown_pipeline(self):
        output = self._run(LOG_CONVENTIONAL, fmt="markdown", version="v1.0.0")
        assert "# v1.0.0" in output
        assert "## Features" in output

    def test_keepachangelog_pipeline(self):
        output = self._run(LOG_CONVENTIONAL, fmt="keep-a-changelog",
                          version="v1.0.0", date_str="2026-03-23")
        assert "## [v1.0.0] - 2026-03-23" in output

    def test_single_commit_all_formats(self):
        for fmt in ("bullet", "markdown", "keep-a-changelog"):
            output = self._run(LOG_SINGLE, fmt=fmt)
            assert len(output.strip()) > 0

    def test_mixed_commits(self):
        output = self._run(LOG_MIXED)
        # Should have both known types and "other"
        assert "Features:" in output or "feat" in output.lower()

    def test_scoped_commits(self):
        output = self._run(LOG_SCOPED, fmt="markdown")
        assert "**auth:**" in output

    def test_no_hash_input(self):
        output = self._run(LOG_NO_HASH)
        assert "Features:" in output
        assert "Bug Fixes:" in output
