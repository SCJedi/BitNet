"""hone-changelog: Convert git log output into formatted release notes/changelog.

Hybrid approach:
  - Pre-process: parse git log lines, detect conventional commit types
  - Group: group commits by type using Python (not the model)
  - Model: ONLY used for cleaning up commit descriptions into human-readable notes
  - Post-process: format into the selected output style

Usage:
  git log v1.0..v1.1 --oneline | hone-changelog
  git log --oneline -20 | hone-changelog
  git log --oneline -10 | hone-changelog --format bullet
"""

import argparse
import re
import sys
from datetime import date

from ..engine import complete

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMIT_TYPES = {
    "feat":     "Features",
    "fix":      "Bug Fixes",
    "refactor": "Refactoring",
    "docs":     "Documentation",
    "test":     "Tests",
    "chore":    "Chores",
    "style":    "Style",
    "perf":     "Performance",
    "ci":       "CI",
    "build":    "Build",
    "revert":   "Reverts",
}

# Order for display (most interesting first)
TYPE_ORDER = [
    "feat", "fix", "perf", "refactor", "docs",
    "test", "build", "ci", "style", "chore", "revert", "other",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_log_line(line: str) -> dict | None:
    """Parse a single git log --oneline line into structured data.

    Expected format: <hash> <message>
    Returns dict with keys: hash, type, scope, description, raw
    Returns None for unparseable lines.
    """
    line = line.strip()
    if not line:
        return None

    # Split into hash and message
    # Hash can be 4-40 hex chars; git log --oneline typically uses 7+
    # but abbreviation length varies by config and repo size
    m = re.match(r"^([0-9a-f]{4,40})\s+(.+)$", line, re.IGNORECASE)
    if m:
        commit_hash = m.group(1)
        message = m.group(2)
    else:
        # No hash prefix — treat whole line as message
        commit_hash = ""
        message = line

    # Try to parse conventional commit: type(scope): description
    cm = re.match(
        r"^(\w+?)(?:\(([^)]*)\))?:\s*(.+)$",
        message,
    )
    if cm:
        raw_type = cm.group(1).lower()
        scope = cm.group(2) or ""
        description = cm.group(3).strip()
        # Validate type
        commit_type = raw_type if raw_type in COMMIT_TYPES else "other"
    else:
        commit_type = "other"
        scope = ""
        description = message.strip()

    return {
        "hash": commit_hash,
        "type": commit_type,
        "scope": scope,
        "description": description,
        "raw": message,
    }


def parse_log(text: str) -> list[dict]:
    """Parse multi-line git log --oneline output into a list of commits."""
    commits = []
    for line in text.splitlines():
        parsed = parse_log_line(line)
        if parsed:
            commits.append(parsed)
    return commits


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_by_type(commits: list[dict]) -> dict[str, list[dict]]:
    """Group commits by their conventional commit type.

    Returns an ordered dict (by TYPE_ORDER) with non-empty groups only.
    """
    groups: dict[str, list[dict]] = {}
    for commit in commits:
        t = commit["type"]
        if t not in groups:
            groups[t] = []
        groups[t].append(commit)

    # Return in display order
    ordered: dict[str, list[dict]] = {}
    for t in TYPE_ORDER:
        if t in groups:
            ordered[t] = groups[t]
    # Any remaining types not in TYPE_ORDER
    for t in groups:
        if t not in ordered:
            ordered[t] = groups[t]
    return ordered


# ---------------------------------------------------------------------------
# Model Integration (light rewriting)
# ---------------------------------------------------------------------------

def rewrite_descriptions(commits: list[dict]) -> list[dict]:
    """Use the model to clean up commit descriptions into human-readable notes.

    Batches all descriptions into a single prompt for efficiency.
    Modifies commits in-place, adding a 'clean' key.
    """
    if not commits:
        return commits

    # Build a batch prompt
    lines = []
    for i, c in enumerate(commits):
        lines.append(f"{i + 1}. {c['description']}")

    batch_text = "\n".join(lines)
    prompt = (
        "Rewrite each commit description below into a clean, human-readable "
        "release note. Use past tense. Keep each on one line. "
        "Output ONLY the numbered list, nothing else.\n\n"
        f"{batch_text}\n\n"
        "Rewritten:\n"
    )

    max_tokens = min(400, max(80, len(commits) * 30))

    try:
        raw = complete(prompt, max_tokens=max_tokens, temperature=0.0, stop_at=None)
        _apply_rewritten(commits, raw)
    except (SystemExit, Exception):
        # Model unavailable — fall back to raw descriptions
        for c in commits:
            if "clean" not in c:
                c["clean"] = c["description"]

    return commits


def _apply_rewritten(commits: list[dict], raw_output: str) -> None:
    """Parse numbered list from model output and apply to commits."""
    lines = raw_output.strip().splitlines()
    for line in lines:
        m = re.match(r"^\s*(\d+)\.\s*(.+)$", line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(commits):
                commits[idx]["clean"] = m.group(2).strip()

    # Fill any gaps with raw description
    for c in commits:
        if "clean" not in c:
            c["clean"] = c["description"]


def clean_descriptions_fallback(commits: list[dict]) -> list[dict]:
    """Fallback: use raw descriptions without model (for --no-model or testing)."""
    for c in commits:
        c["clean"] = c["description"]
    return commits


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _type_heading(commit_type: str) -> str:
    """Get the human-readable heading for a commit type."""
    return COMMIT_TYPES.get(commit_type, commit_type.title())


def format_bullet(
    groups: dict[str, list[dict]],
    version: str | None = None,
    date_str: str | None = None,
) -> str:
    """Format as a simple bullet list grouped by type."""
    lines = []

    # Header
    if version or date_str:
        header_parts = []
        if version:
            header_parts.append(version)
        if date_str:
            header_parts.append(f"({date_str})")
        lines.append(" ".join(header_parts))
        lines.append("")

    for commit_type, commits in groups.items():
        heading = _type_heading(commit_type)
        lines.append(f"{heading}:")
        for c in commits:
            desc = c.get("clean", c["description"])
            scope_prefix = f"[{c['scope']}] " if c["scope"] else ""
            lines.append(f"  - {scope_prefix}{desc}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def format_markdown(
    groups: dict[str, list[dict]],
    version: str | None = None,
    date_str: str | None = None,
) -> str:
    """Format as Markdown with ## headings per type."""
    lines = []

    # Header
    if version:
        header = f"# {version}"
        if date_str:
            header += f" ({date_str})"
        lines.append(header)
        lines.append("")
    elif date_str:
        lines.append(f"# {date_str}")
        lines.append("")

    for commit_type, commits in groups.items():
        heading = _type_heading(commit_type)
        lines.append(f"## {heading}")
        lines.append("")
        for c in commits:
            desc = c.get("clean", c["description"])
            scope_prefix = f"**{c['scope']}:** " if c["scope"] else ""
            hash_suffix = f" ({c['hash'][:7]})" if c["hash"] else ""
            lines.append(f"- {scope_prefix}{desc}{hash_suffix}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def format_keepachangelog(
    groups: dict[str, list[dict]],
    version: str | None = None,
    date_str: str | None = None,
) -> str:
    """Format following Keep a Changelog conventions.

    See https://keepachangelog.com/
    Maps conventional commit types to KaC categories:
      feat → Added, fix → Fixed, perf → Changed,
      refactor → Changed, docs → Changed, revert → Removed,
      build/ci/test/style/chore → Other
    """
    KAC_MAP = {
        "feat": "Added",
        "fix": "Fixed",
        "perf": "Changed",
        "refactor": "Changed",
        "docs": "Changed",
        "revert": "Removed",
        "style": "Other",
        "chore": "Other",
        "build": "Other",
        "ci": "Other",
        "test": "Other",
        "other": "Other",
    }
    KAC_ORDER = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security", "Other"]

    # Re-group into KaC categories
    kac_groups: dict[str, list[dict]] = {}
    for commit_type, commits in groups.items():
        category = KAC_MAP.get(commit_type, "Other")
        if category not in kac_groups:
            kac_groups[category] = []
        kac_groups[category].extend(commits)

    lines = []

    # Header
    v = version or "Unreleased"
    d = date_str or date.today().isoformat()
    lines.append(f"## [{v}] - {d}")
    lines.append("")

    for category in KAC_ORDER:
        if category in kac_groups:
            lines.append(f"### {category}")
            lines.append("")
            for c in kac_groups[category]:
                desc = c.get("clean", c["description"])
                scope_prefix = f"**{c['scope']}:** " if c["scope"] else ""
                lines.append(f"- {scope_prefix}{desc}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


FORMATTERS = {
    "bullet": format_bullet,
    "markdown": format_markdown,
    "keep-a-changelog": format_keepachangelog,
}


# ---------------------------------------------------------------------------
# Input Handling
# ---------------------------------------------------------------------------

def get_input(args) -> str:
    """Read git log from stdin or file."""
    if args.input:
        try:
            with open(args.input) as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        except PermissionError:
            print(f"Error: permission denied: {args.input}", file=sys.stderr)
            sys.exit(1)

    if not sys.stdin.isatty():
        return sys.stdin.read()

    print(
        "Error: no input (pipe git log output or use -i FILE)",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="hone-changelog",
        description="Convert git log output into formatted release notes",
        epilog="Example: git log v1.0..v1.1 --oneline | hone-changelog",
    )
    parser.add_argument(
        "--format", "-f",
        choices=list(FORMATTERS.keys()),
        default="bullet",
        help="Output format (default: bullet)",
    )
    parser.add_argument(
        "--version", "-v",
        default=None,
        help="Version label for the header (e.g., 'v2.1.0')",
    )
    parser.add_argument(
        "--date", "-d",
        default=None,
        help="Date for the header (default: today)",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Read log from file instead of stdin",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip model rewriting (use raw commit descriptions)",
    )
    args = parser.parse_args()

    text = get_input(args).strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    # Parse
    commits = parse_log(text)
    if not commits:
        print("Error: no commits found in input", file=sys.stderr)
        sys.exit(1)

    # Clean descriptions
    if args.no_model:
        clean_descriptions_fallback(commits)
    else:
        rewrite_descriptions(commits)

    # Group
    groups = group_by_type(commits)

    # Format
    formatter = FORMATTERS[args.format]
    date_str = args.date or date.today().isoformat()
    output = formatter(groups, version=args.version, date_str=date_str)
    print(output, end="")


if __name__ == "__main__":
    main()
