"""bt-commit: Generate conventional commit messages from diffs using BitNet."""

import argparse
import os
import re
import subprocess
import sys

from ..engine import complete

VALID_TYPES = [
    "feat", "fix", "refactor", "docs", "test", "chore",
    "style", "perf", "ci", "build",
]

# ---------------------------------------------------------------------------
# Diff Parsing (pre-processing for the model)
# ---------------------------------------------------------------------------

def parse_diff(diff_text: str) -> dict:
    """Extract structured info from a unified diff.

    Returns a dict with:
      files      - list of {path, added, removed, changes}
      total_add  - total lines added
      total_del  - total lines deleted
      is_diff    - True if the input looks like a real diff
    """
    files: list[dict] = []
    current: dict | None = None
    total_add = 0
    total_del = 0
    is_diff = False

    for line in diff_text.splitlines():
        # New file header
        m = re.match(r"^diff --git a/(.*?) b/(.*)", line)
        if m:
            is_diff = True
            if current:
                files.append(current)
            current = {
                "path": m.group(2),
                "added": 0,
                "removed": 0,
                "changes": [],
            }
            continue

        # Fallback: +++ header (for plain unified diffs without git prefix)
        if line.startswith("+++ ") and current is None:
            is_diff = True
            path = line[4:].lstrip("b/").strip()
            current = {"path": path, "added": 0, "removed": 0, "changes": []}
            continue

        if current is None:
            # Also detect --- / +++ pair for non-git diffs
            if line.startswith("--- "):
                is_diff = True
                path = line[4:].lstrip("a/").strip()
                current = {"path": path, "added": 0, "removed": 0, "changes": []}
            continue

        if line.startswith("+") and not line.startswith("+++"):
            current["added"] += 1
            total_add += 1
            content = line[1:].strip()
            # Capture interesting additions (functions, classes, etc.)
            if _is_notable(content):
                current["changes"].append(f"+{content}")
        elif line.startswith("-") and not line.startswith("---"):
            current["removed"] += 1
            total_del += 1
            content = line[1:].strip()
            if _is_notable(content):
                current["changes"].append(f"-{content}")

    if current:
        files.append(current)

    return {
        "files": files,
        "total_add": total_add,
        "total_del": total_del,
        "is_diff": is_diff,
    }


def _is_notable(line: str) -> bool:
    """Return True if a diff line represents a notable code change."""
    patterns = [
        r"^\s*(def |class |function |const |let |var |export |import |from )",
        r"^\s*(async def |async function )",
        r"^\s*(pub fn |fn |impl |struct |enum |trait )",
        r"^\s*(type |interface )",
        r"^\s*#\s*(include|define|ifdef|pragma)",
    ]
    return any(re.search(p, line) for p in patterns)


# ---------------------------------------------------------------------------
# Scope Inference
# ---------------------------------------------------------------------------

def infer_scope(files: list[dict]) -> str | None:
    """Infer a short scope from file paths.

    Returns a single word, or None if ambiguous.
    """
    if not files:
        return None

    paths = [f["path"] for f in files]

    # Single file: use the filename stem (without extension)
    if len(paths) == 1:
        base = os.path.basename(paths[0])
        stem = os.path.splitext(base)[0]
        # Keep it short
        if len(stem) <= 20:
            return stem
        # Fall through to directory-based

    # Find deepest common directory
    normalized = [p.replace("\\", "/") for p in paths]
    try:
        common = os.path.commonpath(normalized)
    except ValueError:
        common = ""

    if common and common != ".":
        parts = common.replace("\\", "/").split("/")
        # Prefer the deepest part of the common path
        scope = parts[-1] if parts else None
        if scope and len(scope) <= 20:
            return scope

    # Fallback: check top-level directories
    top_dirs = set()
    for p in normalized:
        parts = p.split("/")
        if len(parts) > 1:
            top_dirs.add(parts[0])
        else:
            top_dirs.add(os.path.splitext(parts[0])[0])

    if len(top_dirs) == 1:
        scope = top_dirs.pop()
        if len(scope) <= 20:
            return scope

    return None


# ---------------------------------------------------------------------------
# Type Inference (heuristic, pre-model)
# ---------------------------------------------------------------------------

def infer_type(parsed: dict) -> str | None:
    """Heuristic type inference from file paths and change patterns."""
    paths = [f["path"] for f in parsed["files"]]
    all_paths = " ".join(paths).lower()

    # Documentation
    if all(
        p.endswith((".md", ".rst", ".txt"))
        or "/docs/" in p
        or p.startswith("docs/")
        for p in paths
    ):
        return "docs"

    # Tests
    if all("test" in p.lower() or "spec" in p.lower() for p in paths):
        return "test"

    # CI / GitHub Actions
    if all(
        ".github/" in p or "ci" in os.path.basename(p).lower()
        or p.endswith((".yml", ".yaml")) and ("ci" in p.lower() or "workflow" in p.lower())
        for p in paths
    ):
        return "ci"

    # Build / config files
    build_files = {
        "pyproject.toml", "setup.py", "setup.cfg", "package.json",
        "cargo.toml", "makefile", "cmake", "dockerfile",
        "requirements.txt", "go.mod", "go.sum",
    }
    if all(os.path.basename(p).lower() in build_files for p in paths):
        return "build"

    # Bug fix heuristic: small change, mostly modifications
    if (
        parsed["total_add"] < 10
        and parsed["total_del"] < 10
        and parsed["total_add"] > 0
        and parsed["total_del"] > 0
    ):
        return "fix"

    # Delete-heavy or restructuring: likely refactor
    if parsed["total_del"] > parsed["total_add"] and parsed["total_del"] > 5:
        return "refactor"

    return None


# ---------------------------------------------------------------------------
# Prompt Building
# ---------------------------------------------------------------------------

def build_summary(parsed: dict) -> str:
    """Build a concise text summary of the diff for the model."""
    lines = []
    lines.append(
        f"Changed {len(parsed['files'])} file(s): "
        f"+{parsed['total_add']} -{parsed['total_del']} lines"
    )

    for f in parsed["files"][:10]:  # cap at 10 files
        tag = ""
        if f["added"] > 0 and f["removed"] == 0:
            tag = " (new)"
        elif f["removed"] > 0 and f["added"] == 0:
            tag = " (deleted)"
        lines.append(f"  {f['path']}{tag}: +{f['added']} -{f['removed']}")
        for ch in f["changes"][:5]:  # cap notable changes
            lines.append(f"    {ch}")

    return "\n".join(lines)


def build_prompt(
    diff_summary: str,
    commit_type: str | None = None,
    scope: str | None = None,
    include_body: bool = False,
) -> str:
    """Build the prompt for the model."""
    parts = []
    parts.append(
        "Given the change summary below, write a single conventional commit message. "
        "Format: type(scope): description. "
        "Use imperative mood, lowercase description, no period."
    )

    if not include_body:
        parts.append("Output ONLY the commit message line, nothing else.")

    parts.append(f"\n{diff_summary}")

    # Build the start of the commit message so the model just completes it
    prefix = ""
    if commit_type and scope:
        prefix = f"{commit_type}({scope}): "
    elif commit_type:
        prefix = f"{commit_type}: "
    elif scope:
        prefix = f"chore({scope}): "

    if prefix:
        parts.append(f"\n{prefix}")
    else:
        parts.append("\nCommit message:")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def format_commit(
    raw_output: str,
    commit_type: str | None = None,
    scope: str | None = None,
    include_body: bool = False,
) -> str:
    """Post-process model output into a valid conventional commit message."""
    text = raw_output.strip()

    # Remove markdown formatting
    text = re.sub(r"```[\w]*\n?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # Remove common prefixes the model might add
    for prefix in ("Commit message:", "commit:", "Message:", "Subject:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Remove wrapping quotes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Split into subject and body
    parts = re.split(r"\n\n+", text, maxsplit=1)
    subject = parts[0].strip()
    body = parts[1].strip() if len(parts) > 1 else ""

    # Take only the first line of subject
    subject = subject.split("\n")[0].strip()

    # Parse existing type(scope): prefix if present
    m = re.match(r"^(\w+)(?:\(([^)]*)\))?:\s*(.*)", subject)
    if m:
        existing_type = m.group(1).lower()
        existing_scope = m.group(2)
        description = m.group(3)

        # Use forced type/scope if provided, else keep existing
        final_type = commit_type or (existing_type if existing_type in VALID_TYPES else None)
        final_scope = scope or existing_scope
    else:
        description = subject
        final_type = commit_type
        final_scope = scope

    # Clean description
    description = description.strip()
    if description:
        # Lowercase first letter
        description = description[0].lower() + description[1:]
        # Remove trailing period
        description = description.rstrip(".")
        # Remove leading dash/bullet
        description = re.sub(r"^[-*•]\s*", "", description)

    # Assign a default type if none
    if not final_type:
        final_type = "chore"

    # Validate type
    if final_type not in VALID_TYPES:
        final_type = "chore"

    # Build subject line
    if final_scope:
        subject_line = f"{final_type}({final_scope}): {description}"
    else:
        subject_line = f"{final_type}: {description}"

    # Enforce 72-char limit on subject
    if len(subject_line) > 72:
        # Truncate description to fit
        prefix_len = len(f"{final_type}({final_scope}): " if final_scope else f"{final_type}: ")
        max_desc = 72 - prefix_len
        if max_desc > 10:
            description = description[:max_desc].rstrip()
            # Don't cut mid-word
            if " " in description:
                description = description[:description.rfind(" ")]
            description = description.rstrip(".")
        if final_scope:
            subject_line = f"{final_type}({final_scope}): {description}"
        else:
            subject_line = f"{final_type}: {description}"

    # Assemble final message
    if include_body and body:
        return f"{subject_line}\n\n{body}"
    return subject_line


# ---------------------------------------------------------------------------
# Input Handling
# ---------------------------------------------------------------------------

def get_input(args) -> str:
    """Read diff from stdin, file, or --amend mode."""
    if args.amend:
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD~1"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                print(f"Error: git diff HEAD~1 failed: {result.stderr.strip()}", file=sys.stderr)
                sys.exit(1)
            return result.stdout
        except FileNotFoundError:
            print("Error: git not found", file=sys.stderr)
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print("Error: git diff timed out", file=sys.stderr)
            sys.exit(1)

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

    print("Error: no input (pipe a diff or use -i FILE / --amend)", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="bt-commit",
        description="Generate conventional commit messages from diffs",
        epilog="Example: git diff --staged | bt-commit",
    )
    parser.add_argument(
        "--type", "-t", choices=VALID_TYPES, default=None,
        help="Force commit type (feat, fix, refactor, ...)",
    )
    parser.add_argument(
        "--scope", "-s", default=None,
        help="Force scope",
    )
    parser.add_argument(
        "--body", "-b", action="store_true",
        help="Include a body paragraph (default: subject line only)",
    )
    parser.add_argument(
        "--amend", action="store_true",
        help="Read from git diff HEAD~1 instead of stdin",
    )
    parser.add_argument(
        "--input", "-i", default=None,
        help="Read diff from file",
    )
    args = parser.parse_args()

    diff_text = get_input(args).strip()
    if not diff_text:
        print("Error: empty diff", file=sys.stderr)
        sys.exit(1)

    # Pre-process
    parsed = parse_diff(diff_text)

    if not parsed["is_diff"]:
        print("Warning: input doesn't look like a diff, proceeding anyway", file=sys.stderr)

    # Determine type and scope
    commit_type = args.type or infer_type(parsed)
    scope_val = args.scope or infer_scope(parsed["files"])

    # Build prompt
    summary = build_summary(parsed)
    prompt = build_prompt(summary, commit_type=commit_type, scope=scope_val, include_body=args.body)

    # Inference
    max_tokens = 80 if args.body else 40
    stop = None if args.body else "\n"
    raw = complete(prompt, max_tokens=max_tokens, temperature=0.0, stop_at=stop)

    # Post-process
    result = format_commit(raw, commit_type=commit_type, scope=scope_val, include_body=args.body)
    print(result)


if __name__ == "__main__":
    main()
