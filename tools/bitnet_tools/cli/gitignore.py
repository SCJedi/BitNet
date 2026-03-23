"""bt-gitignore: Generate .gitignore rules from natural language or technology keywords.

Template-first approach: built-in templates for common technologies,
model fallback only for unusual/custom requests.

Usage:
  echo "python flask docker" | bt-gitignore          -> comprehensive .gitignore
  echo "node react typescript" | bt-gitignore         -> node-focused .gitignore
  echo "add rules for compiled C files" | bt-gitignore --append  -> just the new rules
"""

import argparse
import re
import sys

from ..engine import complete


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

COMMON = [
    "# OS files",
    ".DS_Store",
    "Thumbs.db",
    "Desktop.ini",
    "ehthumbs.db",
    "",
    "# Editor/IDE",
    "*.swp",
    "*.swo",
    "*~",
    "",
    "# Environment & secrets",
    ".env",
    ".env.local",
    ".env.*.local",
    "",
    "# Logs",
    "*.log",
    "npm-debug.log*",
    "yarn-debug.log*",
    "yarn-error.log*",
]

TEMPLATES = {
    "python": [
        "# Python",
        "__pycache__/",
        "*.py[cod]",
        "*$py.class",
        "*.so",
        "*.egg-info/",
        "dist/",
        "build/",
        "eggs/",
        "*.egg",
        ".Python",
        "develop-eggs/",
        "lib/",
        "lib64/",
        "parts/",
        "sdist/",
        "var/",
        "wheels/",
        "*.manifest",
        "*.spec",
        "pip-log.txt",
        "pip-delete-this-directory.txt",
        ".venv/",
        "venv/",
        "ENV/",
        "env/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        "htmlcov/",
        ".coverage",
        ".coverage.*",
        "coverage.xml",
        "*.cover",
        ".tox/",
        ".nox/",
        "*.pyo",
        "*.pyd",
        ".hypothesis/",
        ".pytype/",
    ],
    "node": [
        "# Node.js",
        "node_modules/",
        "npm-debug.log*",
        "yarn-debug.log*",
        "yarn-error.log*",
        ".pnpm-debug.log*",
        "package-lock.json",
        "yarn.lock",
        ".npm",
        ".yarn/cache",
        ".yarn/unplugged",
        ".yarn/install-state.gz",
        ".pnp.*",
        "dist/",
        "build/",
        ".next/",
        "out/",
        ".nuxt/",
        ".cache/",
        ".parcel-cache/",
        ".turbo/",
    ],
    "javascript": [
        "# JavaScript",
        "node_modules/",
        "dist/",
        "build/",
        ".cache/",
        "coverage/",
        "*.min.js",
        "*.bundle.js",
    ],
    "typescript": [
        "# TypeScript",
        "*.js.map",
        "*.d.ts",
        "tsconfig.tsbuildinfo",
        "dist/",
        "build/",
        "out/",
    ],
    "react": [
        "# React",
        "node_modules/",
        "build/",
        ".env.local",
        ".env.development.local",
        ".env.test.local",
        ".env.production.local",
        "coverage/",
    ],
    "java": [
        "# Java",
        "*.class",
        "*.jar",
        "*.war",
        "*.ear",
        "*.nar",
        "target/",
        ".gradle/",
        "build/",
        "gradle-app.setting",
        "!gradle-wrapper.jar",
        ".gradletasknamecache",
        "hs_err_pid*",
        "replay_pid*",
        ".settings/",
        ".classpath",
        ".project",
        "bin/",
    ],
    "go": [
        "# Go",
        "*.exe",
        "*.exe~",
        "*.dll",
        "*.so",
        "*.dylib",
        "*.test",
        "*.out",
        "vendor/",
        "go.work",
    ],
    "rust": [
        "# Rust",
        "target/",
        "Cargo.lock",
        "**/*.rs.bk",
        "*.pdb",
    ],
    "c": [
        "# C/C++",
        "*.o",
        "*.obj",
        "*.so",
        "*.dll",
        "*.dylib",
        "*.a",
        "*.lib",
        "*.exe",
        "*.out",
        "*.app",
        "*.dSYM/",
        "*.su",
        "*.idb",
        "*.pdb",
        "*.d",
        "*.gcno",
        "*.gcda",
        "*.gcov",
        "CMakeFiles/",
        "CMakeCache.txt",
        "cmake_install.cmake",
        "Makefile",
        "build/",
    ],
    "cpp": None,  # alias for c
    "ruby": [
        "# Ruby",
        "*.gem",
        "*.rbc",
        ".bundle/",
        "vendor/bundle",
        ".config",
        "coverage/",
        "InstalledFiles",
        "pkg/",
        "spec/reports",
        "test/tmp",
        "test/version_tmp",
        "tmp/",
        ".byebug_history",
        ".ruby-version",
        ".ruby-gemset",
        "Gemfile.lock",
    ],
    "php": [
        "# PHP",
        "vendor/",
        "composer.lock",
        "*.cache",
        ".phpunit.result.cache",
        ".php_cs.cache",
        ".php-cs-fixer.cache",
        "storage/",
        "bootstrap/cache/",
    ],
    "docker": [
        "# Docker",
        ".docker/",
        "docker-compose.override.yml",
        "*.pid",
    ],
    "vscode": [
        "# VS Code",
        ".vscode/",
        "*.code-workspace",
        ".history/",
    ],
    "jetbrains": [
        "# JetBrains",
        ".idea/",
        "*.iml",
        "*.iws",
        "*.ipr",
        "out/",
        "cmake-build-*/",
    ],
    "vim": [
        "# Vim",
        "*.swp",
        "*.swo",
        "*~",
        "Session.vim",
        ".netrwhist",
        "tags",
    ],
    "macos": [
        "# macOS",
        ".DS_Store",
        ".AppleDouble",
        ".LSOverride",
        "Icon\\r",
        "._*",
        ".Spotlight-V100",
        ".Trashes",
    ],
    "windows": [
        "# Windows",
        "Thumbs.db",
        "Thumbs.db:encryptable",
        "ehthumbs.db",
        "ehthumbs_vista.db",
        "Desktop.ini",
        "$RECYCLE.BIN/",
        "*.lnk",
    ],
    "linux": [
        "# Linux",
        "*~",
        ".fuse_hidden*",
        ".directory",
        ".Trash-*",
        ".nfs*",
    ],
    "flask": [
        "# Flask",
        "instance/",
        ".webassets-cache",
    ],
    "django": [
        "# Django",
        "*.pyc",
        "db.sqlite3",
        "db.sqlite3-journal",
        "media/",
        "staticfiles/",
        "local_settings.py",
    ],
    "terraform": [
        "# Terraform",
        ".terraform/",
        ".terraform.lock.hcl",
        "*.tfstate",
        "*.tfstate.*",
        "crash.log",
        "override.tf",
        "override.tf.json",
        "*_override.tf",
        "*_override.tf.json",
        ".terraformrc",
        "terraform.rc",
    ],
    "swift": [
        "# Swift/Xcode",
        ".build/",
        "DerivedData/",
        "*.xcodeproj/xcuserdata/",
        "*.xcworkspace/xcuserdata/",
        "*.pbxuser",
        "*.perspectivev3",
        "*.moved-aside",
        "*.hmap",
        "*.ipa",
        "*.dSYM.zip",
        "*.dSYM",
        "Pods/",
    ],
    "kotlin": [
        "# Kotlin",
        "*.class",
        "*.jar",
        "build/",
        ".gradle/",
        "out/",
        ".kotlin/",
    ],
    "latex": [
        "# LaTeX",
        "*.aux",
        "*.lof",
        "*.log",
        "*.lot",
        "*.fls",
        "*.out",
        "*.toc",
        "*.fmt",
        "*.fot",
        "*.cb",
        "*.cb2",
        "*.bbl",
        "*.bcf",
        "*.blg",
        "*.pdf",
        "*.synctex.gz",
        "*.run.xml",
    ],
    "unity": [
        "# Unity",
        "[Ll]ibrary/",
        "[Tt]emp/",
        "[Oo]bj/",
        "[Bb]uild/",
        "[Bb]uilds/",
        "[Ll]ogs/",
        "UserSettings/",
        "*.csproj",
        "*.unityproj",
        "*.sln",
        "*.suo",
        "*.user",
        "*.pidb",
        "*.booproj",
        "crashlytics-build.properties",
    ],
    "angular": [
        "# Angular",
        "node_modules/",
        "dist/",
        ".angular/",
        "e2e/",
        "*.js.map",
    ],
    "vue": [
        "# Vue",
        "node_modules/",
        "dist/",
        ".nuxt/",
        "*.local",
    ],
    "nextjs": [
        "# Next.js",
        ".next/",
        "out/",
        "node_modules/",
    ],
}

# Aliases
TEMPLATES["cpp"] = TEMPLATES["c"]
TEMPLATES["cxx"] = TEMPLATES["c"]
TEMPLATES["objective-c"] = TEMPLATES["swift"]
TEMPLATES["objc"] = TEMPLATES["swift"]
TEMPLATES["js"] = TEMPLATES["node"]
TEMPLATES["nodejs"] = TEMPLATES["node"]
TEMPLATES["ts"] = TEMPLATES["typescript"]
TEMPLATES["rb"] = TEMPLATES["ruby"]
TEMPLATES["rails"] = TEMPLATES["ruby"]
TEMPLATES["next"] = TEMPLATES["nextjs"]
TEMPLATES["nuxt"] = TEMPLATES["vue"]
TEMPLATES["tex"] = TEMPLATES["latex"]


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

def _normalize_keyword(word: str) -> str:
    """Normalize a keyword for template matching."""
    return word.lower().strip().strip(".,;:!?")


def _extract_keywords(text: str) -> list[str]:
    """Extract technology keywords from input text."""
    # Split on whitespace and common separators
    raw = re.split(r'[\s,;/+&]+', text.strip())
    keywords = []
    for w in raw:
        norm = _normalize_keyword(w)
        if norm:
            keywords.append(norm)
    return keywords


def _match_templates(keywords: list[str]) -> list[str]:
    """Match keywords against templates, return list of matched template names."""
    matched = []
    for kw in keywords:
        if kw in TEMPLATES:
            if kw not in matched:
                matched.append(kw)
    return matched


def _combine_templates(template_names: list[str], with_comments: bool = True) -> list[str]:
    """Combine multiple templates into a single list of rules, deduplicating."""
    seen_rules = set()
    lines = []

    for name in template_names:
        tmpl = TEMPLATES.get(name, [])
        if tmpl is None:
            continue

        section_lines = []
        for line in tmpl:
            # Comments are always included if flag is set
            if line.startswith("#"):
                if with_comments:
                    section_lines.append(line)
                continue
            if line == "":
                section_lines.append(line)
                continue
            # Deduplicate actual rules
            if line not in seen_rules:
                seen_rules.add(line)
                section_lines.append(line)

        # Only add section if it has actual rules (not just comments)
        has_rules = any(l and not l.startswith("#") for l in section_lines)
        if has_rules:
            if lines and lines[-1] != "":
                lines.append("")
            lines.extend(section_lines)

    return lines


# ---------------------------------------------------------------------------
# Model fallback for unrecognized input
# ---------------------------------------------------------------------------

MODEL_TEMPLATE = (
    "Generate .gitignore rules for the following technologies/requirements.\n"
    "Output ONLY the gitignore rules, one per line. No explanations.\n"
    "Include comments (lines starting with #) to organize sections.\n\n"
    "{description}\n\n"
    ".gitignore:"
)


def _generate_from_model(text: str) -> list[str]:
    """Use the model to generate gitignore rules for unrecognized input."""
    prompt = MODEL_TEMPLATE.format(description=text)
    raw = complete(prompt, max_tokens=500, temperature=0.0, stop_at=None)
    if not raw:
        return []

    lines = []
    for line in raw.split("\n"):
        line = line.rstrip()
        # Skip empty lines at start
        if not lines and not line:
            continue
        # Skip lines that look like explanations
        if line and not line.startswith("#") and not line.startswith("!") and " " in line and not "/" in line and not "*" in line and not "." in line:
            continue
        lines.append(line)

    # Strip trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()

    return lines


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_gitignore(text: str, append: bool = False, comments: bool = True) -> str:
    """Generate .gitignore content from input text.

    Args:
        text: Natural language or technology keywords.
        append: If True, output only new rules (no common header).
        comments: If True, add section comments.

    Returns:
        Generated .gitignore content as a string.
    """
    keywords = _extract_keywords(text)
    matched = _match_templates(keywords)
    unmatched = [kw for kw in keywords if kw not in TEMPLATES]

    # Filter out noise words that aren't technology keywords
    noise = {
        "add", "rules", "for", "and", "the", "a", "an", "with", "using",
        "in", "to", "of", "my", "project", "app", "application", "website",
        "web", "site", "ignore", "gitignore", "please", "generate", "create",
        "compiled", "files", "file", "code", "source", "stuff", "things",
        "also", "include", "want", "need", "like", "some", "new",
    }
    meaningful_unmatched = [kw for kw in unmatched if kw not in noise]

    lines = []

    # Add common rules unless in append mode
    if not append:
        if comments:
            lines.extend(COMMON)
        else:
            lines.extend(l for l in COMMON if not l.startswith("#"))

    # Add matched templates
    if matched:
        template_lines = _combine_templates(matched, with_comments=comments)
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend(template_lines)

    # If there are meaningful unmatched keywords, use the model
    if meaningful_unmatched and not matched:
        # Nothing matched from templates at all - use model for the whole thing
        model_lines = _generate_from_model(text)
        if model_lines:
            if append:
                lines = model_lines
            else:
                if lines and lines[-1] != "":
                    lines.append("")
                lines.extend(model_lines)
    elif meaningful_unmatched:
        # Some templates matched, but there are extra keywords - use model for those
        extra_text = " ".join(meaningful_unmatched)
        model_lines = _generate_from_model(extra_text)
        if model_lines:
            if lines and lines[-1] != "":
                lines.append("")
            lines.extend(model_lines)

    # If nothing matched at all and no model output, still return common rules
    if not lines and not append:
        if comments:
            lines = list(COMMON)
        else:
            lines = [l for l in COMMON if not l.startswith("#")]

    # Deduplicate final output (preserving order and comments)
    final = []
    seen = set()
    for line in lines:
        if line.startswith("#") or line == "":
            final.append(line)
            continue
        if line not in seen:
            seen.add(line)
            final.append(line)

    # Clean up multiple consecutive blank lines
    cleaned = []
    prev_blank = False
    for line in final:
        if line == "":
            if not prev_blank:
                cleaned.append(line)
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False

    # Strip leading/trailing blank lines
    while cleaned and cleaned[0] == "":
        cleaned.pop(0)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return "\n".join(cleaned) + "\n"


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def get_input(args) -> str:
    """Get input from file, stdin, or error."""
    if args.input_file:
        try:
            with open(args.input_file) as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: file not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)
        except PermissionError:
            print(f"Error: permission denied: {args.input_file}", file=sys.stderr)
            sys.exit(1)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("Error: no input (pipe text or use -i FILE)", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="bt-gitignore",
        description="Generate .gitignore rules from natural language or technology keywords",
        epilog='Example: echo "python flask docker" | bt-gitignore',
    )
    parser.add_argument(
        "--append", "-a", action="store_true",
        help="Output only new rules (for appending to existing .gitignore)",
    )
    parser.add_argument(
        "--comment", "-c", action="store_true", default=True,
        help="Add section comments (default: on)",
    )
    parser.add_argument(
        "--no-comment", action="store_true",
        help="Suppress section comments",
    )
    parser.add_argument(
        "-i", dest="input_file", metavar="FILE",
        help="Read input from file (default: stdin)",
    )
    args = parser.parse_args()

    if args.no_comment:
        args.comment = False

    text = get_input(args).strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    output = generate_gitignore(text, append=args.append, comments=args.comment)
    print(output, end="")


if __name__ == "__main__":
    main()
