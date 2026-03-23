"""bt-env: Scan code and generate .env templates with discovered environment variables.

Hybrid approach: regex-first for extraction, model for descriptions (--comments).
Reads from stdin or -i FILE. Outputs in dotenv, json, yaml, or shell format.
"""

import argparse
import json
import re
import sys

from ..engine import complete


# ---------------------------------------------------------------------------
# Regex patterns for env var extraction
# ---------------------------------------------------------------------------

# Python: os.environ["KEY"], os.getenv("KEY"), os.environ.get("KEY", ...)
_PY_ENVIRON_BRACKET = re.compile(
    r"""os\.environ\s*\[\s*(['"])([A-Z][A-Z0-9_]*)\1\s*\]""",
)
_PY_GETENV = re.compile(
    r"""os\.getenv\s*\(\s*(['"])([A-Z][A-Z0-9_]*)\1""",
)
_PY_ENVIRON_GET = re.compile(
    r"""os\.environ\.get\s*\(\s*(['"])([A-Z][A-Z0-9_]*)\1""",
)

# JavaScript / TypeScript: process.env.KEY, process.env["KEY"]
_JS_PROCESS_DOT = re.compile(
    r"""process\.env\.([A-Z][A-Z0-9_]*)""",
)
_JS_PROCESS_BRACKET = re.compile(
    r"""process\.env\s*\[\s*(['"])([A-Z][A-Z0-9_]*)\1\s*\]""",
)

# Docker / docker-compose: ${KEY}, ENV KEY=val, - KEY=val (under environment:)
_DOCKER_INTERP = re.compile(
    r"""\$\{([A-Z][A-Z0-9_]*)\}""",
)
_DOCKER_ENV_DIRECTIVE = re.compile(
    r"""^ENV\s+([A-Z][A-Z0-9_]*)""", re.MULTILINE,
)

# Shell: $KEY (not inside ${} which is caught above)
_SHELL_VAR = re.compile(
    r"""\$([A-Z][A-Z0-9_]{2,})""",
)

# Generic: ALL_CAPS near config-related keywords
_GENERIC_CONFIG = re.compile(
    r"""(?:env|config|secret|key|token|url|host|port|password|database|redis|aws|smtp|api)"""
    r"""[^A-Za-z0-9]*([A-Z][A-Z0-9_]{2,})""",
    re.IGNORECASE,
)
_GENERIC_CONFIG_BEFORE = re.compile(
    r"""([A-Z][A-Z0-9_]{2,})[^A-Za-z0-9]*"""
    r"""(?:env|config|secret|key|token|url|host|port|password|database|redis|aws|smtp|api)""",
    re.IGNORECASE,
)

# .env file lines: KEY=value
_DOTENV_LINE = re.compile(
    r"""^([A-Z][A-Z0-9_]*)\s*=""", re.MULTILINE,
)

# docker-compose environment list: - KEY=value or - KEY
_COMPOSE_ENV_LIST = re.compile(
    r"""^\s*-\s*([A-Z][A-Z0-9_]*)(?:\s*=|\s*$)""", re.MULTILINE,
)


# Vars that are almost certainly not user-defined env vars
_NOISE = frozenset({
    "HOME", "PATH", "PWD", "USER", "SHELL", "TERM", "LANG", "EDITOR",
    "HOSTNAME", "LOGNAME", "TMPDIR", "TEMP", "TMP",
    "TRUE", "FALSE", "NULL", "NONE", "EOF", "AND", "NOT", "THE",
    "GET", "SET", "PUT", "POST", "DELETE", "PATCH",
    "ENV", "CONFIG", "KEY", "VALUE", "NAME", "TYPE", "FORMAT",
    "FILE", "DIR", "DIRECTORY",
})


def _is_plausible_env_var(name: str) -> bool:
    """Filter out names that are too generic or too short."""
    if len(name) < 3:
        return False
    # Must be ALL_CAPS_UNDERSCORE (reject mixed case like "environ", "getenv")
    if not re.match(r'^[A-Z][A-Z0-9_]*$', name):
        return False
    if name in _NOISE:
        return False
    # Must contain at least one underscore OR be a well-known pattern
    well_known_prefixes = ("AWS_", "DB_", "API_", "REDIS_", "SMTP_",
                           "DATABASE_", "SECRET_", "JWT_", "AUTH_",
                           "OPENAI_", "STRIPE_", "GITHUB_", "DOCKER_",
                           "NODE_", "NEXT_", "REACT_", "VITE_")
    if any(name.startswith(p) for p in well_known_prefixes):
        return True
    if "_" in name:
        return True
    # Single word ALL_CAPS with 4+ chars near config context: let through
    if len(name) >= 6:
        return True
    return False


def extract_env_vars(text: str) -> list[str]:
    """Extract environment variable names from source code text."""
    found: set[str] = set()

    # Python patterns (group 2 = var name after quote char)
    for m in _PY_ENVIRON_BRACKET.finditer(text):
        found.add(m.group(2))
    for m in _PY_GETENV.finditer(text):
        found.add(m.group(2))
    for m in _PY_ENVIRON_GET.finditer(text):
        found.add(m.group(2))

    # JavaScript patterns
    for m in _JS_PROCESS_DOT.finditer(text):
        found.add(m.group(1))
    for m in _JS_PROCESS_BRACKET.finditer(text):
        found.add(m.group(2))

    # Docker / compose
    for m in _DOCKER_INTERP.finditer(text):
        found.add(m.group(1))
    for m in _DOCKER_ENV_DIRECTIVE.finditer(text):
        found.add(m.group(1))

    # Shell
    for m in _SHELL_VAR.finditer(text):
        found.add(m.group(1))

    # .env style
    for m in _DOTENV_LINE.finditer(text):
        found.add(m.group(1))

    # docker-compose environment list
    for m in _COMPOSE_ENV_LIST.finditer(text):
        found.add(m.group(1))

    # Generic context-based: only as fallback when explicit patterns found nothing
    if not found:
        for m in _GENERIC_CONFIG.finditer(text):
            found.add(m.group(1))
        for m in _GENERIC_CONFIG_BEFORE.finditer(text):
            found.add(m.group(1))

    # Filter and sort
    filtered = sorted(v for v in found if _is_plausible_env_var(v))
    return filtered


# ---------------------------------------------------------------------------
# Default placeholder values
# ---------------------------------------------------------------------------

_PLACEHOLDER_HINTS: dict[str, str] = {
    "PORT": "3000",
    "HOST": "localhost",
    "DEBUG": "false",
    "LOG_LEVEL": "info",
    "NODE_ENV": "development",
}


def _placeholder(name: str) -> str:
    """Generate a sensible placeholder value for a variable name."""
    if name in _PLACEHOLDER_HINTS:
        return _PLACEHOLDER_HINTS[name]
    lower = name.lower()
    if "port" in lower:
        return "3000"
    if "host" in lower:
        return "localhost"
    if "url" in lower or "uri" in lower:
        return "https://example.com"
    if "secret" in lower or "key" in lower or "token" in lower:
        return "changeme"
    if "password" in lower or "passwd" in lower:
        return "changeme"
    if "database" in lower or "db_name" in lower:
        return "mydb"
    if "user" in lower:
        return "admin"
    if "email" in lower:
        return "admin@example.com"
    if "debug" in lower:
        return "false"
    if "timeout" in lower:
        return "30"
    if "redis" in lower and "url" not in lower:
        return "redis://localhost:6379"
    return ""


# ---------------------------------------------------------------------------
# Model-based comments
# ---------------------------------------------------------------------------

COMMENT_PROMPT = (
    "For each environment variable below, write a one-line description of what it likely does, "
    "based on the variable name. Output format: one line per variable, exactly:\n"
    "VARNAME: description\n\n"
    "Variables:\n{var_list}\n\n"
    "Descriptions:"
)


def _get_comments(var_names: list[str]) -> dict[str, str]:
    """Use the model to generate a one-line description per variable."""
    if not var_names:
        return {}
    var_list = "\n".join(var_names)
    prompt = COMMENT_PROMPT.format(var_list=var_list)
    raw = complete(prompt, max_tokens=len(var_names) * 30, stop_at=None)

    comments: dict[str, str] = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, desc = line.partition(":")
            key = key.strip()
            desc = desc.strip()
            if key in var_names and desc:
                comments[key] = desc
    return comments


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def format_dotenv(
    var_names: list[str],
    comments: dict[str, str] | None = None,
) -> str:
    """Format as .env file."""
    lines: list[str] = []
    for name in var_names:
        if comments and name in comments:
            lines.append(f"# {comments[name]}")
        lines.append(f"{name}={_placeholder(name)}")
    return "\n".join(lines)


def format_json(
    var_names: list[str],
    comments: dict[str, str] | None = None,
) -> str:
    """Format as JSON object."""
    obj: dict[str, str | dict[str, str]] = {}
    for name in var_names:
        if comments and name in comments:
            obj[name] = {"value": _placeholder(name), "description": comments[name]}
        else:
            obj[name] = _placeholder(name)
    return json.dumps(obj, indent=2)


def format_yaml(
    var_names: list[str],
    comments: dict[str, str] | None = None,
) -> str:
    """Format as YAML."""
    lines: list[str] = []
    for name in var_names:
        if comments and name in comments:
            lines.append(f"# {comments[name]}")
        val = _placeholder(name)
        # Quote values with special yaml chars
        if val and any(c in val for c in ":#{}[]&*!|>'\"%@`"):
            val = f'"{val}"'
        lines.append(f"{name}: {val}")
    return "\n".join(lines)


def format_shell(
    var_names: list[str],
    comments: dict[str, str] | None = None,
) -> str:
    """Format as shell export statements."""
    lines: list[str] = []
    for name in var_names:
        if comments and name in comments:
            lines.append(f"# {comments[name]}")
        val = _placeholder(name)
        lines.append(f'export {name}="{val}"')
    return "\n".join(lines)


FORMATTERS = {
    "dotenv": format_dotenv,
    "json": format_json,
    "yaml": format_yaml,
    "shell": format_shell,
}


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def get_input(args) -> str:
    """Get source code text from file, stdin, or error."""
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
    print("Error: no input (pipe code or use -i FILE)", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="bt-env",
        description="Scan code and generate .env templates with discovered environment variables",
        epilog='Example: cat app.py | bt-env --format dotenv',
    )
    parser.add_argument(
        "--format", "-f", default="dotenv",
        choices=list(FORMATTERS.keys()),
        help="Output format (default: dotenv)",
    )
    parser.add_argument(
        "--comments", "-c", action="store_true",
        help="Add model-generated comments explaining each variable",
    )
    parser.add_argument(
        "-i", dest="input_file", metavar="FILE",
        help="Read from file (default: stdin)",
    )
    args = parser.parse_args()

    text = get_input(args)
    if not text.strip():
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    var_names = extract_env_vars(text)

    if not var_names:
        print("# No environment variables found.", file=sys.stderr)
        sys.exit(0)

    comments = None
    if args.comments:
        comments = _get_comments(var_names)

    formatter = FORMATTERS[args.format]
    output = formatter(var_names, comments)
    print(output)


if __name__ == "__main__":
    main()
