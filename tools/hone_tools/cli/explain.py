"""hone-explain: Universal explainer for technical content.

Pipe anything technical — errors, code, config, regex, CLI commands, logs —
and get a plain-English explanation.
"""

import argparse
import re
import sys

from ..engine import complete

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

MODES = ("error", "code", "config", "regex", "cli", "log", "generic")

# Config file signatures (filename-style markers that appear in content)
_CONFIG_PATTERNS = [
    r"(?:server\s*\{|location\s*/|upstream\s+\w+)",       # nginx
    r"(?:<VirtualHost|<Directory|ServerName)",              # apache
    r"(?:services:\s*\n|volumes:\s*\n|networks:\s*\n)",    # docker-compose
    r"(?:apiVersion:|kind:\s+\w+|metadata:\s*\n)",         # kubernetes
    r"(?:\[Unit\]|\[Service\]|\[Install\])",                # systemd
    r"(?:\[global\]|\[defaults\]|\[frontend\]|\[backend\])",  # haproxy
]

_REGEX_META = r"[\^\$\.\*\+\?\{\}\[\]\(\)\|\\]"


def detect_mode(text: str) -> str:
    """Auto-detect the type of input text."""
    stripped = text.strip()
    lines = stripped.split("\n")

    # --- Error / traceback detection ---
    # Strong signals (case-insensitive)
    if re.search(
        r"(Traceback \(most recent call last\)"
        r"|^\s*File \"[^\"]+\", line \d+"
        r"|^\w+Error:"
        r"|^\w+Exception:"
        r"|Segmentation fault"
        r"|SIGSEGV|SIGABRT|core\s+dump"
        r"|stack\s*trace"
        r"|Uncaught\s+\w+Error"
        r"|TypeError:|ReferenceError:|SyntaxError:"
        r"|undefined is not|Cannot read propert"
        r"|error\s*\[\d+\]:\s*"           # gcc/clang style
        r"|:\d+:\d+:\s*error:"            # gcc "file:line:col: error:"
        r"|fatal\s+error:"
        r"|compilation\s+terminated"
        r"|linker\s+command\s+failed"
        r"|panicked\s+at\b)",
        stripped,
        re.MULTILINE | re.IGNORECASE,
    ):
        return "error"

    # Case-sensitive error markers
    if re.search(r"(\bERROR\b|\bFATAL\b|\bPANIC\b)", stripped):
        return "error"

    # --- Regex detection ---
    # A short line dominated by regex metacharacters
    if len(lines) <= 3:
        # Count regex-specific metacharacters
        meta_chars = len(re.findall(_REGEX_META, stripped))
        alpha_chars = len(re.findall(r"[a-zA-Z0-9]", stripped))
        total = meta_chars + alpha_chars
        if total > 0 and meta_chars / total > 0.3 and meta_chars >= 3:
            # Looks like a regex — check for common regex constructs
            if re.search(r"(\[.*?\]|\\.|\(\?[=!<:]|[+*?]\{|\{\d+)", stripped):
                return "regex"

    # --- CLI command detection ---
    first_line = lines[0].strip()
    # Starts with $ or > prompt
    if re.match(r"^[\$>]\s+", first_line):
        return "cli"
    # Starts with a well-known command
    cli_commands = (
        "curl", "wget", "git", "docker", "kubectl", "npm", "yarn", "pip",
        "apt", "yum", "brew", "cargo", "go", "make", "cmake", "gcc", "g++",
        "python", "node", "java", "javac", "mvn", "gradle", "terraform",
        "ansible", "ssh", "scp", "rsync", "tar", "find", "grep", "awk",
        "sed", "chmod", "chown", "systemctl", "journalctl", "ffmpeg",
        "openssl", "helm", "aws", "gcloud", "az",
    )
    first_word = re.split(r"\s+", first_line)[0] if first_line else ""
    if first_word in cli_commands and len(lines) <= 5:
        return "cli"

    # --- Cron expression ---
    if re.match(r"^[0-9*/,-]+\s+[0-9*/,-]+\s+[0-9*/,-]+\s+[0-9*/,-]+\s+[0-9*/,-]+$", stripped):
        return "config"  # treat cron as config

    # --- Config detection ---
    for pat in _CONFIG_PATTERNS:
        if re.search(pat, stripped, re.MULTILINE | re.IGNORECASE):
            return "config"

    # YAML/TOML/INI structure
    if re.search(r"^\[[\w.-]+\]", stripped, re.MULTILINE) and "=" in stripped:
        return "config"

    # Dockerfile (require at least 2 distinct directives to avoid false positives
    # e.g. SQL "FROM users" should not trigger this)
    dockerfile_directives = re.findall(
        r"^(FROM|RUN|COPY|CMD|ENTRYPOINT|EXPOSE|ENV|ARG|WORKDIR)\s",
        stripped, re.MULTILINE,
    )
    if len(set(dockerfile_directives)) >= 2:
        return "config"

    # --- Log detection ---
    # Timestamped lines (ISO, syslog, bracket)
    timestamp_lines = 0
    for line in lines[:10]:
        if re.match(
            r"(\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}"
            r"|^\[?\d{4}[-/]\d{2}[-/]\d{2}\s"
            r"|^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})",
            line,
        ):
            timestamp_lines += 1
    if timestamp_lines >= 2 or (timestamp_lines >= 1 and len(lines) <= 3):
        return "log"

    # --- Code detection ---
    code_signals = 0
    # Function/class definitions
    if re.search(r"^(def |class |function |const |let |var |import |from |export )", stripped, re.MULTILINE):
        code_signals += 2
    # Braces/semicolons structure
    if re.search(r"[{};]$", stripped, re.MULTILINE):
        code_signals += 1
    # SQL
    if re.search(r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s", stripped, re.MULTILINE | re.IGNORECASE):
        code_signals += 2
    # Indented blocks
    if re.search(r"^    \S", stripped, re.MULTILINE):
        code_signals += 1
    # Common syntax
    if re.search(r"(=>|->|\breturn\b|\bif\b.*:|\bfor\b.*:|\bwhile\b)", stripped):
        code_signals += 1

    if code_signals >= 2:
        return "code"

    # JSON (braces + quotes + colons)
    if (stripped.startswith("{") or stripped.startswith("[")) and '"' in stripped:
        return "code"

    return "generic"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

MODE_PROMPTS = {
    "error": (
        "You are a developer assistant. Explain this error in plain English.\n"
        "What is this error? What caused it? How do you fix it?\n"
        "Be concise and practical.\n\n"
        "{input}\n\n"
        "Explanation:"
    ),
    "error_fix": (
        "You are a developer assistant. This is an error message.\n"
        "Focus on the FIX: give specific, actionable steps to resolve it.\n\n"
        "{input}\n\n"
        "Fix:"
    ),
    "code": (
        "You are a developer assistant. Explain what this code does.\n"
        "Walk through the logic step by step. Be concise.\n\n"
        "{input}\n\n"
        "Explanation:"
    ),
    "config": (
        "You are a developer assistant. Explain what this configuration does.\n"
        "Describe what each section/directive controls. Be concise.\n\n"
        "{input}\n\n"
        "Explanation:"
    ),
    "regex": (
        "You are a developer assistant. Break down this regular expression.\n"
        "Explain what it matches and describe each part.\n\n"
        "{input}\n\n"
        "Breakdown:"
    ),
    "cli": (
        "You are a developer assistant. Explain this command.\n"
        "Describe what it does and explain each flag/argument. Be concise.\n\n"
        "{input}\n\n"
        "Explanation:"
    ),
    "log": (
        "You are a developer assistant. Analyze these log entries.\n"
        "Summarize what happened according to these logs. Be concise.\n\n"
        "{input}\n\n"
        "Analysis:"
    ),
    "generic": (
        "You are a developer assistant. Explain this in plain English.\n"
        "Be concise and practical.\n\n"
        "{input}\n\n"
        "Explanation:"
    ),
}

BRIEF_SUFFIX = "\nRespond with exactly ONE sentence."


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def postprocess(text: str, brief: bool = False) -> str:
    """Clean up model output for display."""
    # Strip markdown heading lines entirely
    text = re.sub(r"^#{1,6}\s+[^\n]*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"```[\w]*\n?", "", text)
    text = re.sub(r"```", "", text)

    # Clean whitespace
    text = text.strip()

    if brief and text:
        # Take first sentence only
        match = re.match(r"^(.+?[.!?])(?:\s|$)", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            # No sentence-ending punctuation — take first line
            text = text.split("\n")[0].strip()

    return text


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def get_input(args) -> str:
    """Read input from file or stdin."""
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
    print("Error: no input (pipe text or use -i FILE)", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="hone-explain",
        description="Universal explainer: pipe any technical content, get a plain-English explanation",
        epilog="Example: cat error.log | hone-explain",
    )
    parser.add_argument(
        "--mode", "-m", choices=list(MODES),
        help="Force detection mode (default: auto-detect)",
    )
    parser.add_argument(
        "--brief", "-b", action="store_true",
        help="One-sentence explanation only",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="For errors: focus on actionable fix steps",
    )
    parser.add_argument(
        "--input", "-i", help="Read from file instead of stdin",
    )
    args = parser.parse_args()

    text = get_input(args).strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    mode = args.mode or detect_mode(text)

    # Choose prompt
    if args.fix and mode == "error":
        prompt_key = "error_fix"
    elif args.fix and not args.mode:
        # --fix without explicit mode: auto-detect may not be error, but
        # user wants fix-oriented output — force error mode
        prompt_key = "error_fix"
    else:
        prompt_key = mode

    prompt_template = MODE_PROMPTS[prompt_key]
    prompt = prompt_template.format(input=text)

    if args.brief:
        prompt += BRIEF_SUFFIX

    # Longer content needs more tokens
    max_tokens = 80 if args.brief else 300
    result = complete(prompt, max_tokens=max_tokens, stop_at=None)
    result = postprocess(result, brief=args.brief)

    if result:
        print(result)
    else:
        print("Error: model returned empty output", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
