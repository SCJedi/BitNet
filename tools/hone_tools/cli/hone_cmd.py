#!/usr/bin/env python3
"""hone — unified entry point for all hone CLI tools."""

import sys
import importlib

VERSION = "0.1.0"

# name -> (module_path, description)
TOOL_REGISTRY = {
    "classify":   ("hone_tools.cli.classify",    "Categorize text (sentiment, urgency, custom labels)"),
    "extract":    ("hone_tools.cli.extract",      "Pull emails, names, dates, URLs, phones from text"),
    "summarize":  ("hone_tools.cli.summarize",    "Condense text to N sentences"),
    "jsonify":    ("hone_tools.cli.jsonify",       "Convert unstructured text to structured JSON"),
    "rewrite":    ("hone_tools.cli.rewrite",      "Transform text style (formal, simple, bullets, commit)"),
    "tldr":       ("hone_tools.cli.tldr",         "Summarize diffs, logs, errors for developers"),
    "namegen":    ("hone_tools.cli.namegen",      "Generate branch, function, class, variable names"),
    "commit":     ("hone_tools.cli.commit",       "Generate conventional commit messages from diffs"),
    "regex":      ("hone_tools.cli.regex",        "Natural language to regex patterns"),
    "cron":       ("hone_tools.cli.cron",         "Natural language to cron expressions"),
    "sql":        ("hone_tools.cli.sql",          "Natural language to SQL queries"),
    "explain":    ("hone_tools.cli.explain",      "Explain errors, code, config, CLI commands"),
    "assert":     ("hone_tools.cli.assert_cmd",   "Natural language test assertions (pass/fail)"),
    "changelog":  ("hone_tools.cli.changelog",    "Git log to release notes"),
    "env":        ("hone_tools.cli.env",          "Scan code for env vars, generate .env templates"),
    "mock":       ("hone_tools.cli.mock",         "Generate mock data from type descriptions"),
    "gitignore":  ("hone_tools.cli.gitignore",    "Generate .gitignore from technology keywords"),
}

CATEGORIES = {
    "Text Processing": ["classify", "extract", "summarize", "jsonify", "rewrite"],
    "Developer Workflow": ["tldr", "namegen", "commit", "changelog", "env"],
    "Code Generation": ["regex", "cron", "sql", "assert", "mock", "gitignore"],
    "Universal": ["explain"],
}


def show_tool_list():
    print(f"hone v{VERSION} -- {len(TOOL_REGISTRY)} CLI tools powered by local AI")
    print()
    for category, tools in CATEGORIES.items():
        print(f"{category}:")
        for name in tools:
            _, desc = TOOL_REGISTRY[name]
            print(f"  hone {name:<12s} {desc}")
        print()
    print("Run 'hone <tool> --help' for detailed usage.")


def main():
    args = sys.argv[1:]

    # No args, --help, or 'list' -> show tool list
    if not args or args[0] in ("--help", "-h", "list"):
        show_tool_list()
        return

    # Version
    if args[0] in ("--version", "version"):
        print(f"hone v{VERSION}")
        return

    tool_name = args[0]

    if tool_name not in TOOL_REGISTRY:
        print(f"hone: unknown tool '{tool_name}'", file=sys.stderr)
        print(f"Run 'hone list' to see available tools.", file=sys.stderr)
        sys.exit(1)

    module_path, _ = TOOL_REGISTRY[tool_name]

    # Rewrite sys.argv so the sub-tool sees itself as the command
    sys.argv = [f"hone {tool_name}"] + args[1:]

    mod = importlib.import_module(module_path)
    mod.main()


if __name__ == "__main__":
    main()
