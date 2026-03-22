"""bt-namegen: Generate code names (branches, functions, etc.) using BitNet."""

import argparse
import re
import sys

from ..engine import complete

# Prompts ask the model to output SEPARATED WORDS, not formatted names.
# We apply formatting ourselves from the word list.
STYLE_PROMPTS = {
    "branch": (
        "Generate a short git branch name for this task. "
        "Output ONLY the words separated by spaces, no formatting.\n"
        "Example input: fix login page css\n"
        "Example output: fix login page css\n\n"
        "Input: {input}\nOutput:"
    ),
    "function": (
        "Generate a short function name for this description. "
        "Output ONLY the words separated by spaces, no formatting.\n"
        "Example input: check if user email is valid\n"
        "Example output: validate user email\n\n"
        "Input: {input}\nOutput:"
    ),
    "file": (
        "Generate a short filename for this description. "
        "Output ONLY the words separated by spaces, no extension, no formatting.\n"
        "Example input: database configuration settings\n"
        "Example output: db config\n\n"
        "Input: {input}\nOutput:"
    ),
    "class": (
        "Generate a short class name for this description. "
        "Output ONLY the words separated by spaces, no formatting.\n"
        "Example input: handles HTTP requests for user auth\n"
        "Example output: http request handler\n\n"
        "Input: {input}\nOutput:"
    ),
    "variable": (
        "Generate a short variable name for this description. "
        "Output ONLY the words separated by spaces, no formatting.\n"
        "Example input: number of failed login attempts\n"
        "Example output: failed login count\n\n"
        "Input: {input}\nOutput:"
    ),
}

# Branch type prefixes to detect and preserve
BRANCH_PREFIXES = (
    "fix", "feat", "feature", "bug", "bugfix", "hotfix", "chore",
    "refactor", "docs", "test", "ci", "build", "perf", "style",
)


def _split_words(s: str) -> list[str]:
    """Split a string into words, handling multiple formats.

    Handles: spaces, hyphens, underscores, camelCase, PascalCase, and
    fully concatenated words (tries camelCase boundary detection).
    """
    # First: replace common separators with spaces
    s = re.sub(r"[_\-/]+", " ", s)

    # Split on camelCase/PascalCase boundaries:
    # - before an uppercase letter that follows a lowercase letter (camelCase)
    # - before an uppercase letter followed by a lowercase letter after uppercase run (HTTPRequest -> HTTP Request)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)

    # Split on spaces and filter empty
    words = [w.strip() for w in s.split() if w.strip()]

    # Remove non-alphanumeric chars from each word
    words = [re.sub(r"[^a-zA-Z0-9]", "", w) for w in words]
    return [w for w in words if w]


def _extract_branch_prefix(words: list[str]) -> tuple[str | None, list[str]]:
    """Extract a branch type prefix (fix, feat, etc.) from word list."""
    if not words:
        return None, words
    first = words[0].lower()
    # Handle two-word prefixes like "bug fix" -> "bugfix"
    if len(words) >= 2:
        compound = first + words[1].lower()
        if compound in BRANCH_PREFIXES:
            return compound, words[2:]
    if first in BRANCH_PREFIXES:
        return first, words[1:]
    return None, words


def to_kebab(words: list[str]) -> str:
    """Convert word list to kebab-case, with branch prefix detection."""
    if not words:
        return ""
    prefix, rest = _extract_branch_prefix(words)
    kebab = "-".join(w.lower() for w in (rest if rest else words))
    if prefix and rest:
        return f"{prefix}/{kebab}"
    return kebab


def to_camel(words: list[str]) -> str:
    """Convert word list to camelCase."""
    if not words:
        return ""
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


def to_pascal(words: list[str]) -> str:
    """Convert word list to PascalCase."""
    return "".join(w.capitalize() for w in words) if words else ""


def to_snake(words: list[str]) -> str:
    """Convert word list to snake_case."""
    return "_".join(w.lower() for w in words) if words else ""


FORMATTERS = {
    "branch": to_kebab,
    "function": to_camel,
    "file": to_snake,
    "class": to_pascal,
    "variable": to_snake,
}


def _words_from_model_output(raw: str) -> list[str]:
    """Extract words from model output, handling various formats."""
    # Strip quotes, backticks, and other wrapping
    raw = raw.strip().strip("`'\"")

    # Remove file extensions if present
    raw = re.sub(r"\.\w{1,4}$", "", raw)

    return _split_words(raw)


def _words_from_input(text: str) -> list[str]:
    """Extract meaningful words from the user's input description."""
    # Remove common filler/stop words for more concise names
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "of", "in", "to", "for", "with", "on", "at", "from", "by",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "that", "this", "these", "those",
        "it", "its", "if", "or", "and", "but", "not", "no", "nor",
        "so", "very", "just", "about", "up", "there", "here", "when",
        "where", "how", "all", "each", "every", "both", "few", "more",
    }
    words = _split_words(text)
    filtered = [w for w in words if w.lower() not in stop_words]
    return filtered if filtered else words


def get_input(args):
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


def main():
    parser = argparse.ArgumentParser(
        prog="bt-namegen",
        description="Generate code names using a local BitNet model",
        epilog="Example: echo 'fix the broken login page css' | bt-namegen --style branch",
    )
    parser.add_argument(
        "--style", "-s", required=True, choices=list(STYLE_PROMPTS.keys()),
        help="Naming style",
    )
    parser.add_argument("--input", "-i", help="Input file (default: stdin)")
    args = parser.parse_args()

    text = get_input(args).strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    prompt = STYLE_PROMPTS[args.style].format(input=text)
    result = complete(prompt, max_tokens=30, stop_at="\n")

    # Extract words from model output
    model_words = _words_from_model_output(result)

    # Fallback: if model output is empty or a single concatenated blob,
    # use the input description words instead
    if not model_words:
        model_words = _words_from_input(text)
    elif len(model_words) == 1 and len(model_words[0]) > 15:
        # Single long word = probably concatenated; fall back to input
        model_words = _words_from_input(text)

    # Apply formatting
    formatter = FORMATTERS[args.style]
    formatted = formatter(model_words)

    if formatted:
        print(formatted)
    else:
        print("unnamed")


if __name__ == "__main__":
    main()
