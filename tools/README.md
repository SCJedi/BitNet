# hone-cli

17 CLI tools powered by a local LLM. Pipe text in, get structured results out. No API keys, no cloud — everything runs on your GPU.

**Default model:** Qwen3.5-4B-Q4_K_M (2.54 GB, ~70 tok/s on RTX 3060)
**Fallback model:** BitNet-b1.58-2B-4T (CPU inference via BitNet llama-cli)

## Installation

```bash
# From PyPI (coming soon)
pip install hone-cli

# From source (development)
git clone https://github.com/SCJedi/hone-cli.git
cd hone-cli
pip install -e .
```

## Quick Setup

hone-cli needs two things: a **llama.cpp binary** and a **GGUF model file**. The fastest way to configure them:

```bash
# Point to your llama-cli and model via environment variables
export HONE_CLI=/path/to/llama-cli
export HONE_MODEL=/path/to/model.gguf

# Verify it works
echo "hello world" | hone-summarize
```

Or create a persistent config file:

```bash
# Linux/macOS: ~/.config/hone/config.json
# Windows: %APPDATA%/hone/config.json
mkdir -p ~/.config/hone
cat > ~/.config/hone/config.json << 'EOF'
{
    "cli": "/path/to/llama-cli",
    "model": "/path/to/model.gguf",
    "use_gpu": true,
    "is_chat_model": true
}
EOF
```

If installed inside the [BitNet repo](https://github.com/microsoft/BitNet), the tools auto-detect the model and binary automatically.

## Tools

### Text Processing

**hone-classify** — Categorize text (sentiment, urgency, custom labels)
```bash
echo "the app crashes on submit" | hone-classify --labels bug,feature,question
echo "I love this product!" | hone-classify --preset sentiment
```

**hone-extract** — Pull emails, names, dates, URLs, phones with regex post-validation
```bash
echo "Contact john@example.com or 212-555-0100" | hone-extract --type emails
cat contacts.txt | hone-extract --type phones --json
```

**hone-summarize** — Condense text to N sentences
```bash
cat article.txt | hone-summarize
cat report.txt | hone-summarize --sentences 3
```

**hone-jsonify** — Unstructured text to structured JSON
```bash
echo "John Doe, age 30, lives in NYC" | hone-jsonify --fields name,age,city
echo "Order #1234, $49.99, shipped" | hone-jsonify --fields order_id,price,status
```

**hone-rewrite** — Style transforms (formal, simple, bullets, commit, punctuate)
```bash
echo "hey fix the css bug pls" | hone-rewrite --style formal
echo "fixed login page on mobile" | hone-rewrite --style commit
```

### Developer Workflow

**hone-tldr** — Dev-focused summaries (auto-detects diff/log/error/generic)
```bash
git diff | hone-tldr
cat error.log | hone-tldr --mode error
```

**hone-namegen** — Generate branch/function/class/variable/file names
```bash
echo "fix broken login page css" | hone-namegen --style branch
echo "validate user email address" | hone-namegen --style function
```

**hone-commit** — Git diff to conventional commit message
```bash
git diff --cached | hone-commit
git diff HEAD~1 | hone-commit --scope auth
```

**hone-changelog** — Git log to release notes, grouped by commit type
```bash
git log --oneline v1.0..v1.1 | hone-changelog
git log --oneline -20 | hone-changelog --format markdown
```

**hone-gitignore** — Template-first .gitignore generator
```bash
hone-gitignore --lang python
hone-gitignore --lang node,rust --append
```

**hone-env** — Scan code for env vars, generate .env templates
```bash
hone-env --scan src/
hone-env --scan . --output .env.example
```

### Code Generation

**hone-regex** — Natural language to regex with --test validation
```bash
echo "match email addresses" | hone-regex
echo "US phone numbers" | hone-regex --test "call 212-555-0100"
```

**hone-cron** — Natural language to cron with --explain and --validate
```bash
echo "every weekday at 9am" | hone-cron
echo "0 */2 * * *" | hone-cron --explain
```

**hone-sql** — Natural language to SQL with schema awareness
```bash
echo "users who signed up this month" | hone-sql --table users
echo "top 10 orders by total" | hone-sql --schema schema.sql
```

**hone-mock** — Generate mock data from type descriptions (JSON/CSV/SQL)
```bash
echo "user with name, email, age" | hone-mock --count 5
echo "product with sku, price, category" | hone-mock --format csv
```

**hone-assert** — Natural language test assertions, binary pass/fail for CI
```bash
echo '{"status":200}' | hone-assert "status code is 200"
curl -s api/health | hone-assert "response contains ok"
```

### Universal

**hone-explain** — Universal explainer for errors, code, config, CLI
```bash
echo "ECONNREFUSED 127.0.0.1:5432" | hone-explain
echo "git rebase -i HEAD~3" | hone-explain
```

## Configuration

### Resolution Order

Configuration is resolved from three sources (highest priority first):

1. **Environment variables** -- `HONE_CLI`, `HONE_MODEL` (override everything)
2. **User config file** -- `~/.config/hone/config.json` (persistent settings)
3. **BitNet repo auto-detection** -- for in-repo development (walks up to find `CMakeLists.txt`)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HONE_MODEL` | Auto-detected | Path to GGUF model file |
| `HONE_CLI` | Auto-detected | Path to llama-cli binary |
| `HONE_THREADS` | `4` | CPU threads for inference |
| `HONE_CTX_SIZE` | `2048` | Context window size |

### Config File

Location: `~/.config/hone/config.json` (Linux/macOS) or `%APPDATA%/hone/config.json` (Windows)

```json
{
    "cli": "/path/to/llama-cli",
    "model": "/path/to/model.gguf",
    "threads": 4,
    "ctx_size": 2048,
    "use_gpu": true,
    "is_chat_model": true
}
```

### In-Repo Auto-Detection

When installed inside the BitNet repo, the tools search for models automatically:
1. `models/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf` + `tools/llama-cpp-latest/bin/llama-cli.exe` (GPU)
2. `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf` + `build/bin/llama-cli.exe` (CPU)

## Piping Examples

```bash
# Classify and extract in a pipeline
cat email.txt | hone-classify --preset topic
cat email.txt | hone-extract --type emails --json | jq '.[0]'

# Summarize a git diff
git diff HEAD~5 | hone-tldr

# Convert unstructured data to JSON for processing
cat business_card.txt | hone-jsonify --fields name,title,company,phone,email

# Generate a branch name from a ticket title
echo "Users can't log in after password reset" | hone-namegen --style branch

# Conventional commit from staged changes
git diff --cached | hone-commit

# Release notes from git history
git log --oneline v1.0..HEAD | hone-changelog

# Generate a .env template from source
hone-env --scan src/ --output .env.example

# Mock data for testing
echo "user with id, name, email, role" | hone-mock --count 10 --format csv

# Validate cron expressions
echo "0 9 * * 1-5" | hone-cron --explain

# CI assertion
curl -s localhost:8080/health | hone-assert "response is healthy"
```

All tools read from stdin by default, or from a file with `-i FILE`.

## Test Results

852 tests across all 17 tools. ~99.8% pass rate.

| Tool | Tests | Pass Rate | Notes |
|------|-------|-----------|-------|
| hone-classify | 48 | 96% | 2 debatable edge cases on ambiguous boundaries |
| hone-extract | 25 | 100% | Regex post-validation eliminates hallucinations |
| hone-summarize | 22 | 100% | Echo detection + min-length guard |
| hone-jsonify | 21 | 100% | Prompt injection defense |
| hone-rewrite | 20 | 100% | Style-specific stop tokens |
| hone-tldr | 36 | 100% | 3-tier error detection |
| hone-namegen | 24 | 100% | Space-separated prompt strategy |
| hone-commit | 60 | 100% | Conventional commit format enforcement |
| hone-regex | 23 | 100% | Pattern validation with test input |
| hone-cron | 110 | 100% | Explain + validate modes |
| hone-sql | 80 | 100% | Schema-aware generation |
| hone-explain | 66 | 100% | Multi-domain detection |
| hone-assert | 60 | 100% | Binary exit codes for CI |
| hone-changelog | 53 | 100% | Commit type grouping |
| hone-env | 21 | 100% | Cross-file env var scanning |
| hone-mock | 64 | 100% | JSON/CSV/SQL output formats |
| hone-gitignore | 119 | 100% | Template-first with fallback |

Full details: [TEST_REPORT_FINAL.md](TEST_REPORT_FINAL.md)

## Architecture

```
User input (stdin or -i FILE)
    |
    v
CLI tool (classify.py, extract.py, etc.)
    |  - Builds prompt from input + flags
    |  - Sets max_tokens, stop_at, temperature
    |
    v
engine.py -> complete()
    |  - Truncates input if > 6000 chars
    |  - Builds llama-cli command
    |  - Runs subprocess with timeout
    |  - Strips banner, ANSI codes, think tags
    |  - clean_output(): strips fences, prefixes,
    |    markdown, deduplicates, truncates
    |
    v
config.py
    |  - Auto-detects model + CLI binary
    |  - Reads env var overrides
    |  - Validates paths exist
    |
    v
llama-cli (subprocess)
    |  - GPU mode: --single-turn -ngl 99 -rea off
    |  - CPU mode: -ngl 0 -b 1
    |
    v
GGUF model file
```

## Requirements

- Python 3.10+
- No Python dependencies (uses only stdlib)
- llama.cpp binary (included in `tools/llama-cpp-latest/` for GPU, or `build/bin/` for BitNet CPU)
- A GGUF model file (Qwen3.5-4B-Q4_K_M recommended)
- GPU: NVIDIA with CUDA support (optional, falls back to CPU)
