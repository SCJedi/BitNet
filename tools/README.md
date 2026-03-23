# bitnet-tools

17 CLI tools powered by a local LLM. Pipe text in, get structured results out. No API keys, no cloud — everything runs on your GPU.

**Default model:** Qwen3.5-4B-Q4_K_M (2.54 GB, ~70 tok/s on RTX 3060)
**Fallback model:** BitNet-b1.58-2B-4T (CPU inference via BitNet llama-cli)

## Quick Start

```bash
# Install (editable, from the tools/ directory)
cd BitNet/tools
pip install -e .

# Verify model and CLI are detected
echo "hello world" | bt-summarize
```

The tools auto-detect the best available model. Qwen3.5-4B with GPU is preferred; BitNet 2B on CPU is the fallback.

## Tools

### Text Processing

**bt-classify** — Categorize text (sentiment, urgency, custom labels)
```bash
echo "the app crashes on submit" | bt-classify --labels bug,feature,question
echo "I love this product!" | bt-classify --preset sentiment
```

**bt-extract** — Pull emails, names, dates, URLs, phones with regex post-validation
```bash
echo "Contact john@example.com or 212-555-0100" | bt-extract --type emails
cat contacts.txt | bt-extract --type phones --json
```

**bt-summarize** — Condense text to N sentences
```bash
cat article.txt | bt-summarize
cat report.txt | bt-summarize --sentences 3
```

**bt-jsonify** — Unstructured text to structured JSON
```bash
echo "John Doe, age 30, lives in NYC" | bt-jsonify --fields name,age,city
echo "Order #1234, $49.99, shipped" | bt-jsonify --fields order_id,price,status
```

**bt-rewrite** — Style transforms (formal, simple, bullets, commit, punctuate)
```bash
echo "hey fix the css bug pls" | bt-rewrite --style formal
echo "fixed login page on mobile" | bt-rewrite --style commit
```

### Developer Workflow

**bt-tldr** — Dev-focused summaries (auto-detects diff/log/error/generic)
```bash
git diff | bt-tldr
cat error.log | bt-tldr --mode error
```

**bt-namegen** — Generate branch/function/class/variable/file names
```bash
echo "fix broken login page css" | bt-namegen --style branch
echo "validate user email address" | bt-namegen --style function
```

**bt-commit** — Git diff to conventional commit message
```bash
git diff --cached | bt-commit
git diff HEAD~1 | bt-commit --scope auth
```

**bt-changelog** — Git log to release notes, grouped by commit type
```bash
git log --oneline v1.0..v1.1 | bt-changelog
git log --oneline -20 | bt-changelog --format markdown
```

**bt-gitignore** — Template-first .gitignore generator
```bash
bt-gitignore --lang python
bt-gitignore --lang node,rust --append
```

**bt-env** — Scan code for env vars, generate .env templates
```bash
bt-env --scan src/
bt-env --scan . --output .env.example
```

### Code Generation

**bt-regex** — Natural language to regex with --test validation
```bash
echo "match email addresses" | bt-regex
echo "US phone numbers" | bt-regex --test "call 212-555-0100"
```

**bt-cron** — Natural language to cron with --explain and --validate
```bash
echo "every weekday at 9am" | bt-cron
echo "0 */2 * * *" | bt-cron --explain
```

**bt-sql** — Natural language to SQL with schema awareness
```bash
echo "users who signed up this month" | bt-sql --table users
echo "top 10 orders by total" | bt-sql --schema schema.sql
```

**bt-mock** — Generate mock data from type descriptions (JSON/CSV/SQL)
```bash
echo "user with name, email, age" | bt-mock --count 5
echo "product with sku, price, category" | bt-mock --format csv
```

**bt-assert** — Natural language test assertions, binary pass/fail for CI
```bash
echo '{"status":200}' | bt-assert "status code is 200"
curl -s api/health | bt-assert "response contains ok"
```

### Universal

**bt-explain** — Universal explainer for errors, code, config, CLI
```bash
echo "ECONNREFUSED 127.0.0.1:5432" | bt-explain
echo "git rebase -i HEAD~3" | bt-explain
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BITNET_MODEL` | Auto-detected | Path to GGUF model file |
| `BITNET_CLI` | Auto-detected | Path to llama-cli binary |
| `BITNET_THREADS` | `4` | CPU threads for inference |
| `BITNET_CTX_SIZE` | `2048` | Context window size |

### Model Selection

The tools search for models in this order:
1. `BITNET_MODEL` / `BITNET_CLI` environment variables (if set)
2. `models/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf` + `tools/llama-cpp-latest/bin/llama-cli.exe` (GPU)
3. `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf` + `build/bin/llama-cli.exe` (CPU)

## Piping Examples

```bash
# Classify and extract in a pipeline
cat email.txt | bt-classify --preset topic
cat email.txt | bt-extract --type emails --json | jq '.[0]'

# Summarize a git diff
git diff HEAD~5 | bt-tldr

# Convert unstructured data to JSON for processing
cat business_card.txt | bt-jsonify --fields name,title,company,phone,email

# Generate a branch name from a ticket title
echo "Users can't log in after password reset" | bt-namegen --style branch

# Conventional commit from staged changes
git diff --cached | bt-commit

# Release notes from git history
git log --oneline v1.0..HEAD | bt-changelog

# Generate a .env template from source
bt-env --scan src/ --output .env.example

# Mock data for testing
echo "user with id, name, email, role" | bt-mock --count 10 --format csv

# Validate cron expressions
echo "0 9 * * 1-5" | bt-cron --explain

# CI assertion
curl -s localhost:8080/health | bt-assert "response is healthy"
```

All tools read from stdin by default, or from a file with `-i FILE`.

## Test Results

852 tests across all 17 tools. ~99.8% pass rate.

| Tool | Tests | Pass Rate | Notes |
|------|-------|-----------|-------|
| bt-classify | 48 | 96% | 2 debatable edge cases on ambiguous boundaries |
| bt-extract | 25 | 100% | Regex post-validation eliminates hallucinations |
| bt-summarize | 22 | 100% | Echo detection + min-length guard |
| bt-jsonify | 21 | 100% | Prompt injection defense |
| bt-rewrite | 20 | 100% | Style-specific stop tokens |
| bt-tldr | 36 | 100% | 3-tier error detection |
| bt-namegen | 24 | 100% | Space-separated prompt strategy |
| bt-commit | 60 | 100% | Conventional commit format enforcement |
| bt-regex | 23 | 100% | Pattern validation with test input |
| bt-cron | 110 | 100% | Explain + validate modes |
| bt-sql | 80 | 100% | Schema-aware generation |
| bt-explain | 66 | 100% | Multi-domain detection |
| bt-assert | 60 | 100% | Binary exit codes for CI |
| bt-changelog | 53 | 100% | Commit type grouping |
| bt-env | 21 | 100% | Cross-file env var scanning |
| bt-mock | 64 | 100% | JSON/CSV/SQL output formats |
| bt-gitignore | 119 | 100% | Template-first with fallback |

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
