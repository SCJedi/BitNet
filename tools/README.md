# bitnet-tools

Seven CLI tools powered by a local LLM. Pipe text in, get structured results out. No API keys, no cloud — everything runs on your GPU.

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

### bt-classify — Classify text into categories

```bash
echo "the app crashes on submit" | bt-classify --labels bug,feature,question
# bug

echo "I love this product!" | bt-classify --preset sentiment
# positive

cat ticket.txt | bt-classify --preset urgency
# critical
```

Presets: `sentiment`, `urgency`, `language`, `topic`. Or use `--labels` with any comma-separated list.

### bt-extract — Extract structured data

```bash
echo "Contact john@example.com or call 212-555-0100" | bt-extract --type emails
# john@example.com

echo "Meeting on March 15, 2024" | bt-extract --type dates --json
# ["2024-03-15"]

cat contacts.txt | bt-extract --type phones --json
# ["212-555-0100", "(800) 555-0199"]
```

Types: `emails`, `phones`, `urls`, `dates`, `names`. Add `--json` for JSON array output. Results are post-validated with regex to filter hallucinations.

### bt-summarize — Summarize text

```bash
cat article.txt | bt-summarize
# One-sentence summary.

cat report.txt | bt-summarize --sentences 3
# Three-sentence summary.

git log --oneline -20 | bt-summarize --sentences 2
# Two-sentence summary of recent commits.
```

Inputs shorter than 40 characters are returned as-is (prevents hallucination on trivial input).

### bt-jsonify — Convert text to JSON

```bash
echo "John Doe, age 30, lives in NYC" | bt-jsonify --fields name,age,city
# {"name": "John Doe", "age": 30, "city": "NYC"}

echo "Order #1234, $49.99, shipped" | bt-jsonify --fields order_id,price,status
# {"order_id": "1234", "price": 49.99, "status": "shipped"}
```

Missing fields default to `null`. Output is filtered to only requested fields (prompt injection safe).

### bt-rewrite — Rewrite text in a style

```bash
echo "hey fix the css bug pls" | bt-rewrite --style formal
# Please address the CSS rendering issue at your earliest convenience.

echo "Python has dynamic typing and garbage collection" | bt-rewrite --style bullets
# * Dynamic typing
# * Garbage collection

echo "fixed login page on mobile" | bt-rewrite --style commit
# Fix mobile login page layout
```

Styles: `formal`, `simple`, `punctuate`, `bullets`, `commit`.

### bt-tldr — Smart TL;DR

```bash
git diff | bt-tldr
# Changed return value from None to computed result.

cat error.log | bt-tldr
# Database connection failed on startup, recovered after retry.

cat traceback.txt | bt-tldr
# ConnectionError: the app can't reach the database at localhost:5432. Check if PostgreSQL is running.
```

Auto-detects input type (diff, log, error, generic) and uses type-specific prompts. Force a mode with `--mode diff|log|error|generic`.

### bt-namegen — Generate code names

```bash
echo "fix broken login page css" | bt-namegen --style branch
# fix/login-page-css

echo "validate user email address" | bt-namegen --style function
# validateUserEmail

echo "store user session data" | bt-namegen --style variable
# store_user_session_data

echo "handle HTTP request routing" | bt-namegen --style class
# HttpRequestRouter
```

Styles: `branch` (kebab-case with prefix), `function` (camelCase), `file` (snake_case), `class` (PascalCase), `variable` (snake_case).

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

# Rewrite commit messages
git log --format="%s" -1 | bt-rewrite --style commit
```

All tools read from stdin by default, or from a file with `-i FILE`.

## Test Results

196 tests across all 7 tools. 99.5% pass rate (194/196).

| Tool | Tests | Pass Rate | Notes |
|------|-------|-----------|-------|
| bt-classify | 48 | 96% | 2 debatable edge cases on ambiguous boundaries |
| bt-extract | 25 | 100% | Regex post-validation eliminates hallucinations |
| bt-summarize | 22 | 100% | Echo detection + min-length guard |
| bt-jsonify | 21 | 100% | Prompt injection defense |
| bt-rewrite | 20 | 100% | Style-specific stop tokens |
| bt-tldr | 36 | 100% | 3-tier error detection |
| bt-namegen | 24 | 100% | Space-separated prompt strategy |

Full details: [TEST_REPORT_FINAL.md](TEST_REPORT_FINAL.md)

## Architecture

```
User input (stdin or -i FILE)
    │
    ▼
CLI tool (classify.py, extract.py, etc.)
    │  - Builds prompt from input + flags
    │  - Sets max_tokens, stop_at, temperature
    │
    ▼
engine.py → complete()
    │  - Truncates input if > 6000 chars
    │  - Builds llama-cli command
    │  - Runs subprocess with timeout
    │  - Strips banner, ANSI codes, think tags
    │  - clean_output(): strips fences, prefixes,
    │    markdown, deduplicates, truncates
    │
    ▼
config.py
    │  - Auto-detects model + CLI binary
    │  - Reads env var overrides
    │  - Validates paths exist
    │
    ▼
llama-cli (subprocess)
    │  - GPU mode: --single-turn -ngl 99 -rea off
    │  - CPU mode: -ngl 0 -b 1
    │
    ▼
GGUF model file
```

## Requirements

- Python 3.10+
- No Python dependencies (uses only stdlib)
- llama.cpp binary (included in `tools/llama-cpp-latest/` for GPU, or `build/bin/` for BitNet CPU)
- A GGUF model file (Qwen3.5-4B-Q4_K_M recommended)
- GPU: NVIDIA with CUDA support (optional, falls back to CPU)
