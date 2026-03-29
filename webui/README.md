# AI Chat Platform

A model-agnostic AI chat platform with tool calling, MCP support, and multi-provider routing. Runs locally with a ChatGPT-style web interface.

## Quick Start

```bash
pip install fastapi uvicorn[standard] httpx openai anthropic mcp
python -m webui.backend
```

Opens http://localhost:8000 with Qwen3.5-9B running locally on GPU.

## Architecture

```
Browser (localhost:8000)
    ↓
FastAPI Backend
    ├── Provider Router → Local llama-server / OpenAI / Anthropic / Ollama
    ├── Tool Loop → 5 built-in tools + MCP tools
    └── MCP Manager → Any MCP server via stdio
        ↓
llama-server (localhost:8080, internal)
```

## Features

### Chat UI
- ChatGPT-style dark theme with streaming responses
- Markdown rendering with syntax-highlighted code blocks + copy buttons
- Conversation history (localStorage)
- Presets/agents system (save, load, export, import)
- System prompt and generation parameter controls

### Multi-Provider
- **Local**: llama-server with any GGUF model (default: Qwen3.5-9B)
- **OpenAI**: GPT-4, GPT-4o, etc. (requires API key)
- **Anthropic**: Claude models (requires API key)
- **Ollama**: Any Ollama-served model
- **Any OpenAI-compatible endpoint**

Switch providers mid-conversation from the dropdown.

### Tool Calling
5 built-in tools the model can use during conversation:
- `read_file` — Read files from disk
- `write_file` — Write files to disk
- `run_command` — Execute shell commands (30s timeout)
- `web_fetch` — HTTP GET (10KB limit)
- `python_eval` — Run Python code (10s timeout)

Tools appear live in the chat as collapsible blocks.

### MCP (Model Context Protocol)
Connect to any MCP server for extensible tools:
- Filesystem, GitHub, databases, web search, etc.
- 7,000+ MCP servers available in the ecosystem
- Add/remove servers from the settings panel
- MCP tools auto-register into the tool system

## Configuration

### Add a Provider
Settings gear → Providers → Fill in type, URL, API key → Add Provider

### Connect an MCP Server
Settings gear → MCP Servers → Fill in name, command, args → Connect Server

Example (filesystem):
- Name: `filesystem`
- Command: `npx`
- Args: `-y,@modelcontextprotocol/server-filesystem,C:/Users`

### Presets
Save system prompt + generation params as named presets. Export/import as JSON.

## Legacy Launcher

The original `launch.py` still works for basic llama-server + static UI:
```bash
python webui/launch.py
```

## Requirements

- Python 3.10+
- NVIDIA GPU with 8+ GB VRAM (for local models)
- llama-server.exe (included in tools/)
- GGUF model (default: Qwen3.5-9B-Q4_K_M in models/)

## Dependencies

```
fastapi>=0.100
uvicorn[standard]>=0.20
httpx>=0.25
openai>=1.0      # for OpenAI/Ollama providers
anthropic>=0.30  # for Anthropic provider
mcp>=1.0         # for MCP client
```
