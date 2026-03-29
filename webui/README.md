# Qwen3.5-9B Chat UI

A ChatGPT-style web interface for the local Qwen3.5-9B model, served via llama-server (llama.cpp).

## Quick Start

```
python webui/launch.py
```

This will:
1. Start llama-server with the Qwen3.5-9B model
2. Wait for the model to load
3. Open your browser to http://localhost:8080

Press Ctrl+C to stop the server.

## Manual Start

1. Start the server:
   ```
   tools/llama-cpp-latest/bin/llama-server.exe -m models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -c 8192 --host 127.0.0.1 --port 8080 --path webui/
   ```
2. Open http://localhost:8080

## Features

- ChatGPT-style dark theme interface
- Streaming responses (token by token)
- Markdown rendering with syntax-highlighted code blocks
- Copy button on code blocks
- Collapsible "Thinking" sections (for Qwen3.5's `<think>` blocks)
- Conversation history persisted in localStorage
- Multiple conversations with sidebar navigation
- Temperature and max tokens controls
- Stop generation button
- Responsive layout (works on mobile)

## Requirements

- Python 3.7+ (for the launcher)
- llama-server.exe (already included in tools/)
- Qwen3.5-9B-Q4_K_M.gguf model (already in models/)
