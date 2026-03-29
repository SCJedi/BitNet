"""
Tool calling system for the AI Chat Platform.

Phase 3: Provides tool definitions, a registry, built-in tools,
and a chat-with-tools loop that handles the OpenAI function calling protocol.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import httpx

from . import config


# ── Tool Definition ──────────────────────────────────────────────────────

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict          # JSON Schema
    callable: Callable        # async def fn(**kwargs) -> str
    enabled: bool = True
    source: str = "builtin"   # "builtin" or "mcp:{server}"


# ── Tool Registry ────────────────────────────────────────────────────────

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        """Return tool info for the API (no callable)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "enabled": t.enabled,
                "source": t.source,
            }
            for t in self._tools.values()
        ]

    def get_openai_tools(self) -> list[dict]:
        """Return tools in OpenAI function calling format (only enabled tools)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
            if t.enabled
        ]

    def set_enabled(self, name: str, enabled: bool) -> None:
        tool = self._tools.get(name)
        if tool is not None:
            tool.enabled = enabled

    def has_enabled_tools(self) -> bool:
        return any(t.enabled for t in self._tools.values())


# ── Built-in Tools ───────────────────────────────────────────────────────

MAX_FILE_SIZE = 10 * 1024  # 10 KB
COMMAND_TIMEOUT = 30
PYTHON_TIMEOUT = 10
FETCH_MAX_SIZE = 10 * 1024  # 10 KB
TOOL_OUTPUT_LIMIT = 2000


async def tool_read_file(path: str) -> str:
    """Read a file, limited to 10KB."""
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return f"Error: '{path}' is not a file or does not exist."
        size = p.stat().st_size
        if size > MAX_FILE_SIZE:
            return f"Error: File is {size} bytes, exceeds {MAX_FILE_SIZE} byte limit."
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"


async def tool_write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {p}"
    except Exception as e:
        return f"Error writing file: {e}"


async def tool_run_command(command: str) -> str:
    """Run a shell command with 30s timeout."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(config.PROJECT_ROOT),
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=COMMAND_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: Command timed out after {COMMAND_TIMEOUT}s"
        output = stdout.decode("utf-8", errors="replace")
        if len(output) > MAX_FILE_SIZE:
            output = output[:MAX_FILE_SIZE] + "\n... (truncated)"
        return output if output else "(no output)"
    except Exception as e:
        return f"Error running command: {e}"


async def tool_web_fetch(url: str) -> str:
    """HTTP GET a URL, return first 10KB of text."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            text = r.text
            if len(text) > FETCH_MAX_SIZE:
                text = text[:FETCH_MAX_SIZE] + "\n... (truncated)"
            return text
    except Exception as e:
        return f"Error fetching URL: {e}"


async def tool_python_eval(code: str) -> str:
    """Run Python code in a subprocess with 10s timeout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(config.PROJECT_ROOT),
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=PYTHON_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: Python execution timed out after {PYTHON_TIMEOUT}s"
        output = stdout.decode("utf-8", errors="replace")
        if len(output) > MAX_FILE_SIZE:
            output = output[:MAX_FILE_SIZE] + "\n... (truncated)"
        return output if output else "(no output)"
    except Exception as e:
        return f"Error running Python: {e}"


def register_builtin_tools(registry: ToolRegistry) -> None:
    """Register all built-in tools."""
    registry.register(ToolDefinition(
        name="read_file",
        description="Read a file from the filesystem. Limited to 10KB.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read",
                },
            },
            "required": ["path"],
        },
        callable=tool_read_file,
    ))

    registry.register(ToolDefinition(
        name="write_file",
        description="Write content to a file on the filesystem.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write",
                },
            },
            "required": ["path", "content"],
        },
        callable=tool_write_file,
    ))

    registry.register(ToolDefinition(
        name="run_command",
        description="Run a shell command and return stdout+stderr. 30s timeout.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
            },
            "required": ["command"],
        },
        callable=tool_run_command,
    ))

    registry.register(ToolDefinition(
        name="web_fetch",
        description="Fetch a URL via HTTP GET and return the first 10KB of text content.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
            },
            "required": ["url"],
        },
        callable=tool_web_fetch,
    ))

    registry.register(ToolDefinition(
        name="python_eval",
        description="Run Python code in a subprocess and return the output. 10s timeout.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute",
                },
            },
            "required": ["code"],
        },
        callable=tool_python_eval,
    ))


# ── Chat with Tools Loop ────────────────────────────────────────────────

async def chat_with_tools(
    provider,
    messages: list[dict],
    registry: ToolRegistry,
    temperature: float,
    max_tokens: int,
    model: str = "",
    max_rounds: int = 10,
    **kwargs,
) -> AsyncIterator[bytes]:
    """
    Generator that yields SSE bytes. Handles the tool call loop:

    1. Call provider with messages + tools
    2. Collect the full response (streaming content deltas to caller)
    3. If response contains tool_calls:
       a. Yield custom SSE event for tool_call
       b. Execute each tool
       c. Yield custom SSE event for tool_result
       d. Append assistant tool_calls message + tool results to messages
       e. Go to step 1
    4. If no tool_calls, we're done
    5. Yield "data: [DONE]"
    """
    tools_openai = registry.get_openai_tools()

    for round_num in range(max_rounds):
        # Accumulate tool calls and content from the streaming response
        tool_calls_acc: dict[int, dict] = {}  # index -> {id, name, arguments}
        content_acc = ""
        finish_reason = None

        # Stream from the provider
        raw_buffer = ""
        async for chunk in provider.chat_stream(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools_openai,
            **kwargs,
        ):
            # chunk is raw bytes from the SSE stream
            raw_buffer += chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk

            # Process complete SSE lines
            while "\n" in raw_buffer:
                line, raw_buffer = raw_buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    # Don't forward [DONE] yet — we may need another round
                    continue

                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                # Check for error
                if "error" in data:
                    yield f"data: {json.dumps(data)}\n\n".encode()
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

                # Accumulate tool calls from streaming deltas
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc["index"]
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc.get("id", ""),
                                "name": "",
                                "arguments": "",
                            }
                        if "function" in tc:
                            if "name" in tc["function"]:
                                tool_calls_acc[idx]["name"] += tc["function"]["name"]
                            if "arguments" in tc["function"]:
                                tool_calls_acc[idx]["arguments"] += tc["function"]["arguments"]

                # Forward content deltas to the client
                if "content" in delta and delta["content"]:
                    content_acc += delta["content"]
                    yield f"data: {json.dumps(data)}\n\n".encode()

        # Process any remaining data in buffer
        remaining = raw_buffer.strip()
        if remaining and remaining.startswith("data: "):
            payload = remaining[6:]
            if payload != "[DONE]":
                try:
                    data = json.loads(payload)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                idx = tc["index"]
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {
                                        "id": tc.get("id", ""),
                                        "name": "",
                                        "arguments": "",
                                    }
                                if "function" in tc:
                                    if "name" in tc["function"]:
                                        tool_calls_acc[idx]["name"] += tc["function"]["name"]
                                    if "arguments" in tc["function"]:
                                        tool_calls_acc[idx]["arguments"] += tc["function"]["arguments"]
                        if "content" in delta and delta["content"]:
                            content_acc += delta["content"]
                            yield f"data: {json.dumps(data)}\n\n".encode()
                except json.JSONDecodeError:
                    pass

        # If no tool calls, we're done
        if not tool_calls_acc:
            break

        # We have tool calls — execute them
        # Build the assistant message with tool_calls
        assistant_tool_calls = []
        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            assistant_tool_calls.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                },
            })

        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if content_acc:
            assistant_msg["content"] = content_acc
        else:
            assistant_msg["content"] = None
        assistant_msg["tool_calls"] = assistant_tool_calls
        messages.append(assistant_msg)

        # Execute each tool and send events
        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            tool_name = tc["name"]

            # Parse arguments
            try:
                args = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                args = {}

            # Yield tool_call event to client
            yield f'data: {json.dumps({"type": "tool_call", "name": tool_name, "arguments": args})}\n\n'.encode()

            # Execute the tool
            tool_def = registry.get(tool_name)
            if tool_def is None:
                result = f"Error: Unknown tool '{tool_name}'"
            else:
                try:
                    result = await tool_def.callable(**args)
                except Exception as e:
                    result = f"Error executing {tool_name}: {e}"

            # Truncate result for the SSE event to client
            result_preview = result[:TOOL_OUTPUT_LIMIT] if len(result) > TOOL_OUTPUT_LIMIT else result

            # Yield tool_result event to client
            yield f'data: {json.dumps({"type": "tool_result", "name": tool_name, "content": result_preview})}\n\n'.encode()

            # Append tool result message for the next round
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        # Loop continues for the next round

    # All done
    yield b"data: [DONE]\n\n"
