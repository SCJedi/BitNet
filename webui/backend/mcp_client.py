"""
MCP (Model Context Protocol) client integration.

Phase 4: Connects to external MCP servers via stdio transport,
discovers their tools, and bridges them into the ToolRegistry.
"""

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    name: str
    command: str                                    # e.g., "npx" or "python"
    args: list[str] = field(default_factory=list)   # e.g., ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
    env: dict[str, str] = field(default_factory=dict)  # extra env vars
    auto_start: bool = True


class MCPConnection:
    """Wraps a single MCP server connection via stdio transport."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session = None
        self._cm_stack: AsyncExitStack | None = None
        self.connected = False
        self.tools: list = []  # list of MCP Tool objects from the server

    async def connect(self):
        """Start the MCP server subprocess and initialize the session."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # Build env — resolve ${VAR} references from the host environment
        env = dict(os.environ)
        for k, v in self.config.env.items():
            if v.startswith("${") and v.endswith("}"):
                env_key = v[2:-1]
                env[k] = os.environ.get(env_key, "")
            else:
                env[k] = v

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=env,
        )

        # Manage async context managers across connect/disconnect boundary
        self._cm_stack = AsyncExitStack()

        try:
            streams = await self._cm_stack.enter_async_context(stdio_client(params))
            self.session = await self._cm_stack.enter_async_context(
                ClientSession(streams[0], streams[1])
            )
            await self.session.initialize()

            # Discover tools
            result = await self.session.list_tools()
            self.tools = result.tools
            self.connected = True
            logger.info(
                "MCP server '%s' connected with %d tools",
                self.config.name, len(self.tools),
            )
        except Exception as e:
            logger.error("Failed to connect MCP server '%s': %s", self.config.name, e)
            await self.disconnect()
            raise

    async def disconnect(self):
        """Disconnect and clean up."""
        if self._cm_stack:
            try:
                await self._cm_stack.aclose()
            except Exception:
                pass
        self.session = None
        self._cm_stack = None
        self.connected = False
        self.tools = []

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool on this MCP server."""
        if not self.session or not self.connected:
            return f"Error: MCP server '{self.config.name}' not connected"
        try:
            result = await self.session.call_tool(name, arguments)
            texts = []
            for block in result.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
                else:
                    texts.append(str(block))
            return "\n".join(texts) if texts else "(empty result)"
        except Exception as e:
            return f"Error calling tool '{name}': {e}"


class MCPManager:
    """Manages multiple MCP server connections and bridges tools to the ToolRegistry."""

    def __init__(self):
        self.connections: dict[str, MCPConnection] = {}
        self.configs: dict[str, MCPServerConfig] = {}

    # ── Server lifecycle ────────────────────────────────────────────────

    async def add_server(self, config: MCPServerConfig) -> list[str]:
        """Add and connect an MCP server. Returns list of tool names."""
        if config.name in self.connections:
            await self.remove_server(config.name)

        self.configs[config.name] = config
        conn = MCPConnection(config)

        try:
            await conn.connect()
            self.connections[config.name] = conn
            return [t.name for t in conn.tools]
        except Exception as e:
            logger.error("Failed to add MCP server '%s': %s", config.name, e)
            self.configs.pop(config.name, None)
            raise

    async def remove_server(self, name: str):
        """Disconnect and remove an MCP server."""
        if name in self.connections:
            await self.connections[name].disconnect()
            del self.connections[name]
        self.configs.pop(name, None)

    def list_servers(self) -> list[dict]:
        """List all configured MCP servers with status."""
        result = []
        for name, cfg in self.configs.items():
            conn = self.connections.get(name)
            result.append({
                "name": name,
                "command": cfg.command,
                "args": cfg.args,
                "env": {
                    k: ("****" if v.startswith("${") else v)
                    for k, v in cfg.env.items()
                },
                "auto_start": cfg.auto_start,
                "connected": conn.connected if conn else False,
                "tool_count": len(conn.tools) if conn and conn.connected else 0,
            })
        return result

    async def shutdown(self):
        """Disconnect all servers."""
        for name in list(self.connections.keys()):
            await self.remove_server(name)

    # ── Tool bridge ─────────────────────────────────────────────────────

    def bridge_to_registry(self, registry) -> int:
        """
        Register all MCP tools into the tool registry.
        Returns the number of tools bridged.
        """
        from .tools import ToolDefinition

        count = 0
        for server_name, conn in self.connections.items():
            if not conn.connected:
                continue
            for mcp_tool in conn.tools:
                tool_name = mcp_tool.name

                # Build a callable that captures the correct connection and tool name.
                # Using default-arg binding to avoid the Python closure-in-loop gotcha.
                def _make_callable(_conn=conn, _tn=tool_name):
                    async def tool_callable(**kwargs):
                        return await _conn.call_tool(_tn, kwargs)
                    return tool_callable

                td = ToolDefinition(
                    name=tool_name,
                    description=getattr(mcp_tool, "description", "") or f"MCP tool from {server_name}",
                    parameters=getattr(mcp_tool, "inputSchema", {}) or {"type": "object", "properties": {}},
                    callable=_make_callable(),
                    enabled=True,
                    source=f"mcp:{server_name}",
                )
                registry.register(td)
                count += 1

        return count

    def unbridge_from_registry(self, registry, server_name: str):
        """Remove all tools from a specific MCP server from the registry."""
        source = f"mcp:{server_name}"
        to_remove = [
            t.name for t in registry._tools.values() if t.source == source
        ]
        for name in to_remove:
            registry.unregister(name)

    # ── Persistence ─────────────────────────────────────────────────────

    def load_from_file(self, path: Path) -> list[MCPServerConfig]:
        """Load MCP server configs from a JSON file. Returns the configs."""
        if not path.exists():
            return []
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load MCP config from %s: %s", path, e)
            return []

        configs = []
        for name, cfg in data.items():
            sc = MCPServerConfig(
                name=name,
                command=cfg.get("command", ""),
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                auto_start=cfg.get("auto_start", True),
            )
            configs.append(sc)
        return configs

    def save_to_file(self, path: Path):
        """Save current configs to a JSON file."""
        data = {}
        for name, cfg in self.configs.items():
            data[name] = {
                "command": cfg.command,
                "args": cfg.args,
                "env": cfg.env,
                "auto_start": cfg.auto_start,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
