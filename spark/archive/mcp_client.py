#!/usr/bin/env python3
"""MCP Client -- outbound tool consumption for the Spark agent.

Connects to external MCP servers declared in config.yaml's mcpServers
section.  Discovers their tools, registers them with SkillRouter as
plugin skills (mcp:<server>:<tool>), and provides call_tool() for the
agent to invoke them.

Design principles:
    - Async lifecycle managed by asyncio (the agent's bus loop).
    - Each server runs as a subprocess (stdio transport) or via SSE.
    - Discovered tools get Tier.NOTIFY in the policy engine by default.
    - Reconnection with exponential backoff on failure.
    - Graceful shutdown: SIGTERM to subprocesses, drain pending calls.
    - Thread-safe: the agent's main loop is sync; we bridge with
      asyncio.run_coroutine_threadsafe().

Usage:
    client = MCPClientManager(config)
    client.start()          # launch connections in background
    tools = client.tools()  # list all discovered tools
    result = client.call_tool('server_name', 'tool_name', {'arg': 'val'})
    client.stop()           # graceful shutdown

Related:
    - mcp_server.py: inbound MCP (exposes Vybn's skills to external clients)
    - policy.py: gates all tool calls, including MCP-originated ones
    - skills.py: SkillRouter where MCP tools get registered
    - Issue #2275: full integration plan
"""

import asyncio
import logging
import os
import sys
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RECONNECT_ATTEMPTS = 5
INITIAL_BACKOFF_SECONDS = 2.0
TOOL_CALL_TIMEOUT = 60  # seconds
CONNECT_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Data objects
# ---------------------------------------------------------------------------

@dataclass
class MCPServerConfig:
    """Parsed config for a single external MCP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"  # "stdio" or "sse"
    auth_token: Optional[str] = None


@dataclass
class MCPTool:
    """A tool discovered from an external MCP server."""
    server_name: str
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)

    @property
    def skill_name(self) -> str:
        """The name used when registered with SkillRouter."""
        return f"mcp:{self.server_name}:{self.name}"

    def to_skill_description(self) -> str:
        """One-line description for the agent's tool instructions."""
        desc = self.description or "(no description)"
        # Truncate long descriptions for context budget
        if len(desc) > 120:
            desc = desc[:117] + "..."
        return f"{self.skill_name}: {desc}"


# ---------------------------------------------------------------------------
# Single server connection
# ---------------------------------------------------------------------------

class MCPServerConnection:
    """Manages the lifecycle of a single MCP server connection.

    Handles:
        - Subprocess launch (stdio transport)
        - Tool discovery via tools/list
        - Tool invocation via tools/call
        - Reconnection with exponential backoff
        - Graceful shutdown
    """

    def __init__(self, server_config: MCPServerConfig):
        self.config = server_config
        self.session: Optional[ClientSession] = None
        self.exit_stack: Optional[AsyncExitStack] = None
        self.tools: List[MCPTool] = []
        self._connected = False
        self._reconnect_attempts = 0

    @property
    def connected(self) -> bool:
        return self._connected and self.session is not None

    async def connect(self) -> bool:
        """Establish connection to the MCP server and discover tools."""
        if ClientSession is None:
            logger.error("mcp library not installed: pip install mcp")
            return False

        if self.config.transport != "stdio":
            logger.warning(
                "Transport '%s' not yet supported for %s; only stdio is implemented",
                self.config.transport, self.config.name,
            )
            return False

        try:
            self.exit_stack = AsyncExitStack()

            # Build environment: inherit current env + server-specific overrides
            env = dict(os.environ)
            env.update(self.config.env)

            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=env,
            )

            logger.info(
                "Connecting to MCP server '%s': %s %s",
                self.config.name,
                self.config.command,
                " ".join(self.config.args),
            )

            # Launch subprocess and get read/write streams
            read, write = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )

            # Create and initialize client session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await asyncio.wait_for(
                self.session.initialize(),
                timeout=CONNECT_TIMEOUT,
            )

            # Discover tools
            await self._discover_tools()

            self._connected = True
            self._reconnect_attempts = 0

            logger.info(
                "Connected to '%s': %d tools discovered",
                self.config.name, len(self.tools),
            )
            return True

        except asyncio.TimeoutError:
            logger.error(
                "Timeout connecting to MCP server '%s' after %ds",
                self.config.name, CONNECT_TIMEOUT,
            )
            await self._cleanup()
            return False
        except Exception as e:
            logger.error(
                "Failed to connect to MCP server '%s': %s",
                self.config.name, e,
            )
            await self._cleanup()
            return False

    async def _discover_tools(self):
        """Query the server for available tools."""
        self.tools = []
        try:
            response = await asyncio.wait_for(
                self.session.list_tools(),
                timeout=CONNECT_TIMEOUT,
            )
            for tool in response.tools:
                mcp_tool = MCPTool(
                    server_name=self.config.name,
                    name=tool.name,
                    description=getattr(tool, "description", "") or "",
                    input_schema=(
                        tool.inputSchema
                        if hasattr(tool, "inputSchema")
                        else {}
                    ),
                )
                self.tools.append(mcp_tool)
                logger.debug(
                    "  tool: %s -- %s",
                    mcp_tool.skill_name, mcp_tool.description[:80],
                )
        except Exception as e:
            logger.error(
                "Failed to discover tools on '%s': %s",
                self.config.name, e,
            )

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        """Invoke a tool on this server. Returns result text."""
        if not self.connected:
            return f"Error: not connected to server '{self.config.name}'"

        try:
            result = await asyncio.wait_for(
                self.session.call_tool(tool_name, arguments),
                timeout=TOOL_CALL_TIMEOUT,
            )
            # MCP call_tool returns a CallToolResult with .content list
            parts = []
            for content_item in result.content:
                if hasattr(content_item, "text"):
                    parts.append(content_item.text)
                else:
                    parts.append(str(content_item))
            return "\n".join(parts) if parts else "(no output)"

        except asyncio.TimeoutError:
            logger.error(
                "Timeout calling %s on '%s' after %ds",
                tool_name, self.config.name, TOOL_CALL_TIMEOUT,
            )
            return f"Error: tool call timed out after {TOOL_CALL_TIMEOUT}s"
        except Exception as e:
            logger.error(
                "Error calling %s on '%s': %s",
                tool_name, self.config.name, e,
            )
            self._connected = False
            return f"Error: {e}"

    async def reconnect(self) -> bool:
        """Attempt reconnection with exponential backoff."""
        if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            logger.warning(
                "Max reconnection attempts (%d) reached for '%s'",
                MAX_RECONNECT_ATTEMPTS, self.config.name,
            )
            return False

        self._reconnect_attempts += 1
        backoff = INITIAL_BACKOFF_SECONDS * (2 ** (self._reconnect_attempts - 1))
        logger.info(
            "Reconnecting to '%s' in %.1fs (attempt %d/%d)",
            self.config.name, backoff,
            self._reconnect_attempts, MAX_RECONNECT_ATTEMPTS,
        )
        await asyncio.sleep(backoff)
        await self._cleanup()
        return await self.connect()

    async def disconnect(self):
        """Graceful shutdown."""
        logger.info("Disconnecting from MCP server '%s'", self.config.name)
        await self._cleanup()
        self._connected = False
        self.tools = []

    async def _cleanup(self):
        """Release resources."""
        if self.exit_stack:
            try:
                await self.exit_stack.aclose()
            except Exception as e:
                logger.debug("Cleanup error for '%s': %s", self.config.name, e)
            self.exit_stack = None
        self.session = None
        self._connected = False


# ---------------------------------------------------------------------------
# Manager: the public API for agent.py
# ---------------------------------------------------------------------------

class MCPClientManager:
    """Manages all external MCP server connections.

    This is the class that agent.py instantiates. It:
        - Parses mcpServers from config.yaml
        - Launches connections in a background asyncio loop
        - Provides sync call_tool() bridged to async via threading
        - Registers discovered tools with SkillRouter

    Thread safety:
        The agent's main loop is synchronous. This manager runs an
        asyncio event loop in a daemon thread and bridges calls via
        run_coroutine_threadsafe().
    """

    def __init__(self, config: dict):
        self.config = config
        self._servers: Dict[str, MCPServerConnection] = {}
        self._all_tools: List[MCPTool] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False

        # Parse server configs
        self._server_configs = self._parse_configs(config)
        if self._server_configs:
            logger.info(
                "MCP client: %d server(s) configured: %s",
                len(self._server_configs),
                ", ".join(c.name for c in self._server_configs),
            )
        else:
            logger.debug("MCP client: no mcpServers configured")

    @staticmethod
    def _parse_configs(config: dict) -> List[MCPServerConfig]:
        """Parse mcpServers from config.yaml."""
        configs = []
        raw_servers = config.get("mcpServers", []) or []
        for entry in raw_servers:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            command = entry.get("command")
            if not name or not command:
                logger.warning(
                    "Skipping MCP server config missing name or command: %s",
                    entry,
                )
                continue
            configs.append(MCPServerConfig(
                name=name,
                command=command,
                args=entry.get("args", []),
                env=entry.get("env", {}),
                transport=entry.get("transport", "stdio"),
                auth_token=entry.get("auth_token"),
            ))
        return configs

    # ----- lifecycle -----

    def start(self):
        """Start background event loop and connect to all servers."""
        if self._started or not self._server_configs:
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="mcp-client",
            daemon=True,
        )
        self._thread.start()

        # Kick off connections
        future = asyncio.run_coroutine_threadsafe(
            self._connect_all(), self._loop
        )
        # Wait for initial connections (with timeout)
        try:
            future.result(timeout=CONNECT_TIMEOUT * len(self._server_configs))
        except Exception as e:
            logger.error("Error during MCP client startup: %s", e)

        self._started = True

    def stop(self):
        """Graceful shutdown of all connections."""
        if not self._started or not self._loop:
            return

        future = asyncio.run_coroutine_threadsafe(
            self._disconnect_all(), self._loop
        )
        try:
            future.result(timeout=10)
        except Exception as e:
            logger.error("Error during MCP client shutdown: %s", e)

        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._started = False
        logger.info("MCP client stopped")

    def _run_loop(self):
        """Entry point for the background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_all(self):
        """Connect to all configured servers."""
        for cfg in self._server_configs:
            conn = MCPServerConnection(cfg)
            self._servers[cfg.name] = conn
            await conn.connect()
        self._rebuild_tool_list()

    async def _disconnect_all(self):
        """Disconnect from all servers."""
        for conn in self._servers.values():
            await conn.disconnect()
        self._servers.clear()
        self._all_tools.clear()

    def _rebuild_tool_list(self):
        """Rebuild the flat list of all discovered tools."""
        self._all_tools = []
        for conn in self._servers.values():
            self._all_tools.extend(conn.tools)

    # ----- public API (sync, called from agent.py) -----

    def tools(self) -> List[MCPTool]:
        """Return all discovered tools across all connected servers."""
        return list(self._all_tools)

    def tool_descriptions(self) -> List[str]:
        """Return one-line descriptions for all tools (for system prompt)."""
        return [t.to_skill_description() for t in self._all_tools]

    def has_tools(self) -> bool:
        """Whether any MCP tools are available."""
        return len(self._all_tools) > 0

    def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        """Invoke a tool on a named server. Sync wrapper for async call.

        This is what SkillRouter calls when it sees an mcp:* skill.
        """
        if not self._started or not self._loop:
            return "Error: MCP client not started"

        conn = self._servers.get(server_name)
        if not conn:
            return f"Error: no MCP server named '{server_name}'"

        if not conn.connected:
            # Try reconnecting
            future = asyncio.run_coroutine_threadsafe(
                conn.reconnect(), self._loop
            )
            try:
                reconnected = future.result(timeout=CONNECT_TIMEOUT)
                if not reconnected:
                    return (
                        f"Error: MCP server '{server_name}' is disconnected "
                        f"and reconnection failed"
                    )
                self._rebuild_tool_list()
            except Exception as e:
                return f"Error reconnecting to '{server_name}': {e}"

        future = asyncio.run_coroutine_threadsafe(
            conn.call_tool(tool_name, arguments), self._loop
        )
        try:
            return future.result(timeout=TOOL_CALL_TIMEOUT + 5)
        except Exception as e:
            return f"Error calling {tool_name}: {e}"

    def get_server_status(self) -> str:
        """Human-readable status of all servers (for /status command)."""
        if not self._server_configs:
            return "No MCP servers configured"
        lines = ["MCP Servers:"]
        for cfg in self._server_configs:
            conn = self._servers.get(cfg.name)
            if conn and conn.connected:
                status = f"connected ({len(conn.tools)} tools)"
            elif conn:
                status = "disconnected"
            else:
                status = "not started"
            lines.append(f"  {cfg.name}: {status}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SkillRouter integration
# ---------------------------------------------------------------------------

def register_mcp_tools(manager: MCPClientManager, skills_router) -> int:
    """Register all MCP tools with the SkillRouter as plugin skills.

    Each MCP tool becomes a skill named mcp:<server>:<tool>.
    When the SkillRouter encounters such a skill, it delegates to
    manager.call_tool().

    Returns the number of tools registered.
    """
    if not manager.has_tools():
        return 0

    count = 0
    for tool in manager.tools():
        skill_name = tool.skill_name

        # Create a closure that captures the tool's server and name
        def make_handler(srv_name, tl_name):
            def handler(action: dict) -> str:
                args = action.get("params", {})
                if not args:
                    # Try to parse argument as the first positional arg
                    arg = action.get("argument", "")
                    if arg:
                        args = {"input": arg}
                return manager.call_tool(srv_name, tl_name, args)
            return handler

        handler = make_handler(tool.server_name, tool.name)

        # Register with SkillRouter's plugin system
        if hasattr(skills_router, "plugin_handlers"):
            skills_router.plugin_handlers[skill_name] = handler
            logger.info("Registered MCP tool: %s", skill_name)
            count += 1
        else:
            logger.warning(
                "SkillRouter has no plugin_handlers; cannot register %s",
                skill_name,
            )

    return count


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def load_mcp_config(config_path: str = None) -> dict:
    """Load config.yaml and return the full config dict."""
    path = Path(config_path) if config_path else Path(__file__).parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

def main():
    """Standalone test: connect to configured MCP servers and list tools."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    config = load_mcp_config()
    manager = MCPClientManager(config)

    if not manager._server_configs:
        print("No mcpServers configured in config.yaml")
        print("Add a section like:")
        print("  mcpServers:")
        print('    - name: example')
        print('      command: python3')
        print('      args: ["my_mcp_server.py"]')
        return

    print(f"Connecting to {len(manager._server_configs)} MCP server(s)...")
    manager.start()

    print(f"\n{manager.get_server_status()}")

    if manager.has_tools():
        print(f"\nDiscovered {len(manager.tools())} tool(s):")
        for desc in manager.tool_descriptions():
            print(f"  {desc}")
    else:
        print("\nNo tools discovered.")

    manager.stop()


if __name__ == "__main__":
    main()
