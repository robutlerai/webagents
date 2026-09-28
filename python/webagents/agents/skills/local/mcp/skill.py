"""
Local MCP Skill

Connects to and uses Model Context Protocol (MCP) servers.
Matches Gemini CLI specification for discovery and execution.
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from contextlib import AsyncExitStack

from ...base import Skill
from webagents.agents.tools.decorators import tool, command

from ..secrets.references import (
    SecretReferenceError,
    at_connect_sentence,
    expand_references,
    mask_map,
    mask_text,
    mask_url,
)
from .config import SANDBOX_UNAVAILABLE, asks_for_sandbox, filter_discovered_tools, qualified_tool_name, sdk_missing, servers_from_config

logger = logging.getLogger("webagents.skills.mcp")

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import get_default_environment, stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
    MCP_IMPORT_ERROR = ""
except ImportError as _import_error:
    MCP_AVAILABLE = False
    # Said when the skill is BUILT (`__init__` raises `sdk_missing`), so the
    # agent file's loader reports it as a skill that failed, with the reason,
    # rather than as a warning nobody reads and an agent with no MCP tools.
    MCP_IMPORT_ERROR = str(_import_error)
    # Stand-in bindings (mongodb-skill pattern): parameter annotations such
    # as ``session: ClientSession`` are evaluated at class creation on
    # Python <= 3.13 and an unbound name raises NameError there, which the
    # ``except ImportError`` guards upstream cannot catch (F-040).
    class ClientSession:  # type: ignore[no-redef]
        pass

    class StdioServerParameters:  # type: ignore[no-redef]
        pass

    stdio_client = None  # type: ignore[assignment]
    sse_client = None  # type: ignore[assignment]

    def get_default_environment() -> Dict[str, str]:  # type: ignore[misc]
        return {}


def mcp_stderr_log(name: str):
    """Where an MCP stdio server's stderr goes (B8, 2026-09-28): appended to
    `<profile folder>/logs/mcp-<name>.log`, the folder the chat's `repl.log`
    is in, never to the terminal the chat draws on. The TypeScript twin is
    `skills/mcp/skill.ts` `mcpStderrLog`. The null device when that folder
    cannot be written, so the chat stays clean either way."""
    import re

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name) or "server"
    try:
        from webagents.cli.config_store import global_dir

        folder = global_dir() / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        return open(folder / f"mcp-{safe}.log", "a", encoding="utf-8")
    except Exception:  # noqa: BLE001 - no log folder is no log, never the terminal
        return open(os.devnull, "w", encoding="utf-8")

class McpConnectError(RuntimeError):
    """Why a server did not connect: the sentence (every resolved value
    masked) and, when a `${secret:NAME}` was not stored, the names, so
    `doctor` can print the `webagents secrets set` command that fixes it."""

    def __init__(self, message: str, missing_secrets: Optional[List[str]] = None, missing_env: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.missing_secrets = list(missing_secrets or [])
        #: `${env:NAME}` variables that are not set (2026-09-26), for `doctor`'s fix line.
        self.missing_env = list(missing_env or [])


#: The address keys a server may write, each resolved at connect time.
_ADDRESS_KEYS = ("url", "httpUrl", "mcpUrlTemplate")


def owner_reference_sources(env: Optional[Any] = None) -> Dict[str, Any]:
    """The sources an agent file's servers resolve `${env:NAME}` and
    `${secret:NAME}` against (S-295, 2026-09-26): the process environment,
    and the CLI's own secret store (the one `webagents secrets set NAME`
    writes, for the active profile), opened on the first reference and never
    before. For the agent-file loaders only (`cli/agent_builder.py`,
    `cli/daemon/manager.py`; the TypeScript `skills/resolve.ts`): they build
    a skill for a file the local owner wrote and runs. A host that builds the
    skill from data its users saved passes no `references`, and the skill
    then expands nothing (see `LocalMcpSkill.__init__`)."""
    store: Dict[str, Any] = {}

    def secret(name: str) -> Optional[str]:
        if "store" not in store:
            from webagents.cli.commands.secrets import _store

            store["store"] = _store(quiet=True)
        return store["store"].get(name)

    return {"env": os.environ if env is None else env, "secret": secret}

# Streamable HTTP (2026-09-26): the transport the TypeScript skill already had.
# Optional on its own, so an `mcp` without it still serves stdio and SSE.
try:
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:
    streamablehttp_client = None  # type: ignore[assignment]

class LocalMcpSkill(Skill):
    """MCP Client capabilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = config or {}
        self.agent_name = config.get("agent_name", "unknown")
        self.agent_path = config.get("agent_path")
        self.base_dir = config.get("base_dir")
        
        # State
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.tools_registry: Dict[str, Dict[str, Any]] = {}
        self.resources_registry: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        #: What the file named, as the normalizer read it, for `server_report()`.
        self._resolution = None
        #: Where the servers were read from (the chat's `/mcp`, spec 3.7): the
        #: config handed in (the agent file's `- mcp:` entry), or `mcp.json`.
        self.config_source = "config"
        #: Why a server did not connect, by name, masked (S-292).
        self.connect_errors: Dict[str, Dict[str, Any]] = {}
        #: Each server's entry as the file wrote it, by name, for the keys
        #: discovery reads (`enabledTools`, `toolPolicies`; S-286 addendum).
        self._server_configs: Dict[str, Dict[str, Any]] = {}
        # EVERY CONNECTION LIVES IN ONE TASK (2026-09-26, the e2e run's HIGH
        # bug). `stdio_client` and `ClientSession` are anyio task groups, and
        # anyio requires a cancel scope to be exited by the task that entered
        # it, in the order it entered them. `initialize()` used to enter them
        # on `exit_stack` in whatever task called it, and the chat's `/reload`
        # then built the NEW agent's servers in the same task before closing
        # the old ones, so the old scopes were exited out of order: anyio
        # kept delivering a cancellation to the chat's main task, which died
        # with `CancelledError` at its next prompt. Now `_serve_connections`
        # runs in its own task: it enters every server there, holds them
        # open until `cleanup()` asks it to stop, and exits them there, in
        # order, whatever task `initialize()` or `cleanup()` ran in.
        self._runner: Optional["asyncio.Task[None]"] = None
        self._closing: Optional[asyncio.Event] = None
        # RESOLUTION IS OFF UNLESS `references` IS SET (S-295, CRITICAL,
        # 2026-09-26). This skill resolved `${env:NAME}` and `${secret:NAME}`
        # in every server's url, headers and env against `os.environ` and the
        # CLI keystore for EVERY skill, so a host that builds one from data
        # its users saved could have its own environment expanded into a URL
        # and sent to that server. `references` is `{"env": mapping, "secret":
        # callable}` (`owner_reference_sources()`); without it nothing is
        # expanded and every field is used as the literal bytes written. Only
        # the agent-file loaders pass it, for a file the local owner wrote.
        self._references = self.config.get("references")

        logger.debug(f"[MCP] __init__ for agent={self.agent_name}, config keys={list(self.config.keys())}")

        if not MCP_AVAILABLE:
            # A load-time error, as in TypeScript (`loadMcpSdk`): the file's
            # loader records the skill as failed and says why, instead of
            # starting an agent that quietly has none of its MCP tools.
            raise RuntimeError(sdk_missing(MCP_IMPORT_ERROR or "the mcp package is not installed"))

    async def initialize(self, agent):
        """Initialize and connect to configured servers"""
        logger.info(f"[MCP] initialize() called for agent={self.agent_name}, already_initialized={self._initialized}")
        
        if self._initialized:
            logger.debug(f"[MCP] Already initialized, skipping. sessions={list(self.sessions.keys())}")
            return
        
        await super().initialize(agent)
        
        if not MCP_AVAILABLE:
            raise RuntimeError(sdk_missing(MCP_IMPORT_ERROR or "the mcp package is not installed"))

        resolution = self._load_mcp_config()
        self._refuse_unprovidable_sandbox(resolution)
        self._resolution = resolution
        for rejected in resolution.rejected:
            logger.warning(f"[MCP] Server '{rejected['name']}' {rejected['reason']}; skipping it.")
        # Said once, at load (S-292): a literal that looks like a key, with
        # the reference and the command that keep it out of the file.
        for warning in resolution.warnings:
            logger.warning(f"[MCP] Server '{warning['name']}' {warning['reason']}")
        if not resolution.servers:
            logger.info(f"[MCP] No MCP servers configured for agent={self.agent_name}")
            self._initialized = True
            return

        logger.info(f"[MCP] Connecting to {len(resolution.servers)} server(s): {[s['name'] for s in resolution.servers]}")
        loop = asyncio.get_running_loop()
        ready: "asyncio.Future[None]" = loop.create_future()
        self._closing = asyncio.Event()
        self._runner = asyncio.create_task(self._serve_connections(resolution, ready), name=f"mcp:{self.agent_name}")
        try:
            await ready
        except asyncio.CancelledError:
            # The caller gave up (a timeout, a stopped chat): the connections
            # go with it, closed by the task that opened them.
            self._runner.cancel()
            raise

        self._initialized = True
        logger.info(f"[MCP] Initialization complete. tools={len(self.tools_registry)}, sessions={list(self.sessions.keys())}")

    async def _serve_connections(self, resolution: Any, ready: "asyncio.Future[None]") -> None:
        """Every server's whole life, in this one task (see `__init__`): open
        them in the file's order, report ready, hold them until `cleanup()`
        sets `_closing`, then close them here, last opened first."""
        try:
            for server in resolution.servers:
                name = server["name"]
                try:
                    await self._connect_server(name, server, resolution.configs.get(name))
                    logger.info(f"[MCP] Connected to server: {name}")
                except Exception as e:  # noqa: BLE001 - one server's failure must not stop the others
                    # The sentence only, already masked by `_connect_server`: a
                    # traceback could carry a transport's request and repeat a value.
                    logger.error(f"[MCP] Server '{name}' failed to connect: {e}")
        except BaseException as error:
            # Cancelled while connecting: nothing is ready, and what opened closes here.
            if not ready.done():
                if isinstance(error, asyncio.CancelledError):
                    ready.cancel()
                else:
                    ready.set_exception(error)
            await self._close_stack()
            raise
        if not ready.done():
            ready.set_result(None)
        try:
            assert self._closing is not None
            await self._closing.wait()
        finally:
            await self._close_stack()

    async def _close_stack(self) -> None:
        """Exit every context on the stack, in the task that entered them, and start a fresh stack."""
        stack, self.exit_stack = self.exit_stack, AsyncExitStack()
        try:
            await stack.aclose()
        except Exception as error:  # noqa: BLE001 - a server that will not close cleanly is logged, not fatal
            logger.warning(f"[MCP] Closing the servers: {error}")

    def _sandbox_skill(self) -> Any:
        """The Docker `SandboxSkill` this agent loaded, if any: the one thing that can box an MCP server here."""
        if not self.agent or not getattr(self.agent, "skills", None):
            return None
        for skill in self.agent.skills.values():
            if skill.__class__.__name__ == "SandboxSkill":
                return skill
        return None

    def _refuse_unprovidable_sandbox(self, resolution: Any) -> None:
        """Move every server whose entry asks for a sandbox (`sandbox: true`)
        to `rejected` when the agent has no Docker sandbox skill to give it
        one (S-313, `config.SANDBOX_UNAVAILABLE`). Decided here, at
        initialize, because the normalizer never sees the agent's skills."""
        if self._sandbox_skill() is not None:
            return
        kept = []
        for server in resolution.servers:
            name = server["name"]
            if asks_for_sandbox(resolution.configs.get(name)):
                resolution.rejected.append({"name": name, "reason": SANDBOX_UNAVAILABLE})
                resolution.configs.pop(name, None)
            else:
                kept.append(server)
        resolution.servers[:] = kept

    def _load_mcp_config(self):
        """The servers to connect to (`config.py`, pinned by the shared fixture):
        the `mcp` entry the agent builder passes, in either shape; else server
        maps written at the top level of the config (an embedder building the
        skill directly); else `mcp.json` next to the agent, which a bare
        `- mcp` entry means, as it does in TypeScript."""
        raw = self.config.get("mcp")
        self.config_source = "config"
        if raw is None:
            found = {
                key: value
                for key, value in self.config.items()
                if key not in ("agent_name", "agent_path", "base_dir")
                and isinstance(value, dict)
                and any(k in value for k in ("command", "url", "httpUrl", "mcpUrlTemplate"))
            }
            raw = found or None
        if not raw:
            raw = self._read_mcp_json()
            self.config_source = "mcp.json"
        return servers_from_config(raw or {})

    def _read_mcp_json(self) -> Optional[Dict[str, Any]]:
        """`mcp.json` next to the agent; None when there is none."""
        base = self.agent_path or self.base_dir
        if not base:
            return None
        agent_dir = Path(base)
        if agent_dir.is_file():
            agent_dir = agent_dir.parent
        config_path = agent_dir / "mcp.json"
        if not config_path.exists():
            return None
        try:
            return json.loads(config_path.read_text())
        except Exception as e:  # noqa: BLE001 - a broken file is said, and means no servers
            logger.error(f"[MCP] Error loading mcp.json: {e}")
            return None

    # -- secrets (S-292) ---------------------------------------------------

    def _lookup_secret(self, name: str) -> Optional[str]:
        """What `${secret:NAME}` reads: the reader the builder handed in
        (`references`, S-295); never anything hard-wired."""
        references = self._references or {}
        reader = references.get("secret")
        return reader(name) if callable(reader) else None

    def _resolve_references(self, name: str, server: Dict[str, Any]) -> tuple:
        """`server` with every reference in `env`, `headers` and the address
        replaced, for this connection only, plus every resolved value. The
        normalizer's `server` stays as written, so nothing that reports or
        saves a configuration can see a value. Raises `McpConnectError` with
        the connect-time sentence, which names the reference and never a value.

        ONLY WITH SOURCES (S-295): a skill built without `references` expands
        nothing, and every field is used as the literal bytes written."""
        values: List[str] = []
        if not isinstance(self._references, dict):
            return dict(server), values
        env_source = self._references.get("env")
        env_source = env_source if env_source is not None else {}

        def expand(value: str, field: str, key: Optional[str] = None) -> str:
            try:
                text, found = expand_references(str(value), self._lookup_secret, env_source)
            except SecretReferenceError as error:
                raise McpConnectError(
                    at_connect_sentence(name, field, key, str(error)),
                    [error.missing_secret] if error.missing_secret else [],
                    [error.missing_env] if getattr(error, "missing_env", None) else [],
                ) from None
            values.extend(found)
            return text

        live = dict(server)
        if isinstance(server.get("env"), dict):
            live["env"] = {key: expand(value, "env", key) for key, value in server["env"].items()}
        if isinstance(server.get("headers"), dict):
            live["headers"] = {key: expand(value, "headers", key) for key, value in server["headers"].items()}
        for key in _ADDRESS_KEYS:
            if isinstance(server.get(key), str) and server[key]:
                live[key] = expand(server[key], key)
        return live, values

    def server_report(self) -> List[Dict[str, Any]]:
        """Every server the file named, for `/mcp` and `doctor`: connected or
        not, its tools, why it failed or was refused, the loader's warnings,
        and its configuration with references as written and every other
        value masked."""
        rows: List[Dict[str, Any]] = []
        resolution = self._resolution
        if resolution is None:
            return rows
        for server in resolution.servers:
            name = server["name"]
            failure = self.connect_errors.get(name)
            row: Dict[str, Any] = {
                "name": name,
                "transport": server.get("transport"),
                "connected": name in self.sessions,
                "tools": sorted(t for t, info in self.tools_registry.items() if info.get("server") == name),
                "missing_secrets": list(failure["missing_secrets"]) if failure else [],
                "missing_env": list(failure.get("missing_env", [])) if failure else [],
                "warnings": [w["reason"] for w in resolution.warnings if w["name"] == name],
            }
            if failure:
                row["error"] = failure["message"]
            if isinstance(server.get("env"), dict):
                row["env"] = mask_map(server["env"])
            if server.get("headers"):
                row["headers"] = mask_map(server["headers"])
            if server.get("url"):
                row["url"] = mask_url(server["url"])
            rows.append(row)
        for rejected in resolution.rejected:
            rows.append({
                "name": rejected["name"],
                "transport": "unknown",
                "connected": False,
                "tools": [],
                "rejected": rejected["reason"],
                "missing_secrets": [],
                "warnings": [],
            })
        return rows

    async def _connect_server(self, name: str, server: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Connect to one server: `server` as the normalizer describes it,
        `config` as the file wrote it (for keys such as `sandbox`). The
        normalizer's `server` is kept AS WRITTEN; the copy with references
        resolved exists for this call only. Whatever fails, the error that
        leaves here is a plain sentence with every resolved value masked,
        recorded for `server_report()`."""
        values: List[str] = []
        self._server_configs[name] = config if config is not None else server
        try:
            live, values = self._resolve_references(name, server)
            await self._open_server(name, live, config if config is not None else server)
        except Exception as error:  # noqa: BLE001 - masked and re-raised as one sentence
            message = mask_text(str(error) or error.__class__.__name__, values)
            missing = error.missing_secrets if isinstance(error, McpConnectError) else []
            unset = error.missing_env if isinstance(error, McpConnectError) else []
            self.connect_errors[name] = {"message": message, "missing_secrets": list(missing), "missing_env": list(unset)}
            raise McpConnectError(message, missing, unset) from None

    async def _open_server(self, name: str, server: Dict[str, Any], config: Dict[str, Any]):
        """Open one server from `server`, its references already resolved."""
        logger.debug(
            f"[MCP] _connect_server: name={name}, transport={server.get('transport')}, "
            f"env={mask_map(server.get('env'))}, headers={mask_map(server.get('headers'))}, "
            f"url={mask_url(server['url']) if server.get('url') else None}"
        )

        # Determine default CWD (Agent's directory)
        # Note: self.agent_path is already the agent DIRECTORY, not the file
        default_cwd = None
        if self.agent_path:
            agent_path = Path(self.agent_path)
            # If it's a file, get the parent directory
            if agent_path.is_file():
                default_cwd = str(agent_path.parent.resolve())
            else:
                default_cwd = str(agent_path.resolve())
            logger.debug(f"[MCP] Using agent_path as CWD: {default_cwd}")
        elif self.base_dir:
            default_cwd = self.base_dir
            logger.debug(f"[MCP] Using base_dir as CWD: {default_cwd}")

        if server.get("transport") == "stdio":
            command = server["command"]
            args = list(server.get("args", []))
            # The MCP SDK's default environment (PATH, HOME and the like) plus
            # ONLY this entry's `env`, resolved (S-292, 2026-09-26). It was
            # `{**os.environ, **env}`: every variable of the agent process,
            # the provider keys the chat had put there included, went to a
            # server that `npx -y` or `uvx` had just fetched.
            declared_env = dict(server.get("env", {}))
            env = {**get_default_environment(), **declared_env}
            cwd = server.get("cwd") or default_cwd

            # Auto-detect Docker Sandbox
            use_sandbox = asks_for_sandbox(config)
            sandbox_skill = self._sandbox_skill()

            # If sandbox explicitly requested OR Sandbox skill present (implicit mode), use it
            if use_sandbox or sandbox_skill:
                if not sandbox_skill and use_sandbox:
                    # A sandbox the agent cannot give is a refusal, never a
                    # server run with the owner's permissions under a line
                    # that says otherwise (S-313). `initialize` refuses such
                    # an entry before it gets here; this holds for a direct
                    # caller.
                    raise McpConnectError(SANDBOX_UNAVAILABLE)
                elif sandbox_skill:
                    # Ensure container is running
                    await sandbox_skill.ensure_started()
                    container_name = sandbox_skill.get_container_name()
                    
                    # Map arguments that look like paths
                    mapped_args = []
                    for arg in args:
                        # Simple heuristic: if it looks like an absolute path in agent dir, map it
                        # If it's relative, keep it (relative to /workspace)
                        if arg.startswith("/") and Path(arg).exists():
                             mapped_args.append(sandbox_skill.map_path(arg))
                        else:
                             mapped_args.append(arg)
                    args = mapped_args
                    
                    # Wrap command in docker exec
                    new_args = ["exec", "-i"]
                    if cwd:
                         # Attempt to map CWD if it's set
                         mapped_cwd = sandbox_skill.map_path(cwd)
                         # If mapping failed (returned same path) but it's absolute, default to /workspace
                         if mapped_cwd == cwd and cwd.startswith("/"):
                              mapped_cwd = "/workspace"
                         new_args.extend(["-w", mapped_cwd]) 
                    
                    # The declared variables, by NAME only: `docker exec -e K`
                    # takes K's value from the docker client's own
                    # environment, which is `env` below, so a resolved
                    # secret never sits on the docker command line (S-292;
                    # this used to write `-e K=value`).
                    for k in declared_env:
                        new_args.extend(["-e", k])

                    new_args.extend([container_name, command])
                    new_args.extend(args)
                    
                    command = "docker"
                    args = new_args
                    cwd = None 

            # Stdio transport
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env,
                cwd=cwd
            )
            logger.info(f"[MCP] Connecting to server '{name}': command={command}, args={args}, cwd={cwd}")
            
            # THE SERVER'S STDERR IS NOT THE CHAT'S (B8, 2026-09-28): the
            # client's default hands the child this process's stderr, so a
            # server's banner and warnings were drawn over the chat. It goes to
            # `<profile folder>/logs/mcp-<name>.log` (`mcp_stderr_log`), as the
            # TypeScript skill sends it.
            errlog = mcp_stderr_log(name)
            self.exit_stack.callback(errlog.close)
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params, errlog=errlog))
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            logger.info(f"[MCP] Session initialized for server '{name}'")
            
            self.sessions[name] = session
            await self._discover_capabilities(name, session)
            logger.info(f"[MCP] Capabilities discovered for server '{name}': tools={len([t for t, info in self.tools_registry.items() if info.get('server') == name])}")
            
        else:
            # A remote server: Streamable HTTP, SSE, or `auto` (Streamable
            # HTTP first, then SSE), the TypeScript skill's rule.
            url = server["url"]
            headers = server.get("headers", {})
            transport = server.get("transport", "auto")
            session = None
            if transport in ("http", "auto"):
                if streamablehttp_client is None:
                    if transport == "http":
                        raise RuntimeError(f"Server '{name}' needs Streamable HTTP, which this mcp package does not have.")
                else:
                    try:
                        session = await self._open_remote(streamablehttp_client(url, headers=headers))
                    except Exception:
                        if transport == "http":
                            raise
                        logger.info(f"[MCP] Server '{name}': Streamable HTTP failed, trying SSE")
            if session is None:
                if sse_client is None:
                    raise RuntimeError(f"Server '{name}' needs SSE, which this mcp package does not have.")
                session = await self._open_remote(sse_client(url, headers=headers))

            self.sessions[name] = session
            await self._discover_capabilities(name, session)

    async def _open_remote(self, transport_cm) -> "ClientSession":
        """A session over `transport_cm`, kept open on this skill's exit stack;
        a transport that fails to initialize is closed again before the error
        leaves, so a fallback attempt starts clean."""
        attempt = AsyncExitStack()
        await attempt.__aenter__()
        try:
            streams = await attempt.enter_async_context(transport_cm)
            read, write = streams[0], streams[1]
            session = await attempt.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await attempt.aclose()
            raise
        await self.exit_stack.enter_async_context(attempt)
        return session

    async def _discover_capabilities(self, server_name: str, session: ClientSession):
        """Discover tools and resources from server"""
        logger.debug(f"[MCP] Discovering capabilities for server '{server_name}'")
        # List tools
        result = await session.list_tools()
        logger.info(f"[MCP] Server '{server_name}' has {len(result.tools)} tools")
        for tool in filter_discovered_tools(list(result.tools), self._server_configs.get(server_name)):
            # `<server>__<tool>`, always: the one rule both SDKs apply (`config.py`).
            tool_name = qualified_tool_name(server_name, tool.name)
            
            self.tools_registry[tool_name] = {
                "server": server_name,
                "original_name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            
            # Create dynamic tool method
            await self._register_dynamic_tool(tool_name, tool)

    def _tool_exists(self, name: str) -> bool:
        """Check if tool name is already registered"""
        # Check against existing agent tools + local registry
        return name in self.tools_registry

    async def _register_dynamic_tool(self, tool_name: str, tool_def: Any):
        """Register a dynamic tool with the agent"""
        
        async def dynamic_tool_func(**kwargs):
            """Dynamic MCP tool wrapper"""
            info = self.tools_registry.get(tool_name)
            if not info:
                return f"Error: Tool {tool_name} not found."
            
            session = self.sessions.get(info["server"])
            if not session:
                return f"Error: Server {info['server']} not connected."
                
            try:
                # Validate inputs against schema (basic check)
                # Note: mcp sdk might handle this, but explicit check helps debugging
                
                result = await session.call_tool(info["original_name"], arguments=kwargs)
                
                # Format output
                output = []
                for content in result.content:
                    if content.type == "text":
                        output.append(content.text)
                    elif content.type == "image":
                        output.append(f"[Image: {content.mimeType}]")
                    elif content.type == "resource":
                        output.append(f"[Resource: {content.resource.uri}]")
                        
                return "\n".join(output)
            except Exception as e:
                return f"Error executing tool {tool_name}: {e}"

        # Set metadata
        dynamic_tool_func.__name__ = tool_name
        dynamic_tool_func.__doc__ = tool_def.description
        
        # 1. Construct OpenAI tool schema from MCP schema
        # MCP inputSchema is typically JSON Schema
        mcp_schema = tool_def.inputSchema or {"type": "object", "properties": {}}
        
        tool_schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_def.description or f"Tool: {tool_name}",
                "parameters": mcp_schema
            }
        }
        
        # 2. Attach metadata expected by BaseAgent.register_tool
        dynamic_tool_func._webagents_is_tool = True
        dynamic_tool_func._webagents_tool_definition = tool_schema
        dynamic_tool_func._tool_name = tool_name
        dynamic_tool_func._tool_description = tool_def.description or f"Tool: {tool_name}"
        dynamic_tool_func._tool_scope = "all" # Default scope
        
        # 3. Add docstring details for fallback/reference
        args_desc = []
        if mcp_schema and "properties" in mcp_schema:
            for arg_name, arg_info in mcp_schema["properties"].items():
                arg_type = arg_info.get("type", "any")
                arg_desc = arg_info.get("description", "")
                args_desc.append(f"  - {arg_name} ({arg_type}): {arg_desc}")
        
        if args_desc:
            dynamic_tool_func.__doc__ += "\n\nArguments:\n" + "\n".join(args_desc)
            
        self.register_tool(dynamic_tool_func)

    def _get_server_completions(self) -> Dict[str, List[str]]:
        """Return server names for autocomplete."""
        return {"server_name": list(self.sessions.keys())}
    
    def _get_tool_completions(self) -> Dict[str, List[str]]:
        """Return tool names for autocomplete."""
        return {"tool_name": list(self.tools_registry.keys())}
    
    def _get_subcommand_completions(self) -> Dict[str, List[str]]:
        """Return subcommands for autocomplete."""
        return {"subcommand": ["servers", "tools", "tool", "call", "resources", "prompts"]}
    
    @command("/mcp", description="MCP commands - servers, tools, call, resources, prompts", scope="all",
             completions=lambda self: self._get_subcommand_completions())
    async def mcp_help(self, subcommand: str = None) -> Dict[str, Any]:
        """Show MCP help or subcommand info.
        
        Args:
            subcommand: Optional subcommand (servers, tools, call, resources, prompts)
            
        Returns:
            Help info
        """
        subcommands = {
            "servers": {"description": "List connected MCP servers", "usage": "/mcp servers"},
            "tools": {"description": "List all MCP tools", "usage": "/mcp tools"},
            "tool": {"description": "Get details about an MCP tool", "usage": "/mcp tool <name>"},
            "call": {"description": "Call an MCP tool directly", "usage": "/mcp call <name> [args_json]"},
            "resources": {"description": "List MCP resources", "usage": "/mcp resources [server]"},
            "prompts": {"description": "List MCP prompts", "usage": "/mcp prompts [server]"},
        }
        
        if not subcommand:
            lines = ["[bold]/mcp[/bold] - Model Context Protocol integration", ""]
            for name, info in subcommands.items():
                lines.append(f"  [cyan]/mcp {name}[/cyan] - {info['description']}")
            lines.append("")
            lines.append(f"Connected: {len(self.sessions)} servers, {len(self.tools_registry)} tools")
            
            return {
                "command": "/mcp",
                "description": "Model Context Protocol integration",
                "subcommands": subcommands,
                "display": "\n".join(lines),
            }
        
        if subcommand in subcommands:
            info = subcommands[subcommand]
            return {
                "command": f"/mcp {subcommand}",
                **info,
                "display": f"[cyan]{info['usage']}[/cyan]\n{info['description']}",
            }
        
        return {
            "error": f"Unknown subcommand: {subcommand}",
            "display": f"[red]Error:[/red] Unknown subcommand: {subcommand}. Available: {', '.join(subcommands.keys())}",
        }
    
    @command("/mcp/servers", description="List connected MCP servers", scope="all")
    async def list_servers(self) -> Dict[str, Any]:
        """List connected MCP servers and their status.
        
        Returns:
            Dict with server list and their tools.
        """
        logger.info(f"[MCP] /mcp/servers called: initialized={self._initialized}, sessions={list(self.sessions.keys())}, tools={len(self.tools_registry)}")
        if not self.sessions:
            return {
                "servers": [],
                "message": "No MCP servers connected.",
                "display": "[yellow]No MCP servers connected.[/yellow]",
            }
        
        servers = []
        lines = ["[bold]MCP Servers:[/bold]"]
        for name in self.sessions:
            server_tools = [
                t_name for t_name, t_info in self.tools_registry.items() 
                if t_info["server"] == name
            ]
            servers.append({
                "name": name,
                "tools": server_tools,
                "tool_count": len(server_tools)
            })
            lines.append(f"  [cyan]{name}[/cyan] ({len(server_tools)} tools)")
        
        return {
            "servers": servers,
            "total": len(servers),
            "display": "\n".join(lines),
        }
    
    @command("/mcp/tools", description="List all MCP tools", scope="all")
    async def list_tools(self) -> Dict[str, Any]:
        """List all tools from connected MCP servers.
        
        Returns:
            Dict with tool list and their details.
        """
        if not self.tools_registry:
            return {
                "tools": [],
                "message": "No tools available.",
                "display": "[yellow]No MCP tools available.[/yellow] Run /mcp servers to check connections.",
            }
        
        tools = []
        lines = ["[bold]MCP Tools:[/bold]"]
        for name, info in self.tools_registry.items():
            tools.append({
                "name": name,
                "server": info["server"],
                "description": info.get("description", ""),
                "original_name": info.get("original_name", name)
            })
            desc = info.get("description", "")[:50]
            lines.append(f"  [cyan]{name}[/cyan] [{info['server']}] {desc}")
        
        return {
            "tools": tools,
            "total": len(tools),
            "display": "\n".join(lines),
        }
    
    @command("/mcp/tool", description="Get details about an MCP tool", scope="all",
             completions=lambda self: self._get_tool_completions())
    async def get_tool_info(self, tool_name: str = "") -> Dict[str, Any]:
        """Get detailed information about an MCP tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool details including input schema
        """
        if not tool_name:
            # No tool specified - show usage
            available = list(self.tools_registry.keys())[:5]
            more = len(self.tools_registry) - 5 if len(self.tools_registry) > 5 else 0
            tools_hint = ", ".join(available) + (f", ... (+{more} more)" if more else "")
            return {
                "error": "No tool name specified",
                "display": f"[yellow]Usage:[/yellow] /mcp tool <name>\n[dim]Available: {tools_hint}[/dim]",
            }
        if tool_name not in self.tools_registry:
            return {
                "error": f"Tool not found: {tool_name}",
                "display": f"[red]Error:[/red] Tool not found: {tool_name}",
            }
        
        info = self.tools_registry[tool_name]
        
        # Build display
        lines = [f"[bold]{tool_name}[/bold]"]
        lines.append(f"  Server: {info['server']}")
        lines.append(f"  Description: {info.get('description', 'N/A')}")
        schema = info.get("input_schema", {})
        if schema.get("properties"):
            lines.append("  Parameters:")
            for param, details in schema["properties"].items():
                ptype = details.get("type", "any")
                pdesc = details.get("description", "")
                lines.append(f"    [dim]{param}[/dim] ({ptype}): {pdesc}")
        
        return {
            "name": tool_name,
            "server": info["server"],
            "original_name": info.get("original_name", tool_name),
            "description": info.get("description", ""),
            "input_schema": info.get("input_schema", {}),
            "display": "\n".join(lines),
        }
    
    @command("/mcp/call", description="Call an MCP tool directly", scope="owner",
             completions=lambda self: self._get_tool_completions())
    async def call_tool(self, tool_name: str, args: str = "") -> Dict[str, Any]:
        """Call an MCP tool with arguments.
        
        Args:
            tool_name: Name of the tool to call
            args: JSON string of arguments
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools_registry:
            return {
                "error": f"Tool not found: {tool_name}",
                "display": f"[red]Error:[/red] Tool not found: {tool_name}",
            }
        
        info = self.tools_registry[tool_name]
        session = self.sessions.get(info["server"])
        if not session:
            return {
                "error": f"Server not connected: {info['server']}",
                "display": f"[red]Error:[/red] Server not connected: {info['server']}",
            }
        
        try:
            # Parse args
            import json
            kwargs = json.loads(args) if args else {}
            
            result = await session.call_tool(info["original_name"], arguments=kwargs)
            
            # Format output
            output = []
            for content in result.content:
                if content.type == "text":
                    output.append(content.text)
                elif content.type == "image":
                    output.append(f"[Image: {content.mimeType}]")
                elif content.type == "resource":
                    output.append(f"[Resource: {content.resource.uri}]")
            
            result_text = "\n".join(output)
            return {
                "result": result_text,
                "status": "success",
                "display": f"[green]✓[/green] {tool_name}\n{result_text}",
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed",
                "display": f"[red]Error:[/red] {e}",
            }
    
    @command("/mcp/resources", description="List MCP resources", scope="all",
             completions=lambda self: self._get_server_completions())
    async def list_resources(self, server_name: str = "") -> Dict[str, Any]:
        """List resources from MCP servers.
        
        Args:
            server_name: Optional server name to filter (lists all if empty)
            
        Returns:
            List of available resources
        """
        resources = []
        
        servers_to_check = [server_name] if server_name else list(self.sessions.keys())
        
        for name in servers_to_check:
            session = self.sessions.get(name)
            if not session:
                continue
            
            try:
                result = await session.list_resources()
                for resource in result.resources:
                    resources.append({
                        "server": name,
                        "uri": str(resource.uri),
                        "name": resource.name or str(resource.uri),
                        "description": resource.description or "",
                        "mime_type": resource.mimeType or ""
                    })
            except Exception as e:
                resources.append({"server": name, "error": str(e)})
        
        # Build display
        if not resources:
            display = "[yellow]No MCP resources available.[/yellow]"
        else:
            lines = ["[bold]MCP Resources:[/bold]"]
            for r in resources:
                if "error" in r:
                    lines.append(f"  [red]{r['server']}[/red]: {r['error']}")
                else:
                    lines.append(f"  [cyan]{r['name']}[/cyan] [{r['server']}]")
            display = "\n".join(lines)
        
        return {"resources": resources, "total": len(resources), "display": display}
    
    @command("/mcp/prompts", description="List MCP prompts", scope="all",
             completions=lambda self: self._get_server_completions())
    async def list_prompts(self, server_name: str = "") -> Dict[str, Any]:
        """List prompts from MCP servers.
        
        Args:
            server_name: Optional server name to filter (lists all if empty)
            
        Returns:
            List of available prompts
        """
        prompts = []
        
        servers_to_check = [server_name] if server_name else list(self.sessions.keys())
        
        for name in servers_to_check:
            session = self.sessions.get(name)
            if not session:
                continue
            
            try:
                result = await session.list_prompts()
                for prompt in result.prompts:
                    prompts.append({
                        "server": name,
                        "name": prompt.name,
                        "description": prompt.description or "",
                        "arguments": [
                            {"name": arg.name, "description": arg.description or "", "required": arg.required}
                            for arg in (prompt.arguments or [])
                        ]
                    })
            except Exception as e:
                prompts.append({"server": name, "error": str(e)})
        
        # Build display
        if not prompts:
            display = "[yellow]No MCP prompts available.[/yellow]"
        else:
            lines = ["[bold]MCP Prompts:[/bold]"]
            for p in prompts:
                if "error" in p:
                    lines.append(f"  [red]{p['server']}[/red]: {p['error']}")
                else:
                    lines.append(f"  [cyan]{p['name']}[/cyan] [{p['server']}] {p.get('description', '')[:40]}")
            display = "\n".join(lines)
        
        return {"prompts": prompts, "total": len(prompts), "display": display}
    
    # The TypeScript tool's sentence (`skills/mcp/skill.ts` `listServers`),
    # pinned by the shared fixture `mcp_tool/config_shapes.json` (`tools`):
    # the model reads the same description under either SDK (2026-09-27).
    @tool(description="List connected MCP servers and their available tools, resources, and prompts.")
    async def list_mcp_servers(self) -> str:
        """List connected MCP servers and their status.

        Returns:
            List of servers and their tools.
        """
        if not self.sessions:
            return "No MCP servers connected."
            
        output = ["Connected MCP Servers:"]
        for name, session in self.sessions.items():
            output.append(f"\n📡 {name}")
            
            # List tools for this server
            server_tools = [
                t_name for t_name, t_info in self.tools_registry.items() 
                if t_info["server"] == name
            ]
            if server_tools:
                output.append(f"  Tools: {', '.join(server_tools)}")
            else:
                output.append("  Tools: (none)")
                
        return "\n".join(output)

    async def cleanup(self):
        """Close the connections: asked of the task that holds them (see
        `__init__`), and awaited, so the servers are gone when this returns
        whichever task called it."""
        runner, closing = self._runner, self._closing
        self._runner, self._closing = None, None
        if runner is not None and not runner.done():
            assert closing is not None
            closing.set()
            try:
                await runner
            except BaseException:  # noqa: BLE001 - said by the runner; a close that failed must not fail the caller
                pass
        else:
            await self._close_stack()
        self.sessions.clear()
        self.connect_errors.clear()
        self._resolution = None
        self._initialized = False
