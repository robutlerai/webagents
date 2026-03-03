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

logger = logging.getLogger("webagents.skills.mcp")

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("mcp package not installed. MCP skill disabled.")

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
        
        logger.debug(f"[MCP] __init__ for agent={self.agent_name}, config keys={list(self.config.keys())}")
        
        if not MCP_AVAILABLE:
            logger.warning("mcp package not installed. MCP skill disabled.")

    async def initialize(self, agent):
        """Initialize and connect to configured servers"""
        logger.info(f"[MCP] initialize() called for agent={self.agent_name}, already_initialized={self._initialized}")
        
        if self._initialized:
            logger.debug(f"[MCP] Already initialized, skipping. sessions={list(self.sessions.keys())}")
            return
        
        await super().initialize(agent)
        
        if not MCP_AVAILABLE:
            logger.warning("[MCP] MCP not available, skipping initialization")
            return
            
        # Load configuration
        mcp_config = self._load_mcp_config()
        logger.debug(f"[MCP] Loaded config: {mcp_config}")
        
        if not mcp_config:
            logger.warning(f"[MCP] No MCP config found for agent={self.agent_name}")
            return
        
        servers = mcp_config.get("mcpServers", {})
        logger.info(f"[MCP] Connecting to {len(servers)} server(s): {list(servers.keys())}")
            
        # Connect to servers
        for name, server_config in servers.items():
            try:
                await self._connect_server(name, server_config)
                logger.info(f"[MCP] Connected to server: {name}")
            except Exception as e:
                logger.error(f"[MCP] Failed to connect to server {name}: {e}", exc_info=True)
        
        self._initialized = True
        logger.info(f"[MCP] Initialization complete. tools={len(self.tools_registry)}, sessions={list(self.sessions.keys())}")

    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP configuration from agent metadata or local file"""
        logger.debug(f"[MCP] _load_mcp_config: self.config = {self.config}")
        
        # 1. Check agent metadata (passed via config) - Preferred
        if "mcp" in self.config:
            logger.debug(f"[MCP] Found 'mcp' key in config: {self.config['mcp']}")
            # Check if it's already wrapped in mcpServers or not
            if "mcpServers" in self.config["mcp"]:
                logger.debug("[MCP] Using config['mcp'] directly (has mcpServers)")
                return self.config["mcp"]
            # Assume the config under "mcp" IS the server map
            logger.debug("[MCP] Wrapping config['mcp'] in mcpServers")
            return {"mcpServers": self.config["mcp"]}
        
        # 2. Check if config keys look like server definitions (from agent yaml)
        # When loaded from agent metadata like `mcp: {sqlite: {command: ...}}`,
        # the config is passed directly as {"sqlite": {"command": ...}, "agent_name": ...}
        servers = {}
        for key, value in self.config.items():
            if key in ("agent_name", "agent_path", "base_dir"):
                continue  # Skip injected metadata
            if isinstance(value, dict) and ("command" in value or "url" in value or "httpUrl" in value):
                servers[key] = value
                logger.debug(f"[MCP] Found server definition in config key '{key}'")
        
        if servers:
            logger.debug(f"[MCP] Built server map from config keys: {list(servers.keys())}")
            return {"mcpServers": servers}
        
        # 3. Check for mcp.json in agent directory (Fallback)
        if self.agent_path:
            # agent_path should be the directory, not the file
            agent_dir = Path(self.agent_path)
            if agent_dir.is_file():
                agent_dir = agent_dir.parent
            config_path = agent_dir / "mcp.json"
            logger.debug(f"[MCP] Checking for mcp.json at: {config_path}")
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text())
                    logger.debug(f"[MCP] Loaded mcp.json: {config}")
                    return config
                except Exception as e:
                    logger.error(f"[MCP] Error loading mcp.json: {e}")
        
        logger.warning(f"[MCP] No MCP configuration found")
        return {}

    async def _connect_server(self, name: str, config: Dict[str, Any]):
        """Connect to a single MCP server"""
        logger.debug(f"[MCP] _connect_server: name={name}, config={config}")
        
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
        
        if "command" in config:
            command = config["command"]
            args = config.get("args", [])
            env = {**os.environ, **config.get("env", {})}
            cwd = config.get("cwd") or default_cwd

            # Auto-detect Docker Sandbox
            use_sandbox = config.get("sandbox", False)
            sandbox_skill = None
            
            if self.agent:
                for skill in self.agent.skills.values():
                     if skill.__class__.__name__ == "SandboxSkill":
                         sandbox_skill = skill
                         break
            
            # If sandbox explicitly requested OR Sandbox skill present (implicit mode), use it
            if use_sandbox or sandbox_skill:
                if not sandbox_skill and use_sandbox:
                     print(f"Warning: 'sandbox: true' requested for MCP server '{name}' but SandboxSkill not found. Running locally.")
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
                    
                    # Set env vars
                    for k, v in config.get("env", {}).items():
                        new_args.extend(["-e", f"{k}={v}"])
                        
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
            
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            logger.info(f"[MCP] Session initialized for server '{name}'")
            
            self.sessions[name] = session
            await self._discover_capabilities(name, session)
            logger.info(f"[MCP] Capabilities discovered for server '{name}': tools={len([t for t, info in self.tools_registry.items() if info.get('server') == name])}")
            
        elif "url" in config or "httpUrl" in config:
            # SSE transport
            url = config.get("url") or config.get("httpUrl")
            headers = config.get("headers", {})
            
            # TODO: Handle OAuth if needed (basic headers supported for now)
            
            read, write = await self.exit_stack.enter_async_context(sse_client(url, headers=headers))
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            
            self.sessions[name] = session
            await self._discover_capabilities(name, session)

    async def _discover_capabilities(self, server_name: str, session: ClientSession):
        """Discover tools and resources from server"""
        logger.debug(f"[MCP] Discovering capabilities for server '{server_name}'")
        # List tools
        result = await session.list_tools()
        logger.info(f"[MCP] Server '{server_name}' has {len(result.tools)} tools")
        for tool in result.tools:
            # Register tool
            tool_name = f"{server_name}__{tool.name}" if self._tool_exists(tool.name) else tool.name
            
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
    
    @command("/mcp/call", description="Call an MCP tool directly", scope="all",
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
    
    @tool
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
        """Cleanup connections"""
        await self.exit_stack.aclose()
