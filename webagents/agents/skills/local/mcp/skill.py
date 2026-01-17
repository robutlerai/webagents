"""
Local MCP Skill

Connects to and uses Model Context Protocol (MCP) servers.
Matches Gemini CLI specification for discovery and execution.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from contextlib import AsyncExitStack

from ...base import Skill
from webagents.agents.tools.decorators import tool

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

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
        
        if not MCP_AVAILABLE:
            print("Warning: mcp package not installed. MCP skill disabled.")

    async def initialize(self, agent):
        """Initialize and connect to configured servers"""
        await super().initialize(agent)
        
        if not MCP_AVAILABLE:
            return
            
        # Load configuration
        mcp_config = self._load_mcp_config()
        if not mcp_config:
            return
            
        # Connect to servers
        for name, server_config in mcp_config.get("mcpServers", {}).items():
            try:
                await self._connect_server(name, server_config)
            except Exception as e:
                print(f"Failed to connect to MCP server {name}: {e}")

    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP configuration from agent metadata or local file"""
        # 1. Check agent metadata (passed via config) - Preferred
        if "mcp" in self.config:
            # Check if it's already wrapped in mcpServers or not
            if "mcpServers" in self.config["mcp"]:
                return self.config["mcp"]
            # Assume the config under "mcp" IS the server map
            return {"mcpServers": self.config["mcp"]}
        
        # 2. Check for mcp.json in agent directory (Fallback)
        if self.agent_path:
            agent_dir = Path(self.agent_path).parent
            config_path = agent_dir / "mcp.json"
            if config_path.exists():
                try:
                    return json.loads(config_path.read_text())
                except Exception as e:
                    print(f"Error loading mcp.json: {e}")
            
        return {}

    async def _connect_server(self, name: str, config: Dict[str, Any]):
        """Connect to a single MCP server"""
        
        # Determine default CWD (Agent's directory)
        default_cwd = None
        if self.agent_path:
             default_cwd = str(Path(self.agent_path).parent.resolve())
        elif self.base_dir:
             default_cwd = self.base_dir
        
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
            
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            
            self.sessions[name] = session
            await self._discover_capabilities(name, session)
            
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
        # List tools
        result = await session.list_tools()
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
