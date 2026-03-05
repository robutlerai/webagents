"""
IntegrationsSkill - MCP Bridge Integration for External Providers

Enables agents to use external integrations (Google, X, etc.) via the Robutler MCP bridge.
Dynamically loads and registers tools from enabled providers on agent startup.

Features:
- Queries MCP bridge for enabled integrations
- Fetches filtered tool lists per provider
- Registers tools dynamically with the agent
- Proxies tool calls through the MCP bridge
- Handles errors gracefully (provider unavailable, rate limited, needs reauth)
"""

import os
import json
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, hook
from webagents.utils.logging import get_logger, log_skill_event


@dataclass
class IntegrationTool:
    """Metadata for an integration tool"""
    name: str
    description: str
    provider: str
    input_schema: Dict[str, Any]


class IntegrationsSkill(Skill):
    """
    Integrations skill for accessing external providers via MCP bridge
    
    Configuration:
        enabled_providers: List of provider IDs to enable (e.g., ["google", "x"])
        robutler_api_url: Base URL for Robutler API (defaults to env or localhost:3000)
        robutler_api_key: API key for authentication (defaults to agent.api_key or env)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        
        self.config = config or {}
        self.enabled_providers: List[str] = self.config.get('enabled_providers', [])
        
        # Robutler API base URL
        self.robutler_api_url = (
            os.getenv('ROBUTLER_API_URL') or
            os.getenv('ROBUTLER_INTERNAL_API_URL') or
            os.getenv('ROBUTLER_API_URL') or
            self.config.get('robutler_api_url') or
            'http://localhost:3000'
        ).rstrip('/')
        
        # Auth token (resolved in initialize)
        self._auth_token: Optional[str] = self.config.get('robutler_api_key')
        
        # Track loaded tools per provider
        self._loaded_tools: Dict[str, List[IntegrationTool]] = {}
        self._tools_loaded: bool = False
        
        # HTTP client (initialized in initialize)
        self.http_client: Optional[Any] = None
        self.logger = None
    
    async def initialize(self, agent) -> None:
        """Initialize Integrations skill with agent context"""
        self.agent = agent
        self.logger = get_logger('skill.webagents.integrations', agent.name)
        
        # Resolve auth token: config > agent.api_key > env
        if not self._auth_token:
            if hasattr(self.agent, 'api_key') and self.agent.api_key:
                self._auth_token = self.agent.api_key
            elif os.getenv('WEBAGENTS_API_KEY'):
                self._auth_token = os.getenv('WEBAGENTS_API_KEY')
            elif os.getenv('SERVICE_TOKEN'):
                self._auth_token = os.getenv('SERVICE_TOKEN')
        
        if HTTPX_AVAILABLE:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=50)
            )
            self.logger.info("Integrations HTTP client initialized")
        else:
            self.logger.warning("httpx not available - integrations functionality will be limited")
        
        # Tools will be loaded lazily when first needed (when we have config ID from request)
        
        log_skill_event(self.agent.name, 'integrations', 'initialized', {
            'enabled_providers': self.enabled_providers,
            'httpx_available': HTTPX_AVAILABLE,
            'has_auth_token': bool(self._auth_token),
        })
    
    def _get_agent_config_id(self) -> Optional[str]:
        """Get agent config ID from request context or agent config"""
        # Try to get from request context first
        try:
            from webagents.server.context.context_vars import get_context
            ctx = get_context()
            if ctx and ctx.request:
                request_headers = getattr(ctx.request, 'headers', {})
                if hasattr(request_headers, 'get'):
                    config_id = (
                        request_headers.get('X-Agent-Config-Id') or
                        request_headers.get('x-agent-config-id')
                    )
                    if config_id:
                        return config_id
        except Exception:
            pass
        
        # Fallback: try agent config
        if self.agent and hasattr(self.agent, 'config'):
            agent_config = self.agent.config or {}
            if isinstance(agent_config, dict):
                return agent_config.get('config_id') or agent_config.get('id')
        
        return None
    
    async def _ensure_tools_loaded(self) -> None:
        """Ensure tools are loaded (called lazily when needed)"""
        if self._tools_loaded:
            return
        
        config_id = self._get_agent_config_id()
        if not config_id:
            self.logger.debug("No config ID available yet, deferring tool loading")
            return
        
        await self._load_provider_tools(config_id)
    
    async def _load_provider_tools(self, config_id: str) -> None:
        """Load tools from all enabled providers"""
        if not HTTPX_AVAILABLE or not self.http_client:
            self.logger.warning("Cannot load provider tools: HTTP client not available")
            return
        
        if not self._auth_token:
            self.logger.warning("Cannot load provider tools: No auth token available")
            return
        
        loaded_count = 0
        
        for provider_id in self.enabled_providers:
            try:
                tools = await self._fetch_provider_tools(provider_id, config_id)
                if tools:
                    self._loaded_tools[provider_id] = tools
                    # Register each tool with the agent
                    for tool_meta in tools:
                        self._register_integration_tool(tool_meta, provider_id)
                        loaded_count += 1
                    self.logger.info(f"✅ Loaded {len(tools)} tools from provider '{provider_id}'")
                else:
                    self.logger.warning(f"⚠️ No tools available for provider '{provider_id}'")
            except Exception as e:
                self.logger.error(f"❌ Failed to load tools from provider '{provider_id}': {e}")
        
        if loaded_count > 0:
            self._tools_loaded = True
            self.logger.info(f"🎉 Successfully loaded {loaded_count} integration tools")
    
    async def _fetch_provider_tools(self, provider_id: str, config_id: str) -> List[IntegrationTool]:
        """Fetch tool list from MCP bridge for a specific provider"""
        url = f"{self.robutler_api_url}/api/integrations/mcp/{provider_id}"
        
        # JSON-RPC request for tools/list
        mcp_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": str(uuid.uuid4())
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._auth_token}",
            "X-Agent-Config-Id": config_id,
        }
        
        try:
            response = await self.http_client.post(
                url,
                json=mcp_request,
                headers=headers,
                timeout=10.0
            )
            
            if response.status_code == 401:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error = error_data.get('error', {})
                if error.get('data', {}).get('needsReauth'):
                    raise Exception(f"Provider '{provider_id}' needs re-authorization")
                raise Exception(f"Authentication failed for provider '{provider_id}'")
            
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After', '60')
                raise Exception(f"Rate limit exceeded for provider '{provider_id}'. Retry after {retry_after}s")
            
            if response.status_code == 400:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error = error_data.get('error', {})
                error_msg = error.get('message', f"Bad request for provider '{provider_id}'")
                # Check if integration not enabled
                if 'not enabled' in error_msg.lower():
                    raise Exception(f"Integration '{provider_id}' is not enabled for this agent")
                raise Exception(error_msg)
            
            response.raise_for_status()
            
            # Parse response (MCP bridge returns { tools: [...] } directly, not JSON-RPC wrapped)
            data = response.json()
            if 'error' in data:
                error = data['error']
                raise Exception(f"MCP error: {error.get('message', 'Unknown error')}")
            
            # Extract tools from response (bridge returns { tools: [...] } directly)
            tools_data = data.get('tools', [])
            
            tools = []
            for tool_data in tools_data:
                tool = IntegrationTool(
                    name=tool_data.get('name', ''),
                    description=tool_data.get('description', ''),
                    provider=provider_id,
                    input_schema=tool_data.get('inputSchema', {})
                )
                tools.append(tool)
            
            return tools
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                error_data = e.response.json() if e.response.headers.get('content-type', '').startswith('application/json') else {}
                error = error_data.get('error', {})
                if error.get('data', {}).get('needsReauth'):
                    raise Exception(f"Provider '{provider_id}' needs re-authorization")
                raise Exception(f"Authentication failed for provider '{provider_id}'")
            raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.TimeoutException:
            raise Exception(f"Timeout while fetching tools from provider '{provider_id}'")
        except Exception as e:
            if isinstance(e, Exception) and str(e):
                raise
            raise Exception(f"Failed to fetch tools from provider '{provider_id}': {str(e)}")
    
    def _register_integration_tool(self, tool_meta: IntegrationTool, provider_id: str) -> None:
        """Register an integration tool with the agent"""
        # Store original tool name for MCP calls
        original_tool_name = tool_meta.name
        
        # Create a wrapper function for the tool
        async def integration_tool_wrapper(**kwargs) -> str:
            """Dynamic integration tool wrapper"""
            return await self._call_integration_tool(tool_meta.provider, original_tool_name, kwargs)
        
        # Set tool metadata
        tool_name = f"{provider_id}_{tool_meta.name}"
        integration_tool_wrapper.__name__ = tool_name
        integration_tool_wrapper._webagents_is_tool = True
        integration_tool_wrapper._tool_description = (
            f"{tool_meta.description} (via {provider_id} integration)"
        )
        integration_tool_wrapper._tool_scope = self.scope
        
        # Build tool definition from input schema
        input_schema = tool_meta.input_schema or {}
        properties = input_schema.get('properties', {})
        required = input_schema.get('required', [])
        
        tool_definition = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_meta.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required if isinstance(required, list) else []
                }
            }
        }
        integration_tool_wrapper._webagents_tool_definition = tool_definition
        
        # Register with agent
        self.register_tool(integration_tool_wrapper, scope=self.scope)
        self.logger.debug(f"🔧 Registered integration tool: {tool_name}")
    
    async def _call_integration_tool(self, provider_id: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Proxy a tool call through the MCP bridge"""
        # Ensure tools are loaded before calling (lazy loading)
        await self._ensure_tools_loaded()
        
        if not HTTPX_AVAILABLE or not self.http_client:
            return f"❌ HTTP client not available for integration tool '{tool_name}'"
        
        if not self._auth_token:
            return f"❌ Authentication token not available for integration tool '{tool_name}'"
        
        config_id = self._get_agent_config_id()
        if not config_id:
            return f"❌ Agent config ID not available for integration tool '{tool_name}'"
        
        url = f"{self.robutler_api_url}/api/integrations/mcp/{provider_id}"
        
        # JSON-RPC request for tools/call
        mcp_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": str(uuid.uuid4())
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._auth_token}",
            "X-Agent-Config-Id": config_id,
        }
        
        try:
            response = await self.http_client.post(
                url,
                json=mcp_request,
                headers=headers,
                timeout=60.0  # Longer timeout for tool execution
            )
            
            if response.status_code == 401:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error = error_data.get('error', {})
                if error.get('data', {}).get('needsReauth'):
                    return f"❌ Provider '{provider_id}' needs re-authorization. Please reconnect the integration."
                return f"❌ Authentication failed for provider '{provider_id}'"
            
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After', '60')
                return f"❌ Rate limit exceeded for provider '{provider_id}'. Please retry after {retry_after} seconds."
            
            if response.status_code == 400:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error = error_data.get('error', {})
                error_msg = error.get('message', f"Bad request for tool '{tool_name}'")
                return f"❌ {error_msg}"
            
            response.raise_for_status()
            
            # Parse response (MCP bridge may return JSON-RPC or direct format)
            data = response.json()
            if 'error' in data:
                error = data['error']
                return f"❌ MCP error: {error.get('message', 'Unknown error')}"
            
            # Extract result content (handle both JSON-RPC and direct formats)
            if 'result' in data:
                result = data['result']
            else:
                result = data
            
            content = result.get('content', [])
            
            # Format response
            if isinstance(content, list) and len(content) > 0:
                # Extract text from content array
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                        elif 'text' in item:
                            text_parts.append(str(item['text']))
                    elif isinstance(item, str):
                        text_parts.append(item)
                
                if text_parts:
                    return '\n'.join(text_parts)
            
            # Fallback: return raw result if no content format
            if isinstance(result, dict) and result:
                return json.dumps(result, indent=2)
            elif result:
                return str(result)
            
            return "✅ Tool executed successfully (no response content)"
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                error_data = e.response.json() if e.response.headers.get('content-type', '').startswith('application/json') else {}
                error = error_data.get('error', {})
                if error.get('data', {}).get('needsReauth'):
                    return f"❌ Provider '{provider_id}' needs re-authorization. Please reconnect the integration."
                return f"❌ Authentication failed for provider '{provider_id}'"
            return f"❌ HTTP {e.response.status_code}: {e.response.text[:200]}"
        except httpx.TimeoutException:
            return f"❌ Timeout while executing tool '{tool_name}' from provider '{provider_id}'"
        except Exception as e:
            return f"❌ Error executing tool '{tool_name}': {str(e)}"
    
    async def cleanup(self):
        """Cleanup Integrations resources"""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
