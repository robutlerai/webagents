"""
NLISkill - Natural Language Interface for Agent-to-Agent Communication

Enables WebAgents agents to communicate with other agents via natural language.
Provides HTTP-based communication with authorization limits and error handling.
"""

import os
import json
import re
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, hook, prompt
from webagents.utils.logging import get_logger, log_skill_event, log_tool_execution, timer


@dataclass
class NLICommunication:
    """Record of an NLI communication"""
    timestamp: datetime
    target_agent_url: str
    message: str
    response: str
    cost_usd: float
    duration_ms: float
    success: bool
    error: Optional[str] = None


@dataclass 
class AgentEndpoint:
    """Agent endpoint configuration"""
    url: str
    name: Optional[str] = None
    description: Optional[str] = None
    capabilities: List[str] = None
    last_contact: Optional[datetime] = None
    success_rate: float = 1.0
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


class NLISkill(Skill):
    """
    Natural Language Interface skill for agent-to-agent communication
    
    Features:
    - HTTP-based communication with other WebAgents agents
    - Authorization limits and cost tracking
    - Communication history and success rate tracking
    - Automatic timeout and retry handling
    - Agent endpoint discovery and management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        
        # Configuration
        self.config = config or {}
        self.default_timeout = self.config.get('timeout', 600.0)  # 10 minutes for long-running tasks
        self.max_retries = self.config.get('max_retries', 2)
        self.default_authorization = self.config.get('default_authorization', 0.10)  # $0.10 default
        self.max_authorization = self.config.get('max_authorization', 5.00)  # $5.00 max per call
        
        # Agent communication base URL configuration
        self.agent_base_url = (
            os.getenv('AGENTS_BASE_URL') or
            self.config.get('agent_base_url') or 
            'http://localhost:2224'  # Default for local development (agents server)
        )
        
        # Communication tracking
        self.communication_history: List[NLICommunication] = []
        self.known_agents: Dict[str, AgentEndpoint] = {}
        
        # HTTP client (will be initialized in initialize method)
        self.http_client: Optional[Any] = None
        
        # Logging
        self.logger = None
    
    def get_agent_url(self, agent_name: str) -> str:
        """Convert agent name to full URL for communication"""
        agent_name = agent_name.lstrip('@')  # Remove @ prefix if present
        base_url = self.agent_base_url.rstrip('/')
        return f"{base_url}/agents/{agent_name}"
        
    async def initialize(self, agent) -> None:
        """Initialize NLI skill with agent context"""
        from webagents.utils.logging import get_logger, log_skill_event
        
        self.agent = agent
        self.logger = get_logger('skill.webagents.nli', agent.name)
        
        # Initialize HTTP client for agent communication
        if HTTPX_AVAILABLE:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.default_timeout),
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=50)
            )
            self.logger.info("NLI HTTP client initialized")
        else:
            self.logger.warning("httpx not available - NLI functionality will be limited")
            
        # Load known agents from config
        known_agents_config = self.config.get('known_agents', [])
        for agent_config in known_agents_config:
            self._register_agent_endpoint(
                url=agent_config['url'],
                name=agent_config.get('name'),
                description=agent_config.get('description'),
                capabilities=agent_config.get('capabilities', [])
            )
            
        log_skill_event(self.agent.name, 'nli', 'initialized', {
            'default_timeout': self.default_timeout,
            'max_retries': self.max_retries,
            'default_authorization': self.default_authorization,
            'known_agents': len(self.known_agents),
            'httpx_available': HTTPX_AVAILABLE
        })

    def _extract_agent_name_or_id(self, agent_url: str) -> Dict[str, Optional[str]]:
        """Extract agent UUID or name from a URL like /agents/<name>/chat/completions.
        Returns dict with either {'id': uuid} or {'name': name}."""
        try:
            uuid_re = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
            m = re.search(rf"/agents/({uuid_re})", agent_url)
            if m:
                return {"id": m.group(1), "name": None}
            # Name before /chat/completions
            from urllib.parse import urlparse
            parsed = urlparse(agent_url)
            parts = [p for p in parsed.path.split('/') if p]
            if len(parts) >= 3 and parts[-1] == 'completions' and parts[-2] == 'chat':
                return {"id": None, "name": parts[-3]}
        except Exception:
            pass
        return {"id": None, "name": None}

    async def _mint_owner_assertion(self, target_agent_id: str, acting_user_id: Optional[str]) -> Optional[str]:
        """Mint a short-lived owner assertion (RS256) via Portal API if possible.
        Requires SERVICE_TOKEN and a platform base URL; returns JWT or None on failure.
        """
        if not target_agent_id:
            return None
        portal_base_url = os.getenv('ROBUTLER_INTERNAL_API_URL') or os.getenv('ROBUTLER_API_URL') or 'http://localhost:3000'
        service_token = os.getenv('SERVICE_TOKEN') or os.getenv('WEBAGENTS_API_KEY')
        if not service_token:
            return None
        # Acting user id is strongly recommended for correct scoping
        origin_user_id = acting_user_id
        try:
            if not HTTPX_AVAILABLE:
                return None
            async with httpx.AsyncClient(timeout=10.0) as client:
                payload: Dict[str, Any] = {"agentId": target_agent_id, "ttlSeconds": 180}
                if origin_user_id:
                    payload["originUserId"] = origin_user_id
                resp = await client.post(
                    f"{portal_base_url.rstrip('/')}/api/auth/owner-assertion",
                    headers={
                        "Authorization": f"Bearer {service_token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                assertion = data.get('assertion')
                return assertion if isinstance(assertion, str) else None
        except Exception:
            return None
    
    async def cleanup(self):
        """Cleanup NLI resources"""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
    
    def _register_agent_endpoint(self, url: str, name: str = None, description: str = None, 
                                capabilities: List[str] = None) -> str:
        """Register a known agent endpoint"""
        parsed = urlparse(url)
        endpoint_key = f"{parsed.netloc}{parsed.path}"
        
        self.known_agents[endpoint_key] = AgentEndpoint(
            url=url,
            name=name or f"Agent at {parsed.netloc}",
            description=description,
            capabilities=capabilities or []
        )
        
        return endpoint_key
    
    def _update_agent_stats(self, agent_url: str, success: bool, duration_ms: float):
        """Update agent endpoint statistics"""
        parsed = urlparse(agent_url)
        endpoint_key = f"{parsed.netloc}{parsed.path}"
        
        if endpoint_key in self.known_agents:
            agent = self.known_agents[endpoint_key]
            agent.last_contact = datetime.utcnow()
            
            # Update success rate (exponential moving average)
            alpha = 0.1  # Learning rate
            if success:
                agent.success_rate = agent.success_rate * (1 - alpha) + 1.0 * alpha
            else:
                agent.success_rate = agent.success_rate * (1 - alpha) + 0.0 * alpha
        else:
            # Register new agent endpoint
            self._register_agent_endpoint(agent_url)


    @prompt(priority=20, scope="all")
    def nli_general_prompt(self, context: Any = None) -> str:
        base_url = self.agent_base_url.rstrip('/')
        agent_name = self.agent.name if self.agent and hasattr(self.agent, 'name') else None
        
        prompt = f"Agents: Convert @name to {base_url}/agents/name, use nli_tool. DON'T call yourself.\n"
        if agent_name:
            prompt += f"You are @{agent_name}. NEVER call {base_url}/agents/{agent_name} via NLI!\n"
        
        return prompt
    
    @tool(description="Communicate with other WebAgents agents via natural language", scope="all")
    async def nli_tool(self, 
                          agent_url: str, 
                          message: str, 
                          authorized_amount: float = None,
                          timeout: float = None,
                          context=None) -> str:
        """
        Natural Language Interface to communicate with other WebAgents agents.
        
        Use this tool to send natural language messages to other agents and receive their responses.
        This enables agent-to-agent collaboration, delegation, and information sharing.
        
        Args:
            agent_url: Full URL of the target agent (e.g., "http://localhost:8001/agent-name")
            message: Natural language message to send to the agent
            authorized_amount: Maximum cost authorization in USD (default: $0.10, max: $5.00)
            timeout: Request timeout in seconds (default: 30.0)
            context: Request context for tracking and billing
            
        Returns:
            Response message from the target agent, or error description if failed
            
        Examples:
            - nli_tool("http://localhost:8001/coding-assistant", "Can you help me debug this Python code?")
            - nli_tool("http://localhost:8002/data-analyst", "Please analyze this sales data", authorized_amount=0.50)
        """
        start_time = datetime.utcnow()
        
        # CRITICAL: Prevent self-calling via NLI
        if self.agent and hasattr(self.agent, 'name'):
            agent_name = self.agent.name
            # Check if the URL contains the agent's own name
            if f"/agents/{agent_name}" in agent_url or agent_url.endswith(f"/{agent_name}"):
                error_msg = f"❌ ERROR: Cannot use nli_tool to call yourself (@{agent_name})! You should execute your own tasks directly instead of delegating to yourself via NLI."
                self.logger.error(error_msg)
                return error_msg
        
        # Validate and normalize parameters
        if authorized_amount is None:
            authorized_amount = self.default_authorization
        
        if authorized_amount > self.max_authorization:
            return f"❌ Authorized amount ${authorized_amount:.2f} exceeds maximum allowed ${self.max_authorization:.2f}"
            
        if timeout is None:
            timeout = self.default_timeout
            
        if not HTTPX_AVAILABLE:
            return "❌ HTTP client not available - install httpx to use NLI functionality"
            
        if not self.http_client:
            return "❌ NLI HTTP client not initialized"
            
        # Prepare request payload
        payload = {
            "model": self.agent.name,  # Identify requesting agent
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ],
            "stream": True,  # Enable streaming to work around LiteLLM bug (non-streaming drops image data)
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        # Add authorization headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"WebAgents-NLI/{self.agent.name}",
            "X-Authorization-Amount": str(authorized_amount),
            "X-Origin-Agent": self.agent.name,
        }

        # Include Authorization if available (target agents commonly require it)
        bearer = os.getenv('WEBAGENTS_API_KEY') or os.getenv('SERVICE_TOKEN')
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
            headers["X-API-Key"] = bearer

        # Try to include X-Owner-Assertion for agent-to-agent auth and forward payment token
        try:
            from webagents.server.context.context_vars import get_context as _gc
            ctx = _gc()
            acting_user_id: Optional[str] = getattr(getattr(ctx, 'auth', None), 'user_id', None) if ctx else None
            
            # CRITICAL: Forward payment token from current context to enable agent-to-agent billing
            payment_token = None
            
            self.logger.debug(f"🔐 Token lookup: ctx={ctx is not None}")
            if ctx:
                self.logger.debug(f"🔐 Token lookup: ctx attrs={list(vars(ctx).keys())[:10]}")
            
            # Method 1: Check if payment_token is directly available in context
            if ctx and hasattr(ctx, 'payment_token') and ctx.payment_token:
                payment_token = ctx.payment_token
                self.logger.debug(f"🔐 Found payment token in context.payment_token: {payment_token[:20]}...")
            
            # Method 2: Extract from request headers (most common case)
            elif ctx and hasattr(ctx, 'request') and ctx.request:
                request_headers = getattr(ctx.request, 'headers', {})
                self.logger.debug(f"🔐 Token lookup: request_headers type={type(request_headers)}")
                if hasattr(request_headers, 'get'):
                    self.logger.debug(f"🔐 Token lookup: checking headers for payment token")
                    payment_token = (
                        request_headers.get('X-Payment-Token') or 
                        request_headers.get('x-payment-token') or
                        request_headers.get('payment_token')
                    )
                    if payment_token:
                        self.logger.debug(f"🔐 Found payment token in request headers: {payment_token[:20]}...")
                    else:
                        self.logger.debug(f"🔐 No payment token in request headers")
            
            # Method 3: Check custom_data for payment context (fallback)
            elif ctx and hasattr(ctx, 'custom_data') and ctx.custom_data:
                self.logger.debug(f"🔐 Token lookup: checking custom_data")
                payment_context = ctx.custom_data.get('payment_context')
                if payment_context and hasattr(payment_context, 'payment_token'):
                    payment_token = payment_context.payment_token
                    self.logger.debug(f"🔐 Found payment token in custom_data.payment_context: {payment_token[:20]}...")
            
            # Forward the payment token if found
            if payment_token:
                headers["X-Payment-Token"] = payment_token
                self.logger.info(f"🔐 ✅ Forwarding payment token for agent-to-agent communication: {payment_token[:20]}...")
            else:
                self.logger.warning(f"🔐 ❌ No payment token found to forward - target agent may require payment")
        except Exception:
            acting_user_id = None

        # Resolve target agent id: UUID directly, else by name via portal
        target = self._extract_agent_name_or_id(agent_url)
        target_agent_id = target.get('id')
        if not target_agent_id:
            name_from_path = target.get('name')
            if name_from_path and HTTPX_AVAILABLE:
                portal_base_url = os.getenv('ROBUTLER_INTERNAL_API_URL') or os.getenv('ROBUTLER_API_URL') or 'http://localhost:3000'
                bearer_lookup = os.getenv('WEBAGENTS_API_KEY') or os.getenv('SERVICE_TOKEN')
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        # 1) User/service /agents/:id-or-name (prefer service token if available)
                        if bearer_lookup:
                            rA = await client.get(
                                f"{portal_base_url.rstrip('/')}/api/agents/{name_from_path}",
                                headers={"Authorization": f"Bearer {bearer_lookup}"}
                            )
                            if rA.status_code == 200:
                                dA = rA.json()
                                target_agent_id = dA.get('agent', {}).get('id') or dA.get('id')
                        # 2) Public endpoint
                        if not target_agent_id:
                            rP = await client.get(
                                f"{portal_base_url.rstrip('/')}/api/agents/public/{name_from_path}"
                            )
                            if rP.status_code == 200:
                                dP = rP.json()
                                target_agent_id = dP.get('agent', {}).get('id') or dP.get('id')
                        # 3) By-name endpoint
                        if not target_agent_id:
                            headers = {"Authorization": f"Bearer {bearer_lookup}"} if bearer_lookup else None
                            rB = await client.get(
                                f"{portal_base_url.rstrip('/')}/api/agents/by-name/{name_from_path}",
                                headers=headers
                            )
                            if rB.status_code == 200:
                                dB = rB.json()
                                target_agent_id = dB.get('agent', {}).get('id') or dB.get('id')
                except Exception:
                    pass

        if target_agent_id:
            assertion = await self._mint_owner_assertion(target_agent_id, acting_user_id)
            if assertion:
                headers["X-Owner-Assertion"] = assertion
                headers["x-owner-assertion"] = assertion
        
        # Ensure URL has correct format for chat completions
        # Handle relative URLs by converting them to full URLs
        if agent_url.startswith('/'):
            # Convert relative URL to full URL
            # Use the local agents server base URL
            base_url = os.getenv('AGENTS_BASE_URL', 'http://localhost:2224')
            agent_url = f"{base_url}{agent_url}"
            self.logger.debug(f"Converted relative URL to full URL: {agent_url}")
        
        parsed_url = urlparse(agent_url)
        if not parsed_url.path.endswith('/chat/completions'):
            if parsed_url.path.endswith('/'):
                agent_url = agent_url + 'chat/completions'
            else:
                agent_url = agent_url + '/chat/completions'
        
        communication = None
        try:
            self.logger.info(f"🔗 Sending NLI message to {agent_url}")
            
            # Send request with retry logic
            last_error = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self.http_client.post(
                        agent_url,
                        json=payload,
                        headers=headers,
                        timeout=timeout
                    )
                    
                    # Calculate duration
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    if response.status_code == 200:
                        # Handle streaming response (SSE format)
                        agent_response = ""
                        async for line in response.aiter_lines():
                            if not line or line.strip() == "":
                                continue
                            if line.startswith("data: "):
                                data_str = line[6:]  # Remove "data: " prefix
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk_data = json.loads(data_str)
                                    if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                        choice = chunk_data['choices'][0]
                                        # Check delta (streaming) or message (non-streaming)
                                        for msg_key in ['delta', 'message']:
                                            if msg_key in choice:
                                                msg = choice[msg_key]
                                                if 'content' in msg and msg['content']:
                                                    agent_response += msg['content']
                                except json.JSONDecodeError:
                                    pass  # Skip malformed chunks
                        
                        if not agent_response:
                            # Fallback: maybe it's actually not streaming despite request?
                            agent_response = str(response_data) if 'response_data' in locals() else "No response"
                        
                        # Track successful communication
                        communication = NLICommunication(
                            timestamp=start_time,
                            target_agent_url=agent_url,
                            message=message,
                            response=agent_response,
                            cost_usd=authorized_amount,  # Assume full authorization used for now
                            duration_ms=duration_ms,
                            success=True
                        )
                        
                        self._update_agent_stats(agent_url, True, duration_ms)
                        self.communication_history.append(communication)
                        try:
                            log_tool_execution(self.agent.name, 'nli_tool', int(duration_ms), success=True)
                        except Exception:
                            pass
                        
                        self.logger.info(f"✅ NLI communication successful ({duration_ms:.0f}ms)")
                        
                        return agent_response
                    
                    else:
                        last_error = f"HTTP {response.status_code}: {response.text}"
                        self.logger.warning(f"❌ NLI attempt {attempt + 1} failed: {last_error}")
                        
                        # Don't retry on client errors (4xx)
                        if 400 <= response.status_code < 500:
                            break
                            
                        # Wait before retry (exponential backoff)
                        if attempt < self.max_retries:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                
                except httpx.TimeoutException as e:
                    last_error = f"Request timeout after {timeout}s"
                    self.logger.warning(f"⏱️  NLI attempt {attempt + 1} timed out")
                    
                    if attempt < self.max_retries:
                        await asyncio.sleep(2 ** attempt)
                    
                except Exception as e:
                    last_error = f"Request failed: {str(e)}"
                    self.logger.warning(f"❌ NLI attempt {attempt + 1} error: {last_error}")
                    
                    if attempt < self.max_retries:
                        await asyncio.sleep(2 ** attempt)
            
            # All retries failed
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            communication = NLICommunication(
                timestamp=start_time,
                target_agent_url=agent_url,
                message=message,
                response="",
                cost_usd=0.0,  # No cost if failed
                duration_ms=duration_ms,
                success=False,
                error=last_error
            )
            
            self._update_agent_stats(agent_url, False, duration_ms)
            self.communication_history.append(communication)
            try:
                log_tool_execution(self.agent.name, 'nli_tool', int(duration_ms), success=False)
            except Exception:
                pass
            
            self.logger.error(f"❌ NLI communication failed after {self.max_retries + 1} attempts: {last_error}")
            
            return f"❌ Failed to communicate with agent at {agent_url}: {last_error}"
            
        except Exception as e:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            error_msg = f"Unexpected error: {str(e)}"
            
            communication = NLICommunication(
                timestamp=start_time,
                target_agent_url=agent_url,
                message=message,
                response="",
                cost_usd=0.0,
                duration_ms=duration_ms,
                success=False,
                error=error_msg
            )
            
            self.communication_history.append(communication)
            self.logger.error(f"❌ NLI communication exception: {error_msg}")
            try:
                log_tool_execution(self.agent.name, 'nli_tool', int(duration_ms), success=False)
            except Exception:
                pass
            
            return f"❌ Communication error: {error_msg}"
    
    async def stream_message(
        self,
        agent_url: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        authorized_amount: float = None,
        timeout: float = None
    ):
        """Stream response from remote agent via NLI
        
        Used by AgentHandoffSkill for streaming remote agent responses.
        
        Args:
            agent_url: Full URL of target agent
            messages: Conversation messages to send
            tools: Tools to pass to remote agent
            authorized_amount: Maximum cost authorization in USD
            timeout: Request timeout in seconds
        
        Yields:
            OpenAI-compatible streaming chunks from remote agent
        """
        # Validate parameters
        if authorized_amount is None:
            authorized_amount = self.default_authorization
        
        if authorized_amount > self.max_authorization:
            raise ValueError(f"Authorized amount ${authorized_amount:.2f} exceeds maximum ${self.max_authorization:.2f}")
        
        if timeout is None:
            timeout = self.default_timeout
        
        if not HTTPX_AVAILABLE:
            raise ValueError("HTTP client not available - install httpx")
        
        # Prepare streaming request payload
        payload = {
            "model": self.agent.name if self.agent else "unknown",
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        if tools:
            payload["tools"] = tools
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"WebAgents-NLI/{self.agent.name if self.agent else 'unknown'}",
            "X-Authorization-Amount": str(authorized_amount),
            "X-Origin-Agent": self.agent.name if self.agent else "unknown",
        }
        
        # Include Authorization if available
        bearer = os.getenv('WEBAGENTS_API_KEY') or os.getenv('SERVICE_TOKEN')
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
            headers["X-API-Key"] = bearer
        
        # Forward payment token if available
        try:
            from webagents.server.context.context_vars import get_context as _gc
            ctx = _gc()
            payment_token = None
            
            if ctx and hasattr(ctx, 'payment_token') and ctx.payment_token:
                payment_token = ctx.payment_token
            elif ctx and hasattr(ctx, 'request') and ctx.request:
                request_headers = getattr(ctx.request, 'headers', {})
                if hasattr(request_headers, 'get'):
                    payment_token = (
                        request_headers.get('X-Payment-Token') or 
                        request_headers.get('x-payment-token')
                    )
            
            if payment_token:
                headers["X-Payment-Token"] = payment_token
                self.logger.info(f"🔐 Forwarding payment token for streaming handoff")
        except Exception:
            pass
        
        # Stream from remote agent
        self.logger.info(f"🌊 Starting streaming handoff to: {agent_url}")
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{agent_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    
                    # Parse SSE stream
                    async for line in response.aiter_lines():
                        if not line or line.startswith(':'):
                            continue
                        
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            
                            if data == '[DONE]':
                                self.logger.debug("🌊 Stream completed: [DONE]")
                                break
                            
                            try:
                                chunk = json.loads(data)
                                yield chunk
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Invalid JSON chunk: {data[:100]}")
                                continue
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Remote agent HTTP error: {e.response.status_code}")
            raise
        except httpx.TimeoutException:
            self.logger.error(f"Remote agent timeout after {timeout}s")
            raise
        except Exception as e:
            self.logger.error(f"Remote agent streaming error: {e}")
            raise
    
    # @tool(description="List known agent endpoints and their statistics", scope="owner")
    async def list_known_agents(self, context=None) -> str:
        """
        List all known agent endpoints with their communication statistics.
        
        Returns:
            Formatted list of known agents with success rates and last contact times
        """
        if not self.known_agents:
            return "📝 No known agent endpoints registered"
            
        result = ["📋 Known Agent Endpoints:\n"]
        
        for endpoint_key, agent in self.known_agents.items():
            last_contact = agent.last_contact.strftime("%Y-%m-%d %H:%M:%S") if agent.last_contact else "Never"
            capabilities = ", ".join(agent.capabilities) if agent.capabilities else "Unknown"
            
            result.append(f"🤖 **{agent.name}**")
            result.append(f"   URL: {agent.url}")
            result.append(f"   Success Rate: {agent.success_rate:.1%}")
            result.append(f"   Last Contact: {last_contact}")
            result.append(f"   Capabilities: {capabilities}")
            if agent.description:
                result.append(f"   Description: {agent.description}")
            result.append("")
            
        return "\n".join(result)
    
    # @tool(description="Show recent NLI communication history", scope="owner") 
    async def show_communication_history(self, limit: int = 10, context=None) -> str:
        """
        Show recent NLI communication history with other agents.
        
        Args:
            limit: Maximum number of recent communications to show (default: 10)
            
        Returns:
            Formatted communication history
        """
        if not self.communication_history:
            return "📝 No NLI communications recorded"
            
        recent_communications = self.communication_history[-limit:]
        result = [f"📈 Recent NLI Communications (last {len(recent_communications)}):\n"]
        
        for i, comm in enumerate(reversed(recent_communications), 1):
            status = "✅" if comm.success else "❌"
            timestamp = comm.timestamp.strftime("%H:%M:%S")
            duration = f"{comm.duration_ms:.0f}ms"
            cost = f"${comm.cost_usd:.3f}" if comm.cost_usd > 0 else "Free"
            
            result.append(f"{i}. {status} [{timestamp}] {comm.target_agent_url} ({duration}, {cost})")
            result.append(f"   Message: {comm.message[:60]}{'...' if len(comm.message) > 60 else ''}")
            
            if comm.success:
                response_preview = comm.response[:80].replace('\n', ' ')
                result.append(f"   Response: {response_preview}{'...' if len(comm.response) > 80 else ''}")
            else:
                result.append(f"   Error: {comm.error}")
                
            result.append("")
            
        # Add summary statistics
        total_comms = len(self.communication_history)
        successful_comms = sum(1 for c in self.communication_history if c.success)
        success_rate = successful_comms / total_comms if total_comms > 0 else 0
        total_cost = sum(c.cost_usd for c in self.communication_history)
        
        result.extend([
            f"📊 **Summary Statistics:**",
            f"   Total Communications: {total_comms}",
            f"   Success Rate: {success_rate:.1%}",
            f"   Total Cost: ${total_cost:.3f}"
        ])
        
        return "\n".join(result)
    
    # @tool(description="Register a new agent endpoint for future communication", scope="owner")
    async def register_agent(self, 
                           agent_url: str,
                           name: str = None, 
                           description: str = None,
                           capabilities: str = None,
                           context=None) -> str:
        """
        Register a new agent endpoint for future NLI communications.
        
        Args:
            agent_url: Full URL of the agent endpoint
            name: Friendly name for the agent (optional)
            description: Description of the agent's purpose (optional)  
            capabilities: Comma-separated list of agent capabilities (optional)
            
        Returns:
            Confirmation of agent registration
        """
        try:
            # Parse capabilities string
            caps_list = []
            if capabilities:
                caps_list = [cap.strip() for cap in capabilities.split(',') if cap.strip()]
                
            endpoint_key = self._register_agent_endpoint(
                url=agent_url,
                name=name,
                description=description,
                capabilities=caps_list
            )
            
            agent = self.known_agents[endpoint_key]
            
            self.logger.info(f"📝 Registered agent endpoint: {agent.name} at {agent_url}")
            
            return f"✅ Registered agent: {agent.name}\n" + \
                   f"   URL: {agent.url}\n" + \
                   f"   Capabilities: {', '.join(agent.capabilities) if agent.capabilities else 'None specified'}"
                   
        except Exception as e:
            error_msg = f"Failed to register agent endpoint: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return f"❌ {error_msg}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get NLI communication statistics"""
        total_comms = len(self.communication_history)
        successful_comms = sum(1 for c in self.communication_history if c.success)
        total_cost = sum(c.cost_usd for c in self.communication_history)
        avg_duration = sum(c.duration_ms for c in self.communication_history) / total_comms if total_comms > 0 else 0
        
        return {
            'total_communications': total_comms,
            'successful_communications': successful_comms,
            'success_rate': successful_comms / total_comms if total_comms > 0 else 0,
            'total_cost_usd': total_cost,
            'average_duration_ms': avg_duration,
            'known_agents': len(self.known_agents),
            'httpx_available': HTTPX_AVAILABLE
        } 