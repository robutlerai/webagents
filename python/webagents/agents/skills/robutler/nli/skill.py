"""
NLISkill - Natural Language Interface for Agent-to-Agent Communication

Enables WebAgents agents to communicate with other agents via natural language.
Provides HTTP-based communication with authorization limits and error handling.

The tool accepts @username, agent display name, or full public URL.
Agent resolution is handled server-side via the agent daemon routing.
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

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, hook, prompt
from webagents.utils.logging import get_logger, log_skill_event, log_tool_execution, timer


@dataclass
class NLICommunication:
    """Record of an NLI communication"""
    timestamp: datetime
    target_agent: str
    target_url: str
    message: str
    response: str
    cost_usd: float
    duration_ms: float
    success: bool
    error: Optional[str] = None
    response_signature: Optional[str] = None


class NLISkill(Skill):
    """
    Natural Language Interface skill for agent-to-agent communication
    
    Features:
    - Accepts @username, display name, or public URL as agent identifier
    - Resolves agent names to endpoints via the local agent daemon
    - HTTP-based communication with other WebAgents agents
    - Authorization limits and cost tracking
    - Communication history and success rate tracking
    - Automatic timeout and retry handling
    """
    
    # Internal URL patterns that should be rejected from LLM input
    INTERNAL_URL_PATTERNS = [
        r'localhost',
        r'127\.0\.0\.1',
        r'0\.0\.0\.0',
        r'10\.\d+\.\d+\.\d+',
        r'172\.(1[6-9]|2\d|3[01])\.\d+\.\d+',
        r'192\.168\.\d+\.\d+',
        r'\.internal',
        r'\.local($|[:/])',
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        
        self.config = config or {}
        self.default_timeout = self.config.get('timeout', 600.0)
        self.max_retries = self.config.get('max_retries', 2)
        self.default_authorization = self.config.get('default_authorization', 0.10)
        self.max_authorization = self.config.get('max_authorization', 5.00)
        self.transport: str = self.config.get('transport', 'uamp')
        
        # Agent communication base URL (the local agent daemon)
        self.agent_base_url = (
            os.getenv('AGENTS_BASE_URL') or
            self.config.get('agent_base_url') or 
            'http://localhost:2224'
        )
        
        # Auth token for agent-to-agent calls (resolved in initialize)
        self._auth_token: Optional[str] = self.config.get('robutler_api_key')
        
        # Communication tracking
        self.communication_history: List[NLICommunication] = []
        self._consecutive_payment_failures = 0
        self._max_payment_failures = 2
        
        # HTTP client (initialized in initialize method)
        self.http_client: Optional[Any] = None
        self.logger = None
    
    def _resolve_agent_to_url(self, agent: str) -> str:
        """Resolve an agent identifier to a full completions URL.
        
        Accepts:
          - @username  ->  {base}/agents/{username}/chat/completions
          - username   ->  {base}/agents/{username}/chat/completions
          - https://example.com/agents/foo  ->  https://example.com/agents/foo/chat/completions
        
        Internal/localhost URLs are REJECTED when they look like the LLM
        fabricated them (the base_url used for routing is always injected
        server-side and never comes from the LLM).
        """
        agent = agent.strip()
        
        # Strip @ prefix
        if agent.startswith('@'):
            agent = agent[1:]
        
        # If it looks like a URL (has ://)
        if '://' in agent:
            # Reject internal URLs provided by the LLM
            for pattern in self.INTERNAL_URL_PATTERNS:
                if re.search(pattern, agent, re.IGNORECASE):
                    raise ValueError(
                        f"Internal URLs are not allowed. Use @username or the agent's display name instead."
                    )
            # It's a public URL - ensure it ends with /chat/completions
            if not agent.rstrip('/').endswith('/chat/completions'):
                agent = agent.rstrip('/') + '/chat/completions'
            return agent
        
        # It's a plain name - route through our daemon
        base = self.agent_base_url.rstrip('/')
        return f"{base}/agents/{agent}/chat/completions"
    
    def _extract_agent_name_or_id(self, agent_url: str) -> Dict[str, Optional[str]]:
        """Extract agent UUID or name from a URL."""
        try:
            uuid_re = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
            m = re.search(rf"/agents/({uuid_re})", agent_url)
            if m:
                return {"id": m.group(1), "name": None}
            parsed = urlparse(agent_url)
            parts = [p for p in parsed.path.split('/') if p]
            if len(parts) >= 3 and parts[-1] == 'completions' and parts[-2] == 'chat':
                return {"id": None, "name": parts[-3]}
        except Exception:
            pass
        return {"id": None, "name": None}
        
    async def initialize(self, agent) -> None:
        """Initialize NLI skill with agent context"""
        from webagents.utils.logging import get_logger, log_skill_event
        
        self.agent = agent
        self.logger = get_logger('skill.webagents.nli', agent.name)
        
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
                timeout=httpx.Timeout(self.default_timeout),
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=50)
            )
            self.logger.info("NLI HTTP client initialized")
        else:
            self.logger.warning("httpx not available - NLI functionality will be limited")
            
        log_skill_event(self.agent.name, 'nli', 'initialized', {
            'default_timeout': self.default_timeout,
            'max_retries': self.max_retries,
            'default_authorization': self.default_authorization,
            'httpx_available': HTTPX_AVAILABLE,
            'websockets_available': WEBSOCKETS_AVAILABLE,
            'transport': self.transport,
            'has_auth_token': bool(self._auth_token),
        })

    async def _mint_owner_assertion(self, target_agent_id: str, acting_user_id: Optional[str]) -> Optional[str]:
        """Mint a short-lived owner assertion (RS256) via Portal API if possible."""
        if not target_agent_id:
            return None
        portal_base_url = os.getenv('ROBUTLER_INTERNAL_API_URL') or os.getenv('ROBUTLER_API_URL') or 'http://localhost:3000'
        service_token = self._auth_token or os.getenv('SERVICE_TOKEN') or os.getenv('WEBAGENTS_API_KEY')
        if not service_token:
            return None
        try:
            if not HTTPX_AVAILABLE:
                return None
            async with httpx.AsyncClient(timeout=10.0) as client:
                payload: Dict[str, Any] = {"agentId": target_agent_id, "ttlSeconds": 180}
                if acting_user_id:
                    payload["originUserId"] = acting_user_id
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
    
    async def _resolve_agent_id(self, agent_name: str) -> Optional[str]:
        """Resolve agent name to UUID via portal API for owner assertion minting."""
        if not HTTPX_AVAILABLE or not agent_name:
            return None
        portal_base_url = (
            os.getenv('ROBUTLER_API_URL') or
            os.getenv('ROBUTLER_INTERNAL_API_URL') or
            os.getenv('ROBUTLER_API_URL') or
            'http://localhost:3000'
        )
        bearer = self._auth_token or os.getenv('WEBAGENTS_API_KEY') or os.getenv('SERVICE_TOKEN')
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
                # /api/agents/[id] accepts both UUID and username
                r = await client.get(
                    f"{portal_base_url.rstrip('/')}/api/agents/{agent_name}",
                    headers=headers,
                )
                if r.status_code == 200:
                    d = r.json()
                    return d.get('agent', {}).get('id') or d.get('id')
        except Exception:
            pass
        return None
    
    async def _delegate_payment(
        self,
        parent_token: str,
        target_agent_id: str,
        authorized_amount: float,
        agent_identifier: str,
    ) -> Optional[str]:
        """Proactively delegate a portion of the parent token's budget to a sub-agent.
        
        Creates a child token via /api/payments/delegate so the downstream agent
        can lock and charge against its own token rather than the parent's.
        
        The delegation amount is the parent token's full remaining balance so that
        expensive downstream tools (e.g. media generation) aren't under-budgeted.
        The delegate API caps the amount at the parent's available balance and the
        child token is audience-restricted to the target agent.
        
        Returns the child token JWT, or None if delegation failed.
        """
        # Decode parent JWT to read max_depth and available balance
        parent_balance: Optional[float] = None
        try:
            import base64
            parts = parent_token.split('.')
            if len(parts) >= 2:
                padded = parts[1] + '=' * (4 - len(parts[1]) % 4)
                claims = json.loads(base64.urlsafe_b64decode(padded))
                payment_claims = claims.get('payment', {})
                max_depth = payment_claims.get('max_depth')
                if max_depth is not None and max_depth <= 0:
                    self.logger.warning("🔐 max_depth=0 — cannot delegate further")
                    return None
                parent_balance = payment_claims.get('balance')
        except Exception:
            pass
        
        portal_base_url = (
            os.getenv('ROBUTLER_API_URL') or
            os.getenv('ROBUTLER_INTERNAL_API_URL') or
            os.getenv('ROBUTLER_API_URL') or
            'http://localhost:3000'
        )
        bearer = self._auth_token
        if not bearer:
            self.logger.warning("🔐 No auth token available for payment delegation")
            return None
        
        # Delegate generously so sub-agents with expensive tools (media
        # generation, etc.) aren't under-budgeted.  Use the parent JWT's
        # balance when available; fall back to a multiple of authorized_amount.
        if parent_balance and parent_balance > 0:
            mint_amount = parent_balance
        else:
            mint_amount = max(authorized_amount * 3, 0.50) if authorized_amount > 0 else 0.50
        
        delegate_url = f"{portal_base_url.rstrip('/')}/api/payments/delegate"
        headers_req = {
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    delegate_url,
                    json={
                        "parentToken": parent_token,
                        "delegateTo": target_agent_id,
                        "amount": mint_amount,
                    },
                    headers=headers_req,
                )
                
                # If we over-requested, parse the available amount and retry
                if resp.status_code == 400:
                    err = resp.json().get('error', '')
                    import re as _re
                    m = _re.search(r'available:\s*([\d.]+)', err)
                    if m:
                        available = float(m.group(1))
                        if available > 0.001:
                            self.logger.info(f"🔐 Retrying delegation with available ${available:.4f}")
                            resp = await client.post(
                                delegate_url,
                                json={
                                    "parentToken": parent_token,
                                    "delegateTo": target_agent_id,
                                    "amount": available,
                                },
                                headers=headers_req,
                            )
                
                if resp.status_code == 200:
                    data = resp.json()
                    child_token = data.get('token')
                    if child_token:
                        actual = data.get('amountDollars', mint_amount)
                        self.logger.info(f"🔐 ✅ Delegated ${actual:.4f} to @{agent_identifier.lstrip('@')}")
                        return child_token
                self.logger.warning(f"🔐 Delegation failed ({resp.status_code}): {resp.text}")
        except Exception as e:
            self.logger.warning(f"🔐 Delegation error: {e}")
        
        return None
    
    async def _handle_402(
        self,
        response: Any,
        current_payment_token: Optional[str],
        authorized_amount: float,
        agent_identifier: str,
    ) -> Optional[str]:
        """Handle a 402 Payment Required response by delegating from the user's payment token.
        
        Only delegates from an existing parent token (provided by Robutler router).
        Never creates tokens from the agent owner's balance.
        
        Returns the delegated child token JWT, or None if delegation failed.
        """
        if not current_payment_token:
            self.logger.warning(
                "🔐 No parent payment token available for delegation. "
                "The user's payment token was not provided by the platform."
            )
            return None

        try:
            resp_body = response.json() if hasattr(response, 'json') else {}
            if callable(resp_body):
                resp_body = resp_body()
        except Exception:
            resp_body = {}
        
        required_amount = authorized_amount
        context_data = resp_body.get('context', resp_body)
        accepts = context_data.get('accepts', []) if isinstance(context_data, dict) else []
        if accepts and isinstance(accepts, list):
            try:
                required_amount = float(accepts[0].get('amount', authorized_amount))
            except (ValueError, TypeError):
                pass
        
        mint_amount = min(required_amount * 3, authorized_amount)
        if mint_amount <= 0:
            mint_amount = min(0.10, authorized_amount)
        
        # Check max_depth before attempting delegation
        try:
            import base64
            parts = current_payment_token.split('.')
            if len(parts) >= 2:
                padded = parts[1] + '=' * (4 - len(parts[1]) % 4)
                claims = json.loads(base64.urlsafe_b64decode(padded))
                max_depth = claims.get('payment', {}).get('max_depth')
                if max_depth is not None and max_depth <= 0:
                    self.logger.warning("🔐 max_depth=0 — cannot delegate further")
                    return None
        except Exception:
            pass
        
        portal_base_url = (
            os.getenv('ROBUTLER_API_URL') or 
            os.getenv('ROBUTLER_INTERNAL_API_URL') or 
            os.getenv('ROBUTLER_API_URL') or 
            'http://localhost:3000'
        )
        bearer = self._auth_token
        if not bearer:
            self.logger.warning("🔐 No auth token available for payment delegation")
            return None
        
        auth_headers = {
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        }
        
        target_user_id = await self._resolve_agent_id(agent_identifier.lstrip('@'))
        if not target_user_id:
            self.logger.warning(f"🔐 Could not resolve agent ID for @{agent_identifier.lstrip('@')}")
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{portal_base_url.rstrip('/')}/api/payments/delegate",
                    json={
                        "parentToken": current_payment_token,
                        "delegateTo": target_user_id,
                        "amount": mint_amount,
                    },
                    headers=auth_headers,
                )
            if resp.status_code == 200:
                data = resp.json()
                child_token = data.get('token')
                if child_token:
                    self.logger.info(f"🔐 ✅ Delegated ${mint_amount:.4f} to @{agent_identifier.lstrip('@')}")
                    return child_token
            self.logger.warning(f"🔐 Delegation failed ({resp.status_code}): {resp.text}")
        except Exception as e:
            self.logger.warning(f"🔐 Delegation error: {e}")
        
        return None
    
    def _resolve_agent_to_uamp_url(self, agent: str) -> str:
        """Resolve an agent identifier to a UAMP WebSocket URL.
        
        Converts the HTTP base URL to a WS URL with /uamp suffix.
        e.g. http://localhost:2224/agents/bob -> ws://localhost:2224/agents/bob/uamp
        """
        agent = agent.strip().lstrip('@')
        
        if '://' in agent:
            parsed = urlparse(agent)
            scheme = 'wss' if parsed.scheme == 'https' else 'ws'
            path = parsed.path.rstrip('/')
            if path.endswith('/chat/completions'):
                path = path[:-len('/chat/completions')]
            return f"{scheme}://{parsed.netloc}{path}/uamp"
        
        base = self.agent_base_url.rstrip('/')
        parsed = urlparse(base)
        scheme = 'wss' if parsed.scheme == 'https' else 'ws'
        return f"{scheme}://{parsed.netloc}{parsed.path}/agents/{agent}/uamp"

    async def _send_via_uamp(
        self,
        agent_identifier: str,
        message: str,
        headers: Dict[str, str],
        payment_token: Optional[str],
        timeout: float,
        context=None,
    ) -> Optional[str]:
        """Send a message via UAMP WebSocket transport. Returns response text or None on failure."""
        if not WEBSOCKETS_AVAILABLE:
            return None
        
        ws_url = self._resolve_agent_to_uamp_url(agent_identifier)
        
        params = []
        auth_token = headers.get('Authorization', '').replace('Bearer ', '')
        if auth_token:
            params.append(f"token={auth_token}")
        if payment_token:
            params.append(f"payment_token={payment_token}")
        if params:
            ws_url = f"{ws_url}?{'&'.join(params)}"
        
        progress_queue = None
        current_tc_id = None
        if context:
            progress_queue = context.get("_progress_queue") if hasattr(context, 'get') else getattr(context, '_progress_queue', None)
            current_tc_id = context.get("_current_tool_call_id") if hasattr(context, 'get') else getattr(context, '_current_tool_call_id', None)
        if not progress_queue:
            try:
                from webagents.server.context.context_vars import get_context as _gc
                ctx = _gc()
                if ctx:
                    progress_queue = ctx.get("_progress_queue") if hasattr(ctx, 'get') else getattr(ctx, '_progress_queue', None)
                    current_tc_id = current_tc_id or (ctx.get("_current_tool_call_id") if hasattr(ctx, 'get') else getattr(ctx, '_current_tool_call_id', None))
            except Exception:
                pass
        
        response_text = ""
        
        try:
            async with asyncio.timeout(timeout):
                async with websockets.connect(ws_url) as ws:
                    import uuid as _uuid
                    
                    session_create = json.dumps({
                        "type": "session.create",
                        "event_id": str(_uuid.uuid4()),
                        "timestamp": int(datetime.utcnow().timestamp() * 1000),
                        "uamp_version": "1.0",
                        "session": {
                            "modalities": ["text"],
                            "extensions": {
                                "X-Payment-Token": payment_token,
                            } if payment_token else {},
                        },
                    })
                    await ws.send(session_create)
                    
                    created = json.loads(await ws.recv())
                    if created.get("type") != "session.created":
                        self.logger.warning(f"UAMP: unexpected response to session.create: {created.get('type')}")
                        return None
                    
                    input_text = json.dumps({
                        "type": "input.text",
                        "event_id": str(_uuid.uuid4()),
                        "timestamp": int(datetime.utcnow().timestamp() * 1000),
                        "text": message,
                        "role": "user",
                    })
                    await ws.send(input_text)
                    
                    response_create = json.dumps({
                        "type": "response.create",
                        "event_id": str(_uuid.uuid4()),
                        "timestamp": int(datetime.utcnow().timestamp() * 1000),
                    })
                    await ws.send(response_create)
                    
                    got_first_content = False
                    while True:
                        raw = await ws.recv()
                        event = json.loads(raw)
                        evt_type = event.get("type", "")
                        
                        if evt_type == "response.delta":
                            delta = event.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                if not got_first_content:
                                    got_first_content = True
                                response_text += text
                                if progress_queue and current_tc_id:
                                    await progress_queue.put({
                                        "type": "tool_progress",
                                        "call_id": current_tc_id,
                                        "text": text,
                                    })
                        
                        elif evt_type == "response.done":
                            break
                        
                        elif evt_type == "response.error":
                            err = event.get("error", {})
                            raise Exception(err.get("message", "Agent response error"))
                        
                        elif evt_type == "payment.required":
                            if payment_token:
                                import uuid as _uuid2
                                submit = json.dumps({
                                    "type": "payment.submit",
                                    "event_id": str(_uuid2.uuid4()),
                                    "timestamp": int(datetime.utcnow().timestamp() * 1000),
                                    "payment": {
                                        "scheme": "token",
                                        "amount": event.get("requirements", {}).get("amount", "0.01"),
                                        "token": payment_token,
                                    },
                                })
                                await ws.send(submit)
                            else:
                                raise Exception("Payment required but no payment token available")
                        
                        elif evt_type == "payment.error":
                            raise Exception(event.get("message", "Payment error"))
            
            return response_text if response_text else None
        
        except (asyncio.TimeoutError, TimeoutError):
            self.logger.warning(f"UAMP: timeout after {timeout}s to {agent_identifier}")
            return None
        except Exception as e:
            self.logger.warning(f"UAMP: transport failed for {agent_identifier}: {e}")
            return None

    async def cleanup(self):
        """Cleanup NLI resources"""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

    @prompt(priority=20, scope="all")
    def nli_general_prompt(self, context: Any = None) -> str:
        agent_name = self.agent.name if self.agent and hasattr(self.agent, 'name') else None
        parts = [
            "To talk to other agents, use nli_tool with @username (e.g. @assistant, @r-banana).",
            "Use discovery_tool first to find agents by capability if you don't know who to contact.",
            "NEVER fabricate agent names or URLs - always discover first.",
            "",
            "IMPORTANT — Before contacting agents:",
            "- If the user's request can be fulfilled by ONE specific agent, call that agent directly.",
            "- If you need to try MULTIPLE agents (e.g. searching for the right one), ASK THE USER for permission first.",
            "  Say something like: 'I can try contacting @agent-a, @agent-b, and @agent-c to find the best option. Shall I proceed?'",
            "- If an NLI call fails (payment error, timeout, etc.), tell the user about the failure and ask before trying alternatives.",
            "- NEVER silently fan out to many agents without user consent — each call costs money.",
        ]
        if agent_name:
            parts.append(f"You are @{agent_name}. NEVER call yourself via NLI!")
        return "\n".join(parts)
    
    @tool(description="Send a message to another AI agent. Use @username to identify the target agent. Use discovery_tool first if you don't know who to contact.", scope="all")
    async def nli_tool(self, 
                       agent: str, 
                       message: str, 
                       authorized_amount: float = None,
                       timeout: float = None,
                       context=None) -> str:
        """
        Natural Language Interface to communicate with other WebAgents agents.
        
        Args:
            agent: Agent identifier - use @username (e.g. @r-banana, @assistant).
                   You can also use a display name or a full public URL.
            message: Natural language message to send to the agent
            authorized_amount: Maximum cost authorization in USD (default: $0.10, max: $5.00)
            timeout: Request timeout in seconds (default: 600)
            context: Request context (injected by framework, not for LLM)
            
        Returns:
            Response message from the target agent, or error description
            
        Examples:
            - nli_tool("@r-banana", "Generate an image of a sunset")
            - nli_tool("@code-reviewer", "Review this Python function", authorized_amount=0.50)
        """
        start_time = datetime.utcnow()
        agent_identifier = agent.strip() if agent else ""
        
        # Validate input
        if not agent_identifier:
            return "❌ Please provide an agent identifier (e.g. @username)"
        if not message or not message.strip():
            return "❌ Please provide a message to send"
        
        if self._consecutive_payment_failures >= self._max_payment_failures:
            return (
                f"❌ PAYMENT DELEGATION FAILED — {self._consecutive_payment_failures} consecutive agents "
                f"rejected due to insufficient balance or delegation depth. "
                f"STOP trying other agents. Inform the user that payment delegation is not working "
                f"and they may need to top up their balance or reduce the delegation chain depth."
            )
        
        # Prevent self-calling
        if self.agent and hasattr(self.agent, 'name'):
            own_name = self.agent.name
            clean = agent_identifier.lstrip('@').lower()
            if clean == own_name.lower():
                return f"❌ Cannot use nli_tool to call yourself (@{own_name})! Execute your own tasks directly."
        
        # Outbound trust check (talkTo rules)
        target_name = agent_identifier.lstrip("@").lower()
        if self.agent and hasattr(self.agent, "config"):
            talk_to = (self.agent.config or {}).get("talk_to")
            if talk_to is not None:
                try:
                    from webagents.trust import evaluate_trust_rules
                    if not evaluate_trust_rules(
                        caller=getattr(self.agent, "name", ""),
                        target=target_name,
                        rules=talk_to,
                    ):
                        return f"❌ Cannot communicate with @{target_name} — not in your trust scope."
                except ImportError:
                    pass

        # Resolve agent identifier to URL
        try:
            agent_url = self._resolve_agent_to_url(agent_identifier)
        except ValueError as e:
            return f"❌ {str(e)}"
        
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
            "model": self.agent.name,
            "messages": [{"role": "user", "content": message}],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        # Build headers -- use the resolved auth token (config > agent key > env)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"WebAgents-NLI/{self.agent.name}",
            "X-Authorization-Amount": str(authorized_amount),
            "X-Origin-Agent": self.agent.name,
        }

        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
            headers["X-API-Key"] = self._auth_token

        # Forward payment token and mint owner assertion
        acting_user_id: Optional[str] = None
        payment_token: Optional[str] = None
        try:
            from webagents.server.context.context_vars import get_context as _gc
            ctx = _gc()
            acting_user_id = getattr(getattr(ctx, 'auth', None), 'user_id', None) if ctx else None
            
            # Priority 1: payments context (set by PaymentSkill, even when billing disabled)
            payments_ctx = getattr(ctx, 'payments', None) if ctx else None
            if payments_ctx and hasattr(payments_ctx, 'payment_token') and payments_ctx.payment_token:
                payment_token = payments_ctx.payment_token
            
            # Priority 2: transport-level context.payment_token
            if not payment_token and ctx and hasattr(ctx, 'payment_token') and ctx.payment_token:
                payment_token = ctx.payment_token
            
            # Priority 3: HTTP request headers
            if not payment_token and ctx and hasattr(ctx, 'request') and ctx.request:
                request_headers = getattr(ctx.request, 'headers', {})
                if hasattr(request_headers, 'get'):
                    payment_token = (
                        request_headers.get('X-Payment-Token') or 
                        request_headers.get('x-payment-token') or
                        request_headers.get('payment_token')
                    )
            
            if payment_token:
                # Check max_depth before making the call
                try:
                    import base64 as _b64
                    parts = payment_token.split('.')
                    if len(parts) >= 2:
                        padded = parts[1] + '=' * (4 - len(parts[1]) % 4)
                        claims = json.loads(_b64.urlsafe_b64decode(padded))
                        max_depth = claims.get('payment', {}).get('max_depth')
                        if max_depth is not None and max_depth <= 0:
                            return (
                                f"❌ Maximum agent delegation depth reached. "
                                f"Cannot call @{agent_identifier.lstrip('@')} — the NLI call chain is too deep."
                            )
                except Exception:
                    pass
            else:
                self.logger.debug(f"🔐 No payment token in context - will handle 402 if needed")
        except Exception:
            pass

        # Resolve agent ID for owner assertion + payment delegation
        target = self._extract_agent_name_or_id(agent_url)
        target_agent_id = target.get('id')
        if not target_agent_id:
            name_from_path = target.get('name')
            if name_from_path:
                target_agent_id = await self._resolve_agent_id(name_from_path)
        
        # Delegate payment: create a child token for the target agent instead
        # of forwarding the raw parent token (whose balance is locked by us).
        # The delegate endpoint accepts both UUIDs and usernames.
        if payment_token:
            delegate_to = target_agent_id or target.get('name') or agent_identifier.lstrip('@')
            child_token = await self._delegate_payment(
                payment_token, delegate_to, authorized_amount, agent_identifier,
            )
            if child_token:
                headers["X-Payment-Token"] = child_token
                self.logger.info(f"🔐 ✅ Delegated child payment token to @{agent_identifier.lstrip('@')}")
            else:
                headers["X-Payment-Token"] = payment_token
                self.logger.warning(f"🔐 ⚠️ Delegation failed, forwarding raw parent token")

        if target_agent_id:
            assertion = await self._mint_owner_assertion(target_agent_id, acting_user_id)
            if assertion:
                headers["X-Owner-Assertion"] = assertion
                headers["x-owner-assertion"] = assertion
        
        # Send request with retry logic
        communication = None
        last_error = None
        try:
            self.logger.info(f"🔗 Sending NLI message to {agent_identifier} -> {agent_url} (transport={self.transport})")
            
            if self.transport in ('uamp', 'auto') and WEBSOCKETS_AVAILABLE:
                uamp_result = await self._send_via_uamp(
                    agent_identifier, message, headers, payment_token, timeout, context,
                )
                if uamp_result is not None:
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    communication = NLICommunication(
                        timestamp=start_time,
                        target_agent=agent_identifier,
                        target_url=agent_url,
                        message=message,
                        response=uamp_result,
                        cost_usd=authorized_amount,
                        duration_ms=duration_ms,
                        success=True,
                    )
                    self.communication_history.append(communication)
                    try:
                        log_tool_execution(self.agent.name, 'nli_tool', int(duration_ms), success=True)
                    except Exception:
                        pass
                    self.logger.info(f"✅ NLI (UAMP) with {agent_identifier} successful ({duration_ms:.0f}ms)")
                    self._consecutive_payment_failures = 0
                    return uamp_result
                elif self.transport == 'uamp':
                    return f"❌ UAMP transport failed for @{agent_identifier.lstrip('@')}"
            
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self.http_client.post(
                        agent_url,
                        json=payload,
                        headers=headers,
                        timeout=timeout
                    )
                    
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    if response.status_code == 200:
                        agent_response = ""
                        progress_queue = None
                        current_tc_id = None
                        if context:
                            progress_queue = context.get("_progress_queue") if hasattr(context, 'get') else getattr(context, '_progress_queue', None)
                            current_tc_id = context.get("_current_tool_call_id") if hasattr(context, 'get') else getattr(context, '_current_tool_call_id', None)
                        if not progress_queue:
                            try:
                                from webagents.server.context.context_vars import get_context as _gc2
                                ctx2 = _gc2()
                                if ctx2:
                                    progress_queue = ctx2.get("_progress_queue") if hasattr(ctx2, 'get') else getattr(ctx2, '_progress_queue', None)
                                    current_tc_id = current_tc_id or (ctx2.get("_current_tool_call_id") if hasattr(ctx2, 'get') else getattr(ctx2, '_current_tool_call_id', None))
                            except Exception as _ctx_err:
                                self.logger.warning(f"⚠️ NLI progress context error: {_ctx_err}")
                        self.logger.info(f"📡 NLI streaming from {agent_identifier}: progress_queue={'YES' if progress_queue else 'NO'} tc_id={current_tc_id}")
                        progress_count = 0
                        got_first_content = False

                        async def _heartbeat():
                            """Emit periodic status events while waiting for sub-agent."""
                            import asyncio as _hb_asyncio
                            await _hb_asyncio.sleep(3)
                            dots = 1
                            while not got_first_content:
                                if progress_queue and current_tc_id:
                                    await progress_queue.put({
                                        "type": "tool_progress",
                                        "call_id": current_tc_id,
                                        "text": "." * dots + " ",
                                    })
                                dots = (dots % 3) + 1
                                await _hb_asyncio.sleep(4)

                        heartbeat_task = asyncio.create_task(_heartbeat()) if progress_queue and current_tc_id else None

                        pending_tool_names: dict[int, str] = {}
                        tool_call_announced = False
                        captured_signature: str | None = None
                        current_event_type: str | None = None

                        try:
                            async for line in response.aiter_lines():
                                if not line or line.strip() == "":
                                    current_event_type = None
                                    continue
                                if line.startswith("event: "):
                                    current_event_type = line[7:].strip()
                                    continue
                                if line.startswith("data: "):
                                    data_str = line[6:]
                                    # Capture response_signature event (optional, after [DONE])
                                    if current_event_type == "response_signature":
                                        try:
                                            sig_data = json.loads(data_str)
                                            captured_signature = sig_data.get("signature")
                                        except Exception:
                                            pass
                                        continue
                                    if data_str == "[DONE]":
                                        continue
                                    try:
                                        chunk_data = json.loads(data_str)
                                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                            choice = chunk_data['choices'][0]
                                            for msg_key in ['delta', 'message']:
                                                if msg_key not in choice:
                                                    continue
                                                msg = choice[msg_key]

                                                if 'tool_calls' in msg and msg['tool_calls']:
                                                    for tc in msg['tool_calls']:
                                                        idx = tc.get('index', 0)
                                                        fn = tc.get('function', {})
                                                        if fn.get('name'):
                                                            pending_tool_names[idx] = fn['name']
                                                    if not tool_call_announced and pending_tool_names and progress_queue and current_tc_id:
                                                        tool_call_announced = True
                                                        got_first_content = True
                                                        names = ", ".join(pending_tool_names.values())
                                                        progress_count += 1
                                                        await progress_queue.put({
                                                            "type": "tool_progress",
                                                            "call_id": current_tc_id,
                                                            "text": f"Calling {names}...\n\n",
                                                        })

                                                if 'content' in msg and msg['content']:
                                                    if not got_first_content:
                                                        got_first_content = True
                                                    text_chunk = msg['content']
                                                    agent_response += text_chunk
                                                    if progress_queue and current_tc_id:
                                                        progress_count += 1
                                                        await progress_queue.put({
                                                            "type": "tool_progress",
                                                            "call_id": current_tc_id,
                                                            "text": text_chunk,
                                                        })
                                    except json.JSONDecodeError:
                                        pass
                        except asyncio.CancelledError:
                            await response.aclose()
                            raise
                        finally:
                            got_first_content = True
                            if heartbeat_task and not heartbeat_task.done():
                                heartbeat_task.cancel()
                                try:
                                    await heartbeat_task
                                except asyncio.CancelledError:
                                    pass

                        self.logger.info(f"📡 NLI streaming done: {progress_count} progress events emitted, response={len(agent_response)} chars")
                        
                        if not agent_response:
                            agent_response = "Agent returned empty response"
                        
                        if captured_signature:
                            self.logger.info(f"🔏 Captured response signature from {agent_identifier}")
                        communication = NLICommunication(
                            timestamp=start_time,
                            target_agent=agent_identifier,
                            target_url=agent_url,
                            message=message,
                            response=agent_response,
                            cost_usd=authorized_amount,
                            duration_ms=duration_ms,
                            success=True,
                            response_signature=captured_signature,
                        )
                        self.communication_history.append(communication)
                        try:
                            log_tool_execution(self.agent.name, 'nli_tool', int(duration_ms), success=True)
                        except Exception:
                            pass
                        
                        self.logger.info(f"✅ NLI communication with {agent_identifier} successful ({duration_ms:.0f}ms)")
                        self._consecutive_payment_failures = 0
                        return agent_response
                    
                    elif response.status_code == 401 or response.status_code == 403:
                        last_error = f"Authentication failed when contacting @{agent_identifier.lstrip('@')}. The target agent requires valid credentials."
                        self.logger.warning(f"❌ NLI auth failure (HTTP {response.status_code}): {response.text}")
                        break
                    elif response.status_code == 402:
                        delegated_token = await self._handle_402(response, payment_token, authorized_amount, agent_identifier)
                        if delegated_token:
                            headers["X-Payment-Token"] = delegated_token
                            payment_token = delegated_token
                            self.logger.info(f"🔐 ✅ Delegated payment token, retrying...")
                            continue
                        self._consecutive_payment_failures += 1
                        if not payment_token:
                            last_error = (
                                f"Payment required by @{agent_identifier.lstrip('@')}. "
                                "The user's payment token was not provided by the platform — "
                                "agent-to-agent payment requires a user-originated token. "
                                "Do NOT try contacting other agents — the payment issue is systemic."
                            )
                        else:
                            last_error = (
                                f"Payment required by @{agent_identifier.lstrip('@')} but delegation failed. "
                                "The payment token may have insufficient balance or max delegation depth reached. "
                                "Do NOT try contacting other agents — they will likely fail for the same reason. "
                                "Inform the user about the payment issue instead."
                            )
                        self.logger.warning(f"❌ NLI payment required (HTTP 402): {response.text}")
                        break
                    else:
                        last_error = f"HTTP {response.status_code}: {response.text}"
                        self.logger.warning(f"❌ NLI attempt {attempt + 1} failed: {last_error}")
                        if 400 <= response.status_code < 500:
                            break
                        if attempt < self.max_retries:
                            await asyncio.sleep(2 ** attempt)
                
                except httpx.TimeoutException:
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
                target_agent=agent_identifier,
                target_url=agent_url,
                message=message,
                response="",
                cost_usd=0.0,
                duration_ms=duration_ms,
                success=False,
                error=last_error
            )
            self.communication_history.append(communication)
            try:
                log_tool_execution(self.agent.name, 'nli_tool', int(duration_ms), success=False)
            except Exception:
                pass
            
            self.logger.error(f"❌ NLI communication with {agent_identifier} failed: {last_error}")
            return f"❌ Failed to communicate with @{agent_identifier.lstrip('@')}: {last_error}"
            
        except Exception as e:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            error_msg = f"Unexpected error: {str(e)}"
            communication = NLICommunication(
                timestamp=start_time,
                target_agent=agent_identifier,
                target_url=agent_url,
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
            return f"❌ Communication error with @{agent_identifier.lstrip('@')}: {error_msg}"
    
    async def stream_message(
        self,
        agent_url: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        authorized_amount: float = None,
        timeout: float = None
    ):
        """Stream response from remote agent via NLI (for handoff use)"""
        if authorized_amount is None:
            authorized_amount = self.default_authorization
        if authorized_amount > self.max_authorization:
            raise ValueError(f"Authorized amount ${authorized_amount:.2f} exceeds maximum ${self.max_authorization:.2f}")
        if timeout is None:
            timeout = self.default_timeout
        if not HTTPX_AVAILABLE:
            raise ValueError("HTTP client not available - install httpx")
        
        payload = {
            "model": self.agent.name if self.agent else "unknown",
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 4096
        }
        if tools:
            payload["tools"] = tools
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"WebAgents-NLI/{self.agent.name if self.agent else 'unknown'}",
            "X-Authorization-Amount": str(authorized_amount),
            "X-Origin-Agent": self.agent.name if self.agent else "unknown",
        }
        
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
            headers["X-API-Key"] = self._auth_token
        
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
        except Exception:
            pass
        
        # Ensure URL ends with /chat/completions
        if not agent_url.rstrip('/').endswith('/chat/completions'):
            agent_url = agent_url.rstrip('/') + '/chat/completions'
        
        self.logger.info(f"🌊 Starting streaming handoff to: {agent_url}")
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", agent_url, json=payload, headers=headers
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or line.startswith(':'):
                            continue
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                yield chunk
                            except json.JSONDecodeError:
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get NLI communication statistics"""
        total = len(self.communication_history)
        successful = sum(1 for c in self.communication_history if c.success)
        total_cost = sum(c.cost_usd for c in self.communication_history)
        avg_duration = sum(c.duration_ms for c in self.communication_history) / total if total > 0 else 0
        
        return {
            'total_communications': total,
            'successful_communications': successful,
            'success_rate': successful / total if total > 0 else 0,
            'total_cost_usd': total_cost,
            'average_duration_ms': avg_duration,
            'httpx_available': HTTPX_AVAILABLE
        }
