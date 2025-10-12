"""
OpenAI Agent Builder Skill

Runs OpenAI hosted agents/workflows and normalizes their responses
to OpenAI chat completion format for seamless handoff integration.
"""

import os
import json
import httpx
import time
import urllib.parse
from typing import Dict, Any, List, Optional, AsyncGenerator
from webagents.agents.skills import Skill
from webagents.agents.skills.base import Handoff
from webagents.agents.tools.decorators import tool, prompt, http
from webagents.utils.logging import get_logger
from webagents.server.context.context_vars import get_context

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, will use existing env vars

logger = get_logger('openai_agent_builder')


class OpenAIAgentBuilderSkill(Skill):
    """Skill for running OpenAI hosted agents/workflows via streaming handoffs"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI Agent Builder Skill
        
        Args:
            config: Configuration dictionary with:
                - workflow_id: OpenAI workflow ID (optional, can be stored in KV)
                - api_key: OpenAI API key (optional, can be stored in KV or OPENAI_API_KEY env var)
                - api_base: OpenAI API base URL (defaults to https://api.openai.com/v1)
                - version: Workflow version (optional, defaults to None = use workflow default)
        """
        super().__init__(config or {}, scope="all")
        
        # Environment variable credentials (fallback when KV not available)
        self.api_key = self.config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.workflow_id = self.config.get('workflow_id')
        
        self.api_base = self.config.get('api_base', 'https://api.openai.com/v1')
        self.version = self.config.get('version')  # Optional: workflow version (None = use default)
        
        # Base URL for setup callback
        env_agents = os.getenv("AGENTS_BASE_URL")
        base_root = (env_agents or "http://localhost:2224").rstrip('/')
        if base_root.endswith("/agents"):
            self.agent_base_url = base_root
        else:
            self.agent_base_url = base_root + "/agents"
        
        self.logger = get_logger('openai_agent_builder')
        
        # State for thinking detection
        self._in_thinking_block = False
        
        # State for widget data accumulation
        self._widget_data_buffer = ""
    
    async def initialize(self, agent):
        """Register as streaming handoff handler"""
        self.agent = agent
        
        # Register as handoff (streaming for real-time workflow execution)
        # Always use simple target name since workflow_id may be loaded from KV later
        target_name = "openai_workflow"
        description = "OpenAI Workflow handler"
        
        # Use priority 15 (lower than default LLM) - this handoff is dynamically invoked, not default
        priority = 15
        
        agent.register_handoff(
            Handoff(
                target=target_name,
                description=description,
                scope="all",
                metadata={
                    'function': self.run_workflow_stream,
                    'priority': priority,
                    'is_generator': True  # Streaming
                }
            ),
            source="openai_agent_builder"
        )
        
        # Register handoff prompt to tell LLM when to use this handoff
        handoff_prompt_text = self._create_handoff_prompt()
        if handoff_prompt_text:
            # Create a prompt function that returns the prompt text
            def openai_workflow_handoff_prompt():
                return handoff_prompt_text
            
            agent.register_prompt(
                openai_workflow_handoff_prompt,
                priority=3,  # Lower priority - only use when explicitly requested
                source="openai_agent_builder_handoff_prompt",
                scope="all"
            )
            self.logger.debug(f"📨 Registered handoff prompt for '{target_name}'")
        
        if self.workflow_id:
            self.logger.info(f"🔧 OpenAI Agent Builder registered with workflow: {self.workflow_id}")
        else:
            self.logger.info("🔧 OpenAI Agent Builder registered (workflow ID will be loaded from KV)")
    
    def _create_handoff_prompt(self) -> Optional[str]:
        """Create handoff prompt to guide LLM on when to use OpenAI workflow"""
        return """
## OpenAI Workflow Available

You have access to an OpenAI hosted workflow/agent that you can invoke using the `use_openai_workflow` tool.

**When to use**: ONLY call `use_openai_workflow()` when the user **explicitly** requests it:
- "use openai workflow" / "use openai agent" / "use the openai workflow"
- "switch to openai" / "hand off to openai"

**When NOT to use**: 
- Do NOT use this for general requests (images, search, documents, etc.)
- Do NOT use this unless the user explicitly mentions "openai" or "workflow"
- Use your other available tools for normal tasks

**How it works**: When you call this tool, the conversation is handed off to the OpenAI workflow, which streams its response directly to the user.
""".strip()
    
    # ---------------- Credential Management ----------------
    
    async def _get_kv_skill(self):
        """Get KV skill for credential storage"""
        return self.agent.skills.get("kv") or self.agent.skills.get("json_storage")
    
    async def _get_owner_id_from_context(self) -> Optional[str]:
        """Get owner ID from request context"""
        try:
            ctx = get_context()
            if not ctx:
                return None
            auth = getattr(ctx, 'auth', None) or (ctx and ctx.get('auth'))
            return getattr(auth, 'owner_id', None) or getattr(auth, 'user_id', None)
        except Exception:
            return None
    
    async def _save_credentials(self, api_key: str, workflow_id: str) -> None:
        """Save OpenAI credentials to KV storage"""
        kv_skill = await self._get_kv_skill()
        if kv_skill and hasattr(kv_skill, 'kv_set'):
            creds = {"api_key": api_key, "workflow_id": workflow_id}
            await kv_skill.kv_set(key="openai_credentials", value=json.dumps(creds), namespace="openai")
    
    async def _load_credentials(self) -> Optional[Dict[str, str]]:
        """Load OpenAI credentials from KV storage"""
        kv_skill = await self._get_kv_skill()
        if kv_skill and hasattr(kv_skill, 'kv_get'):
            try:
                stored = await kv_skill.kv_get(key="openai_credentials", namespace="openai")
                if isinstance(stored, str) and stored.startswith('{'):
                    return json.loads(stored)
            except Exception:
                pass
        return None
    
    def _build_setup_url(self) -> str:
        """Build URL for credential setup form
        
        For localhost environments, includes auth token in URL since
        cookies don't work across different ports (3000 -> 2224).
        In production, same origin means cookies work normally.
        """
        base = self.agent_base_url.rstrip('/')
        url = f"{base}/{self.agent.name}/setup/openai"
        
        # Include auth token for localhost only (cross-port authentication)
        if 'localhost' in base or '127.0.0.1' in base:
            if hasattr(self.agent, 'api_key') and self.agent.api_key:
                url += f"?token={self.agent.api_key}"
        
        return url
    
    def _setup_form_html(self, success: bool = False, error: str = None, token: str = None) -> str:
        """Generate HTML for credential setup form"""
        from string import Template
        
        color_ok = "#16a34a"  # green-600
        color_err = "#dc2626"  # red-600
        accent = color_ok if success else color_err
        title = "OpenAI Setup Complete" if success else ("OpenAI Setup Error" if error else "OpenAI Setup")
        
        # Basic HTML escape
        safe_error = (error or '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('$','$$') if error else ''
        
        if success:
            message_html = f"""
                <div style="background: {color_ok}; color: white; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
                    <div style="font-weight: 600; margin-bottom: 0.25rem;">✓ Credentials saved successfully</div>
                    <div style="font-size: 0.875rem; opacity: 0.9;">Your OpenAI API key and workflow ID have been configured.</div>
                </div>
                <p style="margin-bottom: 1rem;">You can now close this window and return to your agent.</p>
            """
        elif error:
            message_html = f"""
                <div style="background: {color_err}; color: white; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
                    <div style="font-weight: 600; margin-bottom: 0.25rem;">✗ Setup failed</div>
                    <div style="font-size: 0.875rem; opacity: 0.9;">{safe_error}</div>
                </div>
            """
        else:
            message_html = ""
        
        # Include token in form action for localhost cross-port auth
        form_action = f"?token={token}" if token else ""
        
        form_html = "" if success else f"""
            <form method="post" action="{form_action}" style="display: flex; flex-direction: column; gap: 1rem;">
                <div>
                    <label for="api_key" style="display: block; font-weight: 600; margin-bottom: 0.5rem;">OpenAI API Key</label>
                    <input 
                        type="password" 
                        id="api_key" 
                        name="api_key" 
                        required 
                        placeholder="sk-..."
                        style="width: 100%; padding: 0.75rem; border: 1px solid var(--border, #374151); border-radius: 0.5rem; background: var(--input-bg, #1f2937); color: var(--fg, #e5e7eb); font-family: ui-monospace, monospace; font-size: 0.875rem;"
                    />
                </div>
                <div>
                    <label for="workflow_id" style="display: block; font-weight: 600; margin-bottom: 0.5rem;">Workflow ID</label>
                    <input 
                        type="text" 
                        id="workflow_id" 
                        name="workflow_id" 
                        required 
                        placeholder="wf_..."
                        style="width: 100%; padding: 0.75rem; border: 1px solid var(--border, #374151); border-radius: 0.5rem; background: var(--input-bg, #1f2937); color: var(--fg, #e5e7eb); font-family: ui-monospace, monospace; font-size: 0.875rem;"
                    />
                </div>
                <button 
                    type="submit" 
                    style="padding: 0.75rem 1.5rem; background: {accent}; color: white; border: none; border-radius: 0.5rem; font-weight: 600; cursor: pointer; font-size: 1rem;"
                >
                    Save Credentials
                </button>
            </form>
        """
        
        template = Template("""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>WebAgents – OpenAI Setup</title>
    <style>
      :root { color-scheme: light dark; }
      html, body { height: 100%; margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
      body { background: var(--bg, #0b0b0c); color: var(--fg, #e5e7eb); display: grid; place-items: center; padding: 1rem; }
      @media (prefers-color-scheme: light) { 
        body { --bg: #f7f7f8; --card: #ffffff; --border: #e5e7eb; --fg: #0f172a; --input-bg: #ffffff; } 
      }
      .card { 
        background: var(--card, #18181b); 
        border: 1px solid var(--border, #27272a); 
        border-radius: 1rem; 
        padding: 2rem; 
        max-width: 28rem; 
        width: 100%; 
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1); 
      }
      h1 { margin: 0 0 1.5rem 0; font-size: 1.5rem; font-weight: 700; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>${title}</h1>
      ${message}
      ${form}
    </div>
  </body>
</html>""")
        
        return template.substitute(title=title, message=message_html, form=form_html)
    
    # ---------------- HTTP Endpoints ----------------
    
    @http(subpath="/setup/openai", method="get", scope=["owner"])
    async def show_setup_form(self, token: str = None) -> Dict[str, Any]:
        """Show credential setup form (GET endpoint)"""
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=self._setup_form_html(token=token))
    
    @http(subpath="/setup/openai", method="post", scope=["owner"])
    async def setup_credentials(self, api_key: str = "", workflow_id: str = "", token: str = None) -> Dict[str, Any]:
        """Save OpenAI credentials (POST endpoint)"""
        from fastapi.responses import HTMLResponse
        
        # Strip whitespace
        api_key = (api_key or "").strip()
        workflow_id = (workflow_id or "").strip()
        
        if not api_key or not workflow_id:
            return HTMLResponse(content=self._setup_form_html(error="Both API key and workflow ID are required", token=token))
        
        try:
            await self._save_credentials(api_key, workflow_id)
            return HTMLResponse(content=self._setup_form_html(success=True, token=token))
        except Exception as e:
            return HTMLResponse(content=self._setup_form_html(error=str(e), token=token))
    
    # ---------------- Prompts ----------------
    
    @prompt(priority=40, scope=["owner", "all"])
    async def openai_prompt(self) -> str:
        """Provide setup guidance if credentials not configured"""
        kv_skill = await self._get_kv_skill()
        if kv_skill:
            creds = await self._load_credentials()
            if not creds:
                setup_url = self._build_setup_url()
                return f"OpenAI workflow skill available but not configured. Set up credentials at: {setup_url}"
        return "OpenAI workflow integration is available for running hosted workflows."
    
    # ---------------- Tools ----------------
    
    @tool(
        description="Switch to OpenAI workflow for direct streaming response (use when user requests OpenAI workflow/agent)",
        scope=["all"]
    )
    async def use_openai_workflow(self) -> str:
        """Request handoff to OpenAI workflow
        
        Returns handoff request marker. The framework will execute the handoff
        and stream the OpenAI workflow response directly to the user.
        """
        # Load credentials to verify configuration
        api_key = self.api_key
        workflow_id = self.workflow_id
        
        if not api_key or not workflow_id:
            creds = await self._load_credentials()
            if creds:
                api_key = creds.get('api_key')
                workflow_id = creds.get('workflow_id')
        
        if not api_key or not workflow_id:
            setup_url = self._build_setup_url()
            return f"❌ OpenAI credentials not configured. Set up at: {setup_url}"
        
        # Use consistent target name (always "openai_workflow")
        return self.request_handoff("openai_workflow")
    
    @tool(description="Update or remove OpenAI credentials (API key and workflow ID)", scope=["owner"])
    async def update_openai_credentials(self, api_key: str = None, workflow_id: str = None, remove: bool = False) -> str:
        """Update or remove stored OpenAI credentials"""
        kv_skill = await self._get_kv_skill()
        if not kv_skill:
            return "❌ KV skill not available. Credentials are configured via environment variables."
        
        if remove:
            try:
                if hasattr(kv_skill, 'kv_delete'):
                    await kv_skill.kv_delete(key="openai_credentials", namespace="openai")
                return "✓ OpenAI credentials removed"
            except Exception as e:
                return f"❌ Failed to remove credentials: {e}"
        
        if not api_key or not workflow_id:
            return "❌ Both api_key and workflow_id are required"
        
        try:
            await self._save_credentials(api_key, workflow_id)
            return "✓ OpenAI credentials updated successfully"
        except Exception as e:
            return f"❌ Failed to update credentials: {e}"
    
    # ---------------- Usage Tracking ----------------
    
    def _log_workflow_usage(self, usage_data: Dict[str, Any], model: Optional[str]) -> None:
        """Log workflow usage to context for cost tracking
        
        Args:
            usage_data: Usage data from workflow response
            model: Model identifier (optional)
        """
        try:
            context = get_context()
            if not context or not hasattr(context, 'usage'):
                return
            
            # Extract token counts from usage data
            # OpenAI workflows may use different field names
            prompt_tokens = usage_data.get('prompt_tokens', 0) or usage_data.get('input_tokens', 0)
            completion_tokens = usage_data.get('completion_tokens', 0) or usage_data.get('output_tokens', 0)
            total_tokens = usage_data.get('total_tokens', 0) or (prompt_tokens + completion_tokens)
            
            if total_tokens > 0:
                usage_record = {
                    'type': 'llm',
                    'timestamp': time.time(),
                    'model': model or f'openai-workflow-{self.workflow_id}',
                    'prompt_tokens': int(prompt_tokens),
                    'completion_tokens': int(completion_tokens),
                    'total_tokens': int(total_tokens),
                    'streaming': True,
                    'source': 'openai_workflow'
                }
                context.usage.append(usage_record)
                self.logger.info(f"💰 Workflow usage logged: {total_tokens} tokens (prompt={prompt_tokens}, completion={completion_tokens}) for model={model}")
            else:
                self.logger.debug(f"⚠️ Workflow usage data present but no tokens: {usage_data}")
        except Exception as e:
            self.logger.warning(f"Failed to log workflow usage: {e}")
    
    def _wrap_thinking_content(self, delta_text: str, response_data: Dict[str, Any]) -> str:
        """Detect and wrap thinking content in <think> tags
        
        Args:
            delta_text: The delta content from workflow response
            response_data: Full response data for context
            
        Returns:
            Delta text, potentially wrapped in thinking tags
        """
        # Check the 'type' field in response_data for thinking markers
        # OpenAI workflows use: "response.reasoning_summary_text.delta" for thinking
        delta_type = response_data.get('type', '')
        
        # Check if this is reasoning/thinking content
        is_reasoning = 'reasoning' in delta_type.lower()
        is_thinking = 'thinking' in delta_type.lower()
        is_summary = 'summary' in delta_type.lower()
        
        # Reasoning or thinking content should be wrapped
        if is_reasoning or is_thinking or is_summary:
            if not self._in_thinking_block:
                self._in_thinking_block = True
                self.logger.debug(f"🧠 Starting thinking block (type={delta_type})")
                return f"<think>{delta_text}"
            return delta_text
        
        # If we were in a thinking block and now we're not, close it
        if self._in_thinking_block and delta_type and not (is_reasoning or is_thinking or is_summary):
            self._in_thinking_block = False
            self.logger.debug(f"🧠 Ending thinking block (type={delta_type})")
            return f"</think>{delta_text}"
        
        # Regular content - pass through
        return delta_text
    
    def _convert_messages_to_workflow_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI chat messages to OpenAI workflow input format
        
        Args:
            messages: OpenAI format messages [{"role": "user", "content": "..."}]
        
        Returns:
            Workflow input format [{"role": "user", "content": [{"type": "input_text", "text": "..."}]}]
        """
        workflow_input = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Convert string content to workflow format
            if isinstance(content, str):
                workflow_msg = {
                    "role": role,
                    "content": [{"type": "input_text", "text": content}]
                }
            elif isinstance(content, list):
                # Already in structured format
                workflow_msg = {
                    "role": role,
                    "content": content
                }
            else:
                # Fallback
                workflow_msg = {
                    "role": role,
                    "content": [{"type": "input_text", "text": str(content)}]
                }
            
            workflow_input.append(workflow_msg)
        
        return workflow_input
    
    async def run_workflow_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run OpenAI workflow and stream normalized responses
        
        Args:
            messages: OpenAI format chat messages
            tools: Optional tools (not used by workflows currently)
            **kwargs: Additional parameters
        
        Yields:
            OpenAI chat completion streaming chunks
        """
        # Reset usage logging flag and thinking state for this request
        self._usage_logged = False
        self._in_thinking_block = False
        self._widget_data_buffer = ""
        
        # Try to load credentials from KV first, fallback to instance variables
        stored_creds = await self._load_credentials()
        
        if stored_creds:
            api_key = stored_creds.get("api_key")
            workflow_id = stored_creds.get("workflow_id")
            self.logger.debug("🔑 Using credentials from KV storage")
        else:
            # Fallback to environment variables / config
            api_key = self.api_key
            workflow_id = self.workflow_id
            self.logger.debug("🔑 Using credentials from environment/config")
        
        # Check if credentials are available
        if not api_key or not workflow_id:
            kv_skill = await self._get_kv_skill()
            if kv_skill:
                setup_url = self._build_setup_url()
                error_msg = f"OpenAI credentials not configured. Please set up your API key and workflow ID: {setup_url}"
            else:
                error_msg = "OpenAI API key or workflow ID not configured. Please set OPENAI_API_KEY environment variable and workflow_id in config."
            
            self.logger.error(f"❌ {error_msg}")
            yield {
                'id': f'error-{int(time.time())}',
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': f'openai-workflow-{workflow_id or "unknown"}',
                'choices': [{
                    'index': 0,
                    'delta': {'role': 'assistant', 'content': error_msg},
                    'finish_reason': 'stop'
                }]
            }
            return
        
        workflow_url = f"{self.api_base}/workflows/{workflow_id}/run"
        
        # Filter to only user messages (workflows don't handle system/assistant roles)
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        
        if not user_messages:
            # No user messages, use empty input
            workflow_input = []
        else:
            # Convert only user messages to workflow input format
            workflow_input = self._convert_messages_to_workflow_input(user_messages)
        
        # Build request payload matching OpenAI workflows v6 format
        payload = {
            "input_data": {
                "input": workflow_input
            },
            "state_values": [],
            "session": True,  # Enable session for multi-turn conversations
            "tracing": {
                "enabled": True  # Enable tracing for debugging
            },
            "stream": True
        }
        
        # Include version if explicitly specified
        if self.version is not None:
            payload["version"] = str(self.version)
        
        self.logger.debug(f"🔄 Calling OpenAI workflow: {workflow_url}")
        
        headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json"
        }
        
        # Initialize chunk ID counter
        chunk_id = 0
        accumulated_content = ""
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream('POST', workflow_url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    
                    # Parse SSE stream
                    async for line in response.aiter_lines():
                        if not line or line.startswith(':'):
                            continue
                        
                        # Parse SSE format: "event: type" and "data: json"
                        if line.startswith('event: '):
                            current_event = line[7:].strip()
                            continue
                        
                        if line.startswith('data: '):
                            data_str = line[6:].strip()
                            
                            try:
                                data = json.loads(data_str)
                                event_type = data.get('type', current_event if 'current_event' in locals() else '')
                                
                                # Handle workflow.node.agent.response - streaming content deltas
                                if event_type == 'workflow.node.agent.response':
                                    response_data = data.get('data', {})
                                    delta_text = response_data.get('delta')
                                    
                                    # Check for usage data in the response
                                    response_obj = response_data.get('response', {})
                                    if response_obj and isinstance(response_obj, dict):
                                        usage_data = response_obj.get('usage')
                                        model = response_obj.get('model')
                                        
                                        if usage_data and isinstance(usage_data, dict):
                                            # Log usage once (check if we haven't logged it yet)
                                            if not self._usage_logged:
                                                self._usage_logged = True
                                                self._log_workflow_usage(usage_data, model)
                                    
                                    # Yield streaming delta if present and non-empty
                                    if delta_text and isinstance(delta_text, str):
                                        chunk_id += 1
                                        
                                        # Wrap thinking content if this is a reasoning model
                                        wrapped_delta = self._wrap_thinking_content(delta_text, response_data)
                                        
                                        # Accumulate content for widget data detection
                                        # Widget data is JSON that appears right before a widget event
                                        self._widget_data_buffer += wrapped_delta
                                        accumulated_content += wrapped_delta
                                        
                                        # Build delta object
                                        delta_obj = {'content': wrapped_delta}
                                        if chunk_id == 1:
                                            delta_obj['role'] = 'assistant'
                                        
                                        yield {
                                            'id': f'chatcmpl-wf-{self.workflow_id}',
                                            'object': 'chat.completion.chunk',
                                            'created': data.get('workflow_run', {}).get('created_at', 0),
                                            'model': f'openai-workflow-{self.workflow_id}',
                                            'choices': [{
                                                'index': 0,
                                                'delta': delta_obj,
                                                'finish_reason': None
                                            }]
                                        }
                                        continue  # Skip other processing for this event
                                
                                # Handle workflow.finished event
                                if event_type == 'workflow.finished':
                                    self.logger.debug(f"📥 Workflow finished. Total content: {len(accumulated_content)} chars")
                                    
                                    # Check for usage data as fallback (if not already logged)
                                    if not self._usage_logged:
                                        workflow_result = data.get('result', {})
                                        if workflow_result and isinstance(workflow_result, dict):
                                            usage_data = workflow_result.get('usage')
                                            model = workflow_result.get('model')
                                            
                                            if usage_data and isinstance(usage_data, dict):
                                                self._usage_logged = True
                                                self._log_workflow_usage(usage_data, model)
                                    
                                    # Close thinking block if still open
                                    if self._in_thinking_block:
                                        self.logger.debug("🧠 Closing thinking block at workflow finish")
                                        yield {
                                            'id': f'chatcmpl-wf-{self.workflow_id}',
                                            'object': 'chat.completion.chunk',
                                            'created': data.get('workflow_run', {}).get('created_at', 0),
                                            'model': f'openai-workflow-{self.workflow_id}',
                                            'choices': [{
                                                'index': 0,
                                                'delta': {'content': '</think>'},
                                                'finish_reason': None
                                            }]
                                        }
                                        self._in_thinking_block = False
                                    
                                    # Yield finish chunk (content already streamed via deltas)
                                    yield {
                                        'id': f'chatcmpl-wf-{self.workflow_id}',
                                        'object': 'chat.completion.chunk',
                                        'created': data.get('workflow_run', {}).get('created_at', 0),
                                        'model': f'openai-workflow-{self.workflow_id}',
                                        'choices': [{
                                            'index': 0,
                                            'delta': {},
                                            'finish_reason': 'stop'
                                        }]
                                    }
                                
                                # Handle workflow.failed event
                                elif event_type == 'workflow.failed':
                                    error_msg = data.get('workflow_run', {}).get('error', 'Unknown error')
                                    self.logger.error(f"❌ Workflow failed: {json.dumps(error_msg, indent=2)}")
                                    # Yield error message
                                    yield {
                                        'id': f'chatcmpl-wf-{self.workflow_id}',
                                        'object': 'chat.completion.chunk',
                                        'created': data.get('workflow_run', {}).get('created_at', 0),
                                        'model': f'openai-workflow-{self.workflow_id}',
                                        'choices': [{
                                            'index': 0,
                                            'delta': {
                                                'role': 'assistant',
                                                'content': f"Workflow error: {error_msg}"
                                            },
                                            'finish_reason': 'stop'
                                        }]
                                    }
                                
                                # Handle workflow.node.agent.widget event
                                elif event_type == 'workflow.node.agent.widget':
                                    # Check for widget data in multiple possible locations
                                    widget_json = data.get('widget')
                                    widget_data_obj = data.get('data') or data.get('props') or data.get('widget_data')
                                    
                                    self.logger.debug(f"🎨 Widget event received - widget: {bool(widget_json)}, data: {bool(widget_data_obj)}")
                                    
                                    if widget_json:
                                        # Close thinking block if still open (widgets should be outside thinking)
                                        if self._in_thinking_block:
                                            self.logger.debug("🧠 Closing thinking block before widget")
                                            chunk_id += 1
                                            yield {
                                                'id': f'chatcmpl-wf-{self.workflow_id}',
                                                'object': 'chat.completion.chunk',
                                                'created': data.get('workflow_run', {}).get('created_at', 0),
                                                'model': f'openai-workflow-{self.workflow_id}',
                                                'choices': [{
                                                    'index': 0,
                                                    'delta': {'content': '</think>\n'},
                                                    'finish_reason': None
                                                }]
                                            }
                                            accumulated_content += '</think>\n'
                                            self._in_thinking_block = False
                                        
                                        chunk_id += 1
                                        
                                        # Extract widget data - prefer explicit data field from event
                                        widget_data = None
                                        if widget_data_obj:
                                            # Widget event contains the data - use it directly
                                            widget_data = json.dumps(widget_data_obj) if isinstance(widget_data_obj, dict) else str(widget_data_obj)
                                            self.logger.debug(f"🎨 Using widget data from event (length={len(widget_data)})")
                                        elif self._widget_data_buffer:
                                            # Fallback: extract from buffer
                                            # Look for JSON object at the end of the buffer
                                            buffer_stripped = self._widget_data_buffer.strip()
                                            # Remove </think> tag if present in buffer
                                            buffer_stripped = buffer_stripped.replace('</think>', '').strip()
                                            
                                            if buffer_stripped.endswith('}'):
                                                # Find the matching opening brace
                                                brace_count = 0
                                                start_idx = -1
                                                for i in range(len(buffer_stripped) - 1, -1, -1):
                                                    if buffer_stripped[i] == '}':
                                                        brace_count += 1
                                                    elif buffer_stripped[i] == '{':
                                                        brace_count -= 1
                                                        if brace_count == 0:
                                                            start_idx = i
                                                            break
                                                
                                                if start_idx >= 0:
                                                    try:
                                                        widget_data = buffer_stripped[start_idx:]
                                                        # Validate it's valid JSON
                                                        json.loads(widget_data)
                                                        self.logger.debug(f"🎨 Found widget data in buffer (length={len(widget_data)})")
                                                    except json.JSONDecodeError:
                                                        widget_data = None
                                        
                                        # Build widget content with data attribute if found
                                        if widget_data:
                                            # Escape single quotes in JSON to prevent attribute parsing issues
                                            escaped_data = widget_data.replace("'", "&#39;")
                                            widget_content = f"\n<widget kind='openai' data='{escaped_data}'>{widget_json}</widget>\n"
                                        else:
                                            widget_content = f"\n<widget kind='openai'>{widget_json}</widget>\n"
                                        
                                        accumulated_content += widget_content
                                        self._widget_data_buffer = ""  # Clear buffer after widget
                                        
                                        self.logger.debug(f"🎨 Rendering widget (structure length={len(widget_json)}, has_data={widget_data is not None})")
                                        
                                        yield {
                                            'id': f'chatcmpl-wf-{self.workflow_id}',
                                            'object': 'chat.completion.chunk',
                                            'created': data.get('workflow_run', {}).get('created_at', 0),
                                            'model': f'openai-workflow-{self.workflow_id}',
                                            'choices': [{
                                                'index': 0,
                                                'delta': {'content': widget_content},
                                                'finish_reason': None
                                            }]
                                        }
                                
                                # Log other events for debugging
                                elif event_type in ['workflow.started', 'workflow.node.started', 'workflow.node.finished']:
                                    self.logger.debug(f"🔄 Workflow event: {event_type}")
                                
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Failed to parse SSE data: {e}")
                                continue
        
        except httpx.HTTPStatusError as e:
            # Don't try to read response.text on streaming responses
            error_msg = f"HTTP {e.response.status_code}"
            try:
                # Try to read error body if not streaming
                if hasattr(e.response, '_content') and e.response._content is not None:
                    error_msg = f"{error_msg} - {e.response.text[:200]}"
            except Exception:
                pass
            
            self.logger.error(f"OpenAI workflow API error: {error_msg}")
            
            # Yield error message
            yield {
                'id': f'chatcmpl-wf-{self.workflow_id}',
                'object': 'chat.completion.chunk',
                'created': 0,
                'model': f'openai-workflow-{self.workflow_id}',
                'choices': [{
                    'index': 0,
                    'delta': {
                        'role': 'assistant',
                        'content': f"Error running workflow: {error_msg}"
                    },
                    'finish_reason': 'stop'
                }]
            }
        
        except Exception as e:
            self.logger.error(f"Error running OpenAI workflow: {e}", exc_info=True)
            # Yield error message
            yield {
                'id': f'chatcmpl-wf-{self.workflow_id}',
                'object': 'chat.completion.chunk',
                'created': 0,
                'model': f'openai-workflow-{self.workflow_id}',
                'choices': [{
                    'index': 0,
                    'delta': {
                        'role': 'assistant',
                        'content': f"Error running workflow: {str(e)}"
                    },
                    'finish_reason': 'stop'
                }]
            }

