"""
OpenAI Agent Builder Skill

Runs OpenAI hosted agents/workflows and normalizes their responses
to OpenAI chat completion format for seamless handoff integration.
"""

import os
import json
import httpx
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from webagents.agents.skills import Skill
from webagents.agents.skills.base import Handoff
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
                - workflow_id: OpenAI workflow ID (e.g., wf_68e56f477fe48190ad3056eff9ad5e0200d2d26229af6c70)
                - api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
                - api_base: OpenAI API base URL (defaults to https://api.openai.com/v1)
                - version: Workflow version (optional, defaults to None = use workflow default)
        """
        super().__init__(config or {})
        
        self.workflow_id = self.config.get('workflow_id')
        if not self.workflow_id:
            raise ValueError("workflow_id is required in OpenAIAgentBuilderSkill config")
        
        self.api_key = self.config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found in config or OPENAI_API_KEY env var")
        
        self.api_base = self.config.get('api_base', 'https://api.openai.com/v1')
        self.version = self.config.get('version')  # Optional: workflow version (None = use default)
        
        self.logger = get_logger('openai_agent_builder')
        
        # State for thinking detection
        self._in_thinking_block = False
    
    async def initialize(self, agent):
        """Register as streaming handoff handler"""
        self.agent = agent
        
        # Register as handoff (streaming for real-time workflow execution)
        agent.register_handoff(
            Handoff(
                target=f"openai_workflow_{self.workflow_id}",
                description=f"OpenAI Workflow handler using {self.workflow_id}",
                scope="all",
                metadata={
                    'function': self.run_workflow_stream,
                    'priority': 10,
                    'is_generator': True  # Streaming
                }
            ),
            source="openai_agent_builder"
        )
        
        self.logger.info(f"🔧 OpenAI Agent Builder registered with workflow: {self.workflow_id}")
    
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
        
        workflow_url = f"{self.api_base}/workflows/{self.workflow_id}/run"
        
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
            "authorization": f"Bearer {self.api_key}",
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

