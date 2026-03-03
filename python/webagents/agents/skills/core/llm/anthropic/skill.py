"""
Anthropic Skill - WebAgents V2.0

Native integration with Anthropic Claude API.
Uses the official anthropic SDK for direct API access.

Features:
- Direct Claude API access
- Streaming and non-streaming support
- Tool calling
- Multi-modal support (images, PDFs)
- Extended thinking support
- UAMP adapter for protocol conversion
"""

import os
import json
import asyncio
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Union, TYPE_CHECKING
from dataclasses import dataclass

try:
    import anthropic
    from anthropic import AsyncAnthropic
    from anthropic.types import Message, TextBlock, ToolUseBlock
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = None  # type: ignore

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent

from webagents.agents.skills.base import Skill, Handoff
from webagents.agents.tools.decorators import tool
from webagents.utils.logging import get_logger, log_skill_event, timer
from .uamp_adapter import AnthropicUAMPAdapter


@dataclass
class AnthropicModelConfig:
    """Configuration for a specific Anthropic model"""
    name: str
    max_output_tokens: int = 4096
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = True
    supports_computer_use: bool = False


class AnthropicSkill(Skill):
    """
    Native Anthropic skill using the official SDK.
    """
    
    DEFAULT_MODELS = {
        "claude-3-5-sonnet-20241022": AnthropicModelConfig("claude-3-5-sonnet-20241022", 8192, True, True, True, True),
        "claude-3-5-haiku-20241022": AnthropicModelConfig("claude-3-5-haiku-20241022", 8192, True, True, True),
        "claude-3-opus-20240229": AnthropicModelConfig("claude-3-opus-20240229", 4096, True, True, True),
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self.config = config or {}
        
        self.model = config.get('model', 'claude-3-5-sonnet-20241022')
        self.temperature = config.get('temperature', 0.7) if config else 0.7
        self.max_tokens = config.get('max_tokens', 4096)
        
        self.api_key = self._load_api_key(config)
        self.base_url = config.get('base_url')
        
        # Configurable Anthropic built-in tools (e.g., computer use)
        self.anthropic_tools_config = config.get('anthropic_tools', [])
        
        # Thinking config
        self.thinking_config = config.get('thinking', {})
        
        self._client: Optional[AsyncAnthropic] = None
        self.agent = None
        self.logger = get_logger('skill.llm.anthropic', 'init')
        self._adapter = AnthropicUAMPAdapter(model=self.model)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic SDK not available. Install with: pip install anthropic")

    def _load_api_key(self, config: Dict[str, Any] = None) -> str:
        if config and 'api_key' in config:
            return config['api_key']
        return os.environ.get('ANTHROPIC_API_KEY', '')

    def _get_client(self) -> "AsyncAnthropic":
        if self._client is None:
            self._client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client

    async def initialize(self, agent: 'BaseAgent') -> None:
        self.agent = agent
        self.logger = get_logger('skill.llm.anthropic', agent.name)
        
        # Verify client
        self._get_client()
        
        # Register as handoff
        agent.register_handoff(
            Handoff(
                target=f"anthropic_{self.model.replace('.', '_').replace('-', '_')}",
                description=f"Anthropic completion handler using {self.model}",
                scope="all",
                metadata={
                    'function': self.chat_completion_stream,
                    'priority': 10,
                    'is_generator': True
                }
            ),
            source="anthropic"
        )
        self.logger.info(f"Registered Anthropic as handoff with model: {self.model}")

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        if stream:
            raise ValueError("Use chat_completion_stream() for streaming")
            
        target_model = model or self.model
        client = self._get_client()
        
        system_prompt, anthropic_messages = self._convert_messages(messages)
        tool_params = self._prepare_tools(tools)
        
        try:
            params = {
                "model": target_model,
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
                "temperature": kwargs.get('temperature', self.temperature),
            }
            
            if system_prompt:
                params["system"] = system_prompt
            
            response_format = kwargs.get('response_format')
            tool_params, is_forced_tool = self._apply_response_format(
                response_format, params, tool_params
            )
                
            if tool_params:
                params["tools"] = tool_params
            
            # Apply thinking config if enabled
            if self.thinking_config.get('enabled', False):
                budget_tokens = self.thinking_config.get('budget_tokens')
                if not budget_tokens:
                    effort = self.thinking_config.get('effort', 'low')
                    if effort == 'low':
                        budget_tokens = 1024
                    elif effort == 'medium':
                        budget_tokens = 4096 
                    elif effort == 'high':
                        budget_tokens = 8192
                    else:
                        budget_tokens = 1024

                if self.max_tokens and budget_tokens >= self.max_tokens:
                    budget_tokens = int(self.max_tokens * 0.8)
                    
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                }
                
                if tool_params and "tool_choice" not in kwargs and not is_forced_tool:
                    pass 
                elif "tool_choice" in kwargs:
                    params["tool_choice"] = kwargs["tool_choice"]

            response = await client.messages.create(**params)
            normalized = self._normalize_response(response, target_model)
            
            if is_forced_tool:
                normalized = self._unwrap_forced_tool_response(normalized)
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Anthropic completion failed: {e}")
            raise

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        
        target_model = model or self.model
        client = self._get_client()
        
        system_prompt, anthropic_messages = self._convert_messages(messages)
        tool_params = self._prepare_tools(tools)
        
        try:
            params = {
                "model": target_model,
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
                "temperature": kwargs.get('temperature', self.temperature),
                "stream": True,
            }
            
            if system_prompt:
                params["system"] = system_prompt
            
            response_format = kwargs.get('response_format')
            tool_params, is_forced_tool = self._apply_response_format(
                response_format, params, tool_params
            )
                
            if tool_params:
                params["tools"] = tool_params
            
            # Apply thinking config if enabled
            if self.thinking_config.get('enabled', False):
                budget_tokens = self.thinking_config.get('budget_tokens')
                if not budget_tokens:
                    effort = self.thinking_config.get('effort', 'low')
                    if effort == 'low':
                        budget_tokens = 1024
                    elif effort == 'medium':
                        budget_tokens = 4096 
                    elif effort == 'high':
                        budget_tokens = 8192
                    else:
                        budget_tokens = 1024

                if self.max_tokens and budget_tokens >= self.max_tokens:
                    budget_tokens = int(self.max_tokens * 0.8)
                    
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                }
            
            # Streaming implementation
            async with client.messages.stream(**params) as stream:
                chunk_index = 0
                forced_tool_args = [] if is_forced_tool else None
                async for event in stream:
                    chunk_index += 1
                    normalized_chunk = self._normalize_streaming_event(event, target_model, chunk_index)
                    if normalized_chunk:
                        if is_forced_tool:
                            self._collect_forced_tool_args(normalized_chunk, forced_tool_args)
                        else:
                            yield normalized_chunk
                
                if is_forced_tool and forced_tool_args:
                    yield {
                        "id": f"anthropic-structured",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": target_model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": "".join(forced_tool_args)},
                            "finish_reason": "stop"
                        }]
                    }
                        
        except Exception as e:
            self.logger.error(f"Anthropic streaming failed: {e}")
            raise

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert standard messages to Anthropic format"""
        anthropic_messages = []
        system_prompt = None
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "system":
                # Anthropic uses a top-level system parameter, not a message
                if system_prompt:
                    system_prompt += "\n" + content
                else:
                    system_prompt = content
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                # Handle tool calls in assistant messages if present (basic text for now)
                # TODO: Handle previous tool calls correctly if needed for history
                anthropic_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Handle tool results
                # Anthropic expects tool results in a specific format within user messages or separate blocks
                # Simplification: Append to user message or treat as user message with tool result content
                # This needs robust implementation for full tool loop support
                anthropic_messages.append({
                    "role": "user", 
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", "unknown"),
                            "content": content
                        }
                    ]
                })
                
        return system_prompt, anthropic_messages

    def _apply_response_format(
        self,
        response_format: Optional[Dict[str, Any]],
        params: Dict[str, Any],
        tool_params: Optional[List[Dict[str, Any]]]
    ) -> tuple[Optional[List[Dict[str, Any]]], bool]:
        """Apply response_format to Anthropic params. Returns (updated tool_params, is_forced_tool)."""
        if not response_format or not isinstance(response_format, dict):
            return tool_params, False
        
        rf_type = response_format.get('type', '')
        
        if rf_type == 'json_schema':
            json_schema = response_format.get('json_schema', {})
            schema_name = json_schema.get('name', 'structured_output')
            schema = json_schema.get('schema', {})
            
            structured_tool = {
                "name": schema_name,
                "description": f"Return structured output matching the {schema_name} schema.",
                "input_schema": schema
            }
            
            if tool_params is None:
                tool_params = []
            tool_params.append(structured_tool)
            
            params["tool_choice"] = {"type": "tool", "name": schema_name}
            return tool_params, True
        
        elif rf_type == 'json_object':
            system = params.get("system", "")
            suffix = "\n\nYou must respond with valid JSON only. No markdown, no explanation."
            params["system"] = (system + suffix) if system else suffix.strip()
            return tool_params, False
        
        return tool_params, False

    def _prepare_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Prepare tools for Anthropic API"""
        final_tools = []
        
        # 1. Dynamic tools from agent
        if tools:
            for tool in tools:
                # Anthropic tool format: {name, description, input_schema}
                # OpenAI format: {type: function, function: {name, description, parameters}}
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    final_tools.append({
                        "name": func.get("name"),
                        "description": func.get("description"),
                        "input_schema": func.get("parameters")
                    })
        
        # 2. Configured built-in tools (harmonized)
        config_tools = self.config.get('tools', []) or self.config.get('anthropic_tools', [])
        
        for tool_config in config_tools:
            tool_name = tool_config if isinstance(tool_config, str) else list(tool_config.keys())[0]
            
            if tool_name == "computer_use":
                # https://docs.anthropic.com/en/docs/build-with-claude/computer-use
                final_tools.append({
                    "type": "computer_20241022",
                    "name": "computer",
                    "display_width_px": 1024,
                    "display_height_px": 768,
                    "display_number": 0,
                })
            elif tool_name == "bash":
                final_tools.append({
                    "type": "bash_20241022",
                    "name": "bash"
                })
            elif tool_name == "text_editor":
                final_tools.append({
                    "type": "text_editor_20241022",
                    "name": "str_replace_editor"
                })
            elif tool_name in ["code_execution", "code_interpreter"]:
                 # Map code execution to bash for Anthropic (closest equivalent)
                 # Or warn if no direct mapping desired
                 final_tools.append({
                    "type": "bash_20241022",
                    "name": "bash"
                })
                
        return final_tools if final_tools else None

    def _normalize_response(self, response: Any, model: str) -> Dict[str, Any]:
        """Normalize Anthropic response to OpenAI-like format"""
        content = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
        
        choices = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls if tool_calls else None
            },
            "finish_reason": response.stop_reason
        }]
        
        return {
            "id": response.id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        }

    def _normalize_streaming_event(self, event: Any, model: str, index: int) -> Optional[Dict[str, Any]]:
        """Normalize Anthropic streaming event to OpenAI chunk format"""
        # Anthropic streaming events: content_block_delta, content_block_start, etc.
        
        delta = {}
        finish_reason = None
        
        if event.type == "content_block_start":
            if event.content_block.type == "tool_use":
                delta["tool_calls"] = [{
                    "index": 0,
                    "id": event.content_block.id,
                    "type": "function",
                    "function": {
                        "name": event.content_block.name,
                        "arguments": ""
                    }
                }]
                
        elif event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                delta["content"] = event.delta.text
            elif event.delta.type == "input_json_delta":
                delta["tool_calls"] = [{
                    "index": 0,
                    "function": {
                        "arguments": event.delta.partial_json
                    }
                }]
                
        elif event.type == "message_delta":
            finish_reason = event.delta.stop_reason
            
        elif event.type == "message_stop":
            finish_reason = "stop"
            
        if not delta and not finish_reason:
            return None
            
        return {
            "id": f"anthropic-{index}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }

    def _unwrap_forced_tool_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Unwrap forced tool_use response into text content (for json_schema response_format)."""
        choices = response.get("choices", [])
        if not choices:
            return response
        
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls")
        
        if tool_calls:
            first_call = tool_calls[0]
            arguments_json = first_call.get("function", {}).get("arguments", "{}")
            message["content"] = arguments_json
            message["tool_calls"] = None
        
        return response

    def _collect_forced_tool_args(self, chunk: Dict[str, Any], args_buffer: List[str]) -> None:
        """Collect tool call argument fragments from streaming chunks for forced-tool unwrapping."""
        choices = chunk.get("choices", [])
        if not choices:
            return
        
        delta = choices[0].get("delta", {})
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                args_fragment = func.get("arguments", "")
                if args_fragment:
                    args_buffer.append(args_fragment)
