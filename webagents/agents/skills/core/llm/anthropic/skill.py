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
                
            if tool_params:
                params["tools"] = tool_params
            
            # Apply thinking config if enabled
            if self.thinking_config.get('enabled', False):
                budget_tokens = self.thinking_config.get('budget_tokens')
                if not budget_tokens:
                    # Map 'effort' to token budget if explicit tokens not provided
                    effort = self.thinking_config.get('effort', 'low')
                    if effort == 'low':
                        budget_tokens = 1024
                    elif effort == 'medium':
                        budget_tokens = 4096 
                    elif effort == 'high':
                        budget_tokens = 8192
                    else:
                        budget_tokens = 1024 # Default fallback

                # Ensure budget doesn't exceed max tokens (Anthropic requirement: budget < max_tokens)
                if self.max_tokens and budget_tokens >= self.max_tokens:
                    budget_tokens = int(self.max_tokens * 0.8) # Default to 80% if misconfigured
                    
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                }
                
                # Enforce tool_choice constraints when thinking is enabled
                if tool_params and "tool_choice" not in kwargs:
                    # Anthropic requires tool_choice to be 'auto' or 'none' with thinking
                    # We default to 'auto' if tools are present
                    # (params["tool_choice"] is not set by default, which implies auto)
                    pass 
                elif "tool_choice" in kwargs:
                    # If user forced a tool, we might need to warn or override, 
                    # but for now we trust the framework passes compatible args or let it fail
                    params["tool_choice"] = kwargs["tool_choice"]

            response = await client.messages.create(**params)
            return self._normalize_response(response, target_model)
            
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
                
            if tool_params:
                params["tools"] = tool_params
            
            # Apply thinking config if enabled
            if self.thinking_config.get('enabled', False):
                budget_tokens = self.thinking_config.get('budget_tokens')
                if not budget_tokens:
                    # Map 'effort' to token budget if explicit tokens not provided
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
                async for event in stream:
                    chunk_index += 1
                    normalized_chunk = self._normalize_streaming_event(event, target_model, chunk_index)
                    if normalized_chunk:
                        yield normalized_chunk
                        
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
            "id": f"anthropic-{index}", # Placeholder ID
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }
