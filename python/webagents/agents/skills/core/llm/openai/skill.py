"""
OpenAI Skill - WebAgents V2.0

Native integration with OpenAI API.
Uses the official openai SDK for direct API access.

Features:
- Direct OpenAI API access
- Streaming and non-streaming support
- Tool calling
- Multi-modal support (GPT-4o)
- O1/O3 Reasoning model support
- UAMP adapter for protocol conversion
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator, Union, TYPE_CHECKING
from dataclasses import dataclass

try:
    import openai
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionChunk
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool
from webagents.utils.logging import get_logger, log_skill_event, timer
from .uamp_adapter import OpenAIUAMPAdapter


@dataclass
class OpenAIModelConfig:
    """Configuration for a specific OpenAI model"""
    name: str
    max_output_tokens: int = 4096
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = False
    is_reasoning: bool = False # For o1/o3 models


class OpenAISkill(Skill):
    """
    Native OpenAI skill using the official SDK.
    """
    
    DEFAULT_MODELS = {
        "gpt-4o": OpenAIModelConfig("gpt-4o", 4096, True, True, True),
        "gpt-4o-mini": OpenAIModelConfig("gpt-4o-mini", 16384, True, True, True),
        "o1-preview": OpenAIModelConfig("o1-preview", 32768, False, False, False, True),
        "o1-mini": OpenAIModelConfig("o1-mini", 65536, False, False, False, True),
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self.config = config or {}
        
        self.model = config.get('model', 'gpt-4o') if config else 'gpt-4o'
        self.temperature = config.get('temperature', 0.7) if config else 0.7
        self.max_tokens = config.get('max_tokens') if config else None
        
        # Thinking/Reasoning config
        self.thinking_config = config.get('thinking', {})
        
        self.api_key = self._load_api_key(config)
        self.base_url = config.get('base_url')
        
        self._client: Optional[AsyncOpenAI] = None
        self.agent = None
        self.logger = get_logger('skill.llm.openai', 'init')
        self._adapter = OpenAIUAMPAdapter(model=self.model)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI SDK not available. Install with: pip install openai")

    def _load_api_key(self, config: Dict[str, Any] = None) -> str:
        if config and 'api_key' in config:
            return config['api_key']
        return os.environ.get('OPENAI_API_KEY', '')

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client

    async def initialize(self, agent: 'BaseAgent') -> None:
        self.agent = agent
        self.logger = get_logger('skill.llm.openai', agent.name)
        
        # Verify client
        self._get_client()
        
        # Register as handoff
        from webagents.agents.skills.base import Handoff
        agent.register_handoff(
            Handoff(
                target=f"openai_{self.model.replace('.', '_')}",
                description=f"OpenAI completion handler using {self.model}",
                scope="all",
                metadata={
                    'function': self.chat_completion_stream,
                    'priority': 10,
                    'is_generator': True
                }
            ),
            source="openai"
        )
        self.logger.info(f"Registered OpenAI as handoff with model: {self.model}")

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
        
        # Handle o1/reasoning models constraints (no system prompt, no tools, no streaming usually)
        # Note: o1 support is evolving, checking simplified config
        is_reasoning = target_model.startswith("o1") or target_model.startswith("o3")
        
        # Apply reasoning_effort from config if enabled (for o1/o3)
        reasoning_effort = None
        if self.thinking_config.get('enabled', False):
            # Map 'effort' (low/medium/high) to reasoning_effort
            reasoning_effort = self.thinking_config.get('effort', 'medium')
        
        params = self._prepare_params(messages, target_model, tools, stream, reasoning_effort=reasoning_effort, **kwargs)
        
        try:
            response = await client.chat.completions.create(**params)
            return self._normalize_response(response, target_model)
        except Exception as e:
            self.logger.error(f"OpenAI completion failed: {e}")
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
        
        # Apply reasoning_effort from config if enabled
        reasoning_effort = None
        if self.thinking_config.get('enabled', False):
            reasoning_effort = self.thinking_config.get('effort', 'medium')
        
        params = self._prepare_params(messages, target_model, tools, True, reasoning_effort=reasoning_effort, **kwargs)
        
        try:
            stream = await client.chat.completions.create(**params)
            chunk_index = 0
            async for chunk in stream:
                chunk_index += 1
                yield self._normalize_streaming_chunk(chunk, target_model, chunk_index)
        except Exception as e:
            self.logger.error(f"OpenAI streaming failed: {e}")
            raise

    def _prepare_params(self, messages, model, tools, stream, reasoning_effort=None, **kwargs):
        # Convert messages if needed (OpenAI expects standard format, usually pass-through)
        # Handle o1 specific adjustments
        is_reasoning = model.startswith("o1") or model.startswith("o3")
        
        params = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        if reasoning_effort and is_reasoning:
            params["reasoning_effort"] = reasoning_effort
        
        if not is_reasoning:
            params["temperature"] = kwargs.get('temperature', self.temperature)
            if self.max_tokens:
                params["max_completion_tokens" if model.startswith("o") else "max_tokens"] = self.max_tokens
            
            # Combine agent tools (functions) with config tools (e.g. web_search)
            combined_tools = []
            
            # Add config tools first
            config_tools = self.config.get('tools', []) or self.config.get('openai_tools', [])
            for tool_config in config_tools:
                tool_type = tool_config if isinstance(tool_config, str) else list(tool_config.keys())[0]
                
                # Map to OpenAI tool format
                if tool_type == 'web_search':
                    # OpenAI doesn't have a native 'web_search' tool type in the same way Google does
                    # BUT recent updates added specific tool types.
                    # Based on search results, we construct the tool payload directly.
                    # If user configures 'web_search', we add it.
                    combined_tools.append({"type": "web_search"})
                elif tool_type == 'file_search':
                    # Expect config to provide vector_store_ids if needed
                    # e.g. openai_tools: [{"file_search": {"vector_store_ids": [...]}}]
                    tool_payload = {"type": "file_search"}
                    if isinstance(tool_config, dict) and "file_search" in tool_config:
                        tool_payload.update(tool_config["file_search"])
                    combined_tools.append(tool_payload)
                elif tool_type in ['code_interpreter', 'code_execution']:
                    combined_tools.append({"type": "code_interpreter"})
                elif tool_type == 'computer_use':
                    combined_tools.append({"type": "computer_use"})
                # 'function' tools are handled below via the 'tools' arg
            
            # Add agent functions
            if tools:
                combined_tools.extend(tools)
                
            if combined_tools:
                params["tools"] = combined_tools
                
            # Passthrough other params
            for k in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop', 'response_format']:
                if k in kwargs:
                    params[k] = kwargs[k]
        else:
            # Reasoning models have strict params
            if self.max_tokens:
                params["max_completion_tokens"] = self.max_tokens
                
        return params

    def _normalize_response(self, response, model) -> Dict[str, Any]:
        # OpenAI response is already close to our internal format, but we ensure consistency
        return response.model_dump()

    def _normalize_streaming_chunk(self, chunk, model, index) -> Dict[str, Any]:
        # Normalize OpenAI chunk
        return chunk.model_dump()

