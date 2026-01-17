import os
import json
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator, Union, TYPE_CHECKING
from dataclasses import dataclass

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent

from webagents.agents.skills.base import Skill, Handoff
from webagents.agents.tools.decorators import tool
from webagents.utils.logging import get_logger, log_skill_event, timer


@dataclass
class XAIModelConfig:
    """Configuration for a specific xAI model"""
    name: str
    max_output_tokens: int = 4096
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = False


class XAISkill(Skill):
    """
    xAI (Grok) skill using the OpenAI SDK (compatible API).
    """
    
    DEFAULT_MODELS = {
        "grok-beta": XAIModelConfig("grok-beta", 131072, True, True, False), # Confirming specs
        "grok-2": XAIModelConfig("grok-2", 131072, True, True, True),
        "grok-2-mini": XAIModelConfig("grok-2-mini", 131072, True, True, True),
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self.config = config or {}
        
        self.model = config.get('model', 'grok-beta')
        self.temperature = config.get('temperature', 0.7) if config else 0.7
        self.max_tokens = config.get('max_tokens')
        
        # Thinking config
        self.thinking_config = config.get('thinking', {})
        
        self.api_key = self._load_api_key(config)
        self.base_url = config.get('base_url', 'https://api.x.ai/v1')
        
        # Configurable xAI/Grok built-in tools (if any specific ones exist, otherwise standard)
        self.xai_tools_config = config.get('xai_tools', [])
        
        self._client: Optional[AsyncOpenAI] = None
        self.agent = None
        self.logger = get_logger('skill.llm.xai', 'init')
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI SDK not available (required for xAI). Install with: pip install openai")

    def _load_api_key(self, config: Dict[str, Any] = None) -> str:
        if config and 'api_key' in config:
            return config['api_key']
        return os.environ.get('XAI_API_KEY', '')

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client

    async def initialize(self, agent: 'BaseAgent') -> None:
        self.agent = agent
        self.logger = get_logger('skill.llm.xai', agent.name)
        
        # Verify client
        self._get_client()
        
        # Register as handoff
        agent.register_handoff(
            Handoff(
                target=f"xai_{self.model.replace('.', '_').replace('-', '_')}",
                description=f"xAI (Grok) completion handler using {self.model}",
                scope="all",
                metadata={
                    'function': self.chat_completion_stream,
                    'priority': 10,
                    'is_generator': True
                }
            ),
            source="xai"
        )
        self.logger.info(f"Registered xAI as handoff with model: {self.model}")

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
        
        params = self._prepare_params(messages, target_model, tools, stream, **kwargs)
        
        try:
            response = await client.chat.completions.create(**params)
            return self._normalize_response(response, target_model)
        except Exception as e:
            self.logger.error(f"xAI completion failed: {e}")
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
        
        params = self._prepare_params(messages, target_model, tools, True, **kwargs)
        
        try:
            stream = await client.chat.completions.create(**params)
            chunk_index = 0
            async for chunk in stream:
                chunk_index += 1
                yield self._normalize_streaming_chunk(chunk, target_model, chunk_index)
        except Exception as e:
            self.logger.error(f"xAI streaming failed: {e}")
            raise

    def _prepare_params(self, messages, model, tools, stream, **kwargs):
        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get('temperature', self.temperature)
        }
        
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
            
        # Combine dynamic tools with configurable xAI built-in tools
        all_tools = []
        if tools:
            all_tools.extend(tools)
        
        # xAI specific tools logic (harmonized)
        config_tools = self.config.get('tools', []) or self.config.get('xai_tools', [])
        
        for tool_config in config_tools:
             tool_name = tool_config if isinstance(tool_config, str) else list(tool_config.keys())[0]
             
             if tool_name == 'web_search':
                all_tools.append({"type": "web_search"}) # Assuming standard name
             elif tool_name in ['code_interpreter', 'code_execution']:
                all_tools.append({"type": "code_interpreter"}) # Assuming standard name
        
        if all_tools:
            params["tools"] = all_tools
            
        # Passthrough other params
        for k in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop', 'response_format']:
            if k in kwargs:
                params[k] = kwargs[k]
                
        return params

    def _normalize_response(self, response, model) -> Dict[str, Any]:
        # xAI response is standard OpenAI format
        return response.model_dump()

    def _normalize_streaming_chunk(self, chunk, model, index) -> Dict[str, Any]:
        # xAI chunk is standard OpenAI format
        return chunk.model_dump()
