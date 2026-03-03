"""
xAI (Grok) Skill - WebAgents V2.0

Native integration with xAI API via the OpenAI-compatible SDK.

Features:
- Direct xAI API access (OpenAI-compatible)
- Streaming and non-streaming support
- Tool calling
- Vision support (grok-4 models)
- Reasoning model support (grok-3-mini, grok-4)
- Usage tracking for server-side billing
- UAMP adapter for protocol conversion
"""

import os
import json
from typing import Dict, Any, List, Optional, AsyncGenerator, TYPE_CHECKING
from dataclasses import dataclass

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent

from webagents.agents.skills.base import Skill, Handoff
from webagents.utils.logging import get_logger


@dataclass
class XAIModelConfig:
    """Configuration for a specific xAI model"""
    name: str
    max_output_tokens: int = 4096
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = False
    is_reasoning: bool = False
    context_window: int = 131072


class XAISkill(Skill):
    """
    xAI (Grok) skill using the OpenAI SDK (compatible API).
    """

    DEFAULT_MODELS = {
        "grok-3":                  XAIModelConfig("grok-3", 131072, True, True, False, False, 131072),
        "grok-3-mini":             XAIModelConfig("grok-3-mini", 131072, True, True, False, True, 131072),
        "grok-4-0709":             XAIModelConfig("grok-4-0709", 131072, True, True, True, True, 256000),
        "grok-4-fast-reasoning":   XAIModelConfig("grok-4-fast-reasoning", 131072, True, True, True, True, 2000000),
        "grok-4-fast-non-reasoning": XAIModelConfig("grok-4-fast-non-reasoning", 131072, True, True, True, False, 2000000),
        "grok-code-fast-1":        XAIModelConfig("grok-code-fast-1", 131072, True, True, False, False, 256000),
    }

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self.config = config or {}

        self.model = self.config.get('model', 'grok-3')
        self.temperature = self.config.get('temperature', 0.7)
        self.max_tokens = self.config.get('max_tokens')

        self.thinking_config = self.config.get('thinking', {})

        self.api_key = self._load_api_key(self.config)
        self.base_url = self.config.get('base_url', 'https://api.x.ai/v1')

        self.xai_tools_config = self.config.get('xai_tools', [])

        self._client: Optional[AsyncOpenAI] = None
        self.agent = None
        self.logger = get_logger('skill.llm.xai', 'init')
        self._adapter = None

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

    def _get_adapter(self):
        if self._adapter is None:
            from .uamp_adapter import XAIUAMPAdapter
            self._adapter = XAIUAMPAdapter(model=self.model)
        return self._adapter

    async def initialize(self, agent: 'BaseAgent') -> None:
        self.agent = agent
        self.logger = get_logger('skill.llm.xai', agent.name)

        self._get_client()

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

    def _is_reasoning_model(self, model: str) -> bool:
        cfg = self.DEFAULT_MODELS.get(model)
        if cfg:
            return cfg.is_reasoning
        return 'reasoning' in model or model == 'grok-3-mini'

    def _append_usage_record(self, usage_data: dict, model: str) -> None:
        """Append an LLM usage record to context for PaymentSkill to consume."""
        if not self.agent or not hasattr(self.agent, 'context'):
            return
        context = self.agent.context
        if not hasattr(context, 'usage'):
            context.usage = []

        prompt_tokens = usage_data.get('prompt_tokens', 0)
        completion_tokens = usage_data.get('completion_tokens', 0)

        record = {
            'type': 'llm',
            'model': f'xai/{model}',
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
        }

        cached_details = usage_data.get('prompt_tokens_details', {})
        if cached_details and cached_details.get('cached_tokens'):
            record['cached_read_tokens'] = cached_details['cached_tokens']

        context.usage.append(record)

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

        is_reasoning = self._is_reasoning_model(target_model)
        reasoning_effort = None
        if self.thinking_config.get('enabled', False) and is_reasoning:
            reasoning_effort = self.thinking_config.get('effort', 'medium')

        params = self._prepare_params(messages, target_model, tools, stream, reasoning_effort=reasoning_effort, **kwargs)

        try:
            response = await client.chat.completions.create(**params)
            result = response.model_dump()

            if result.get('usage'):
                self._append_usage_record(result['usage'], target_model)

            return result
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

        is_reasoning = self._is_reasoning_model(target_model)
        reasoning_effort = None
        if self.thinking_config.get('enabled', False) and is_reasoning:
            reasoning_effort = self.thinking_config.get('effort', 'medium')

        params = self._prepare_params(messages, target_model, tools, True, reasoning_effort=reasoning_effort, **kwargs)

        try:
            stream = await client.chat.completions.create(**params)
            chunk_index = 0
            async for chunk in stream:
                chunk_index += 1
                chunk_dict = chunk.model_dump()

                if chunk_dict.get('usage'):
                    self._append_usage_record(chunk_dict['usage'], target_model)

                yield chunk_dict
        except Exception as e:
            self.logger.error(f"xAI streaming failed: {e}")
            raise

    def _prepare_params(self, messages, model, tools, stream, reasoning_effort=None, **kwargs):
        is_reasoning = self._is_reasoning_model(model)

        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        if stream:
            params["stream_options"] = {"include_usage": True}

        if reasoning_effort and is_reasoning:
            params["reasoning_effort"] = reasoning_effort

        if not is_reasoning:
            params["temperature"] = kwargs.get('temperature', self.temperature)
            if self.max_tokens:
                params["max_tokens"] = self.max_tokens

            all_tools = []
            if tools:
                all_tools.extend(tools)

            config_tools = self.config.get('tools', []) or self.config.get('xai_tools', [])
            for tool_config in config_tools:
                tool_name = tool_config if isinstance(tool_config, str) else list(tool_config.keys())[0]
                if tool_name == 'web_search':
                    all_tools.append({"type": "web_search"})
                elif tool_name in ['code_interpreter', 'code_execution']:
                    all_tools.append({"type": "code_interpreter"})

            if all_tools:
                params["tools"] = all_tools
        else:
            if self.max_tokens:
                params["max_completion_tokens"] = self.max_tokens

        for k in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop', 'response_format']:
            if k in kwargs:
                params[k] = kwargs[k]

        return params

    def _normalize_response(self, response, model) -> Dict[str, Any]:
        return response.model_dump()

    def _normalize_streaming_chunk(self, chunk, model, index) -> Dict[str, Any]:
        return chunk.model_dump()
