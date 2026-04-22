"""
Fireworks AI Skill - WebAgents V2.0

Native integration with Fireworks AI API via the OpenAI-compatible SDK.
Provides access to a wide range of open-source and proprietary models.

Features:
- OpenAI-compatible API (AsyncOpenAI with custom base_url)
- Streaming and non-streaming support
- Tool calling
- Vision support (for multimodal models)
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
class FireworksModelConfig:
    """Configuration for a specific Fireworks model"""
    name: str
    max_output_tokens: int = 4096
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = False
    context_window: int = 131072


class FireworksAISkill(Skill):
    """
    Fireworks AI skill using the OpenAI SDK (compatible API).
    """

    DEFAULT_MODELS = {
        "deepseek-v3p2":             FireworksModelConfig("deepseek-v3p2", 131072, True, True, False, 163840),
        "deepseek-v3p1":             FireworksModelConfig("deepseek-v3p1", 131072, True, True, False, 163840),
        "glm-5":                     FireworksModelConfig("glm-5", 131072, True, True, False, 202752),
        "glm-4p7":                   FireworksModelConfig("glm-4p7", 131072, True, True, False, 131072),
        "kimi-k2p6":                 FireworksModelConfig("kimi-k2p6", 131072, True, True, True, 262144),
        "kimi-k2p5":                 FireworksModelConfig("kimi-k2p5", 131072, True, True, True, 262144),
        "kimi-k2-thinking":          FireworksModelConfig("kimi-k2-thinking", 131072, True, True, False, 131072),
        "kimi-k2-instruct-0905":     FireworksModelConfig("kimi-k2-instruct-0905", 131072, True, True, False, 131072),
        "minimax-m2p5":              FireworksModelConfig("minimax-m2p5", 131072, True, True, False, 196608),
        "minimax-m2p1":              FireworksModelConfig("minimax-m2p1", 131072, True, True, False, 131072),
        "gpt-oss-120b":              FireworksModelConfig("gpt-oss-120b", 131072, True, True, False, 131072),
        "gpt-oss-20b":               FireworksModelConfig("gpt-oss-20b", 131072, True, True, False, 131072),
        "llama-v3p3-70b-instruct":   FireworksModelConfig("llama-v3p3-70b-instruct", 131072, True, True, False, 131072),
        "qwen3-8b":                  FireworksModelConfig("qwen3-8b", 32768, True, True, False, 131072),
        "qwen3-vl-30b-a3b-thinking": FireworksModelConfig("qwen3-vl-30b-a3b-thinking", 131072, True, True, True, 131072),
        "qwen3-vl-30b-a3b-instruct": FireworksModelConfig("qwen3-vl-30b-a3b-instruct", 131072, True, True, True, 131072),
        "cogito-671b-v2":            FireworksModelConfig("cogito-671b-v2", 131072, True, True, False, 131072),
    }

    BASE_URL = 'https://api.fireworks.ai/inference/v1'

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self.config = config or {}

        self.model = self.config.get('model', 'deepseek-v3p2')
        self.temperature = self.config.get('temperature', 0.7)
        self.max_tokens = self.config.get('max_tokens')

        self.api_key = self._load_api_key(self.config)
        self.base_url = self.config.get('base_url', self.BASE_URL)

        self._client: Optional[AsyncOpenAI] = None
        self.agent = None
        self.logger = get_logger('skill.llm.fireworks', 'init')
        self._adapter = None

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI SDK not available (required for Fireworks AI). Install with: pip install openai")

    def _load_api_key(self, config: Dict[str, Any] = None) -> str:
        if config and 'api_key' in config:
            return config['api_key']
        return os.environ.get('FIREWORKS_API_KEY', '')

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def _get_adapter(self):
        if self._adapter is None:
            from .uamp_adapter import FireworksUAMPAdapter
            self._adapter = FireworksUAMPAdapter(model=self.model)
        return self._adapter

    async def initialize(self, agent: 'BaseAgent') -> None:
        self.agent = agent
        self.logger = get_logger('skill.llm.fireworks', agent.name)

        self._get_client()

        agent.register_handoff(
            Handoff(
                target=f"fireworks_{self.model.replace('.', '_').replace('-', '_')}",
                description=f"Fireworks AI completion handler using {self.model}",
                scope="all",
                metadata={
                    'function': self.chat_completion_stream,
                    'priority': 10,
                    'is_generator': True
                }
            ),
            source="fireworks"
        )
        self.logger.info(f"Registered Fireworks AI as handoff with model: {self.model}")

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
            'model': f'fireworks/{model}',
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

        params = self._prepare_params(messages, target_model, tools, stream, **kwargs)

        try:
            response = await client.chat.completions.create(**params)
            result = response.model_dump()

            if result.get('usage'):
                self._append_usage_record(result['usage'], target_model)

            return result
        except Exception as e:
            self.logger.error(f"Fireworks AI completion failed: {e}")
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
            async for chunk in stream:
                chunk_dict = chunk.model_dump()

                if chunk_dict.get('usage'):
                    self._append_usage_record(chunk_dict['usage'], target_model)

                yield chunk_dict
        except Exception as e:
            self.logger.error(f"Fireworks AI streaming failed: {e}")
            raise

    def _prepare_params(self, messages, model, tools, stream, **kwargs):
        # Fireworks uses the accounts/fireworks/models/ prefix for their model IDs
        # but also accepts short names
        fireworks_model = f"accounts/fireworks/models/{model}"

        params = {
            "model": fireworks_model,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get('temperature', self.temperature),
        }

        if stream:
            params["stream_options"] = {"include_usage": True}

        if self.max_tokens:
            params["max_tokens"] = self.max_tokens

        if tools:
            params["tools"] = tools

        for k in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop', 'response_format']:
            if k in kwargs:
                params[k] = kwargs[k]

        return params
