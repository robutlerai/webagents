"""
LiteLLM Skill - WebAgents V2.0

Cross-provider LLM routing using LiteLLM for unified access to:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude-3.5, Claude-3, etc.)
- XAI/Grok (grok-beta, etc.)
- Google (Gemini, etc.)
- And many more providers

Features:
- Automatic provider routing based on model names
- Streaming and non-streaming support
- Tool calling with OpenAI compatibility
- Automatic fallbacks and error handling
- Cost tracking and usage monitoring
- Model parameter optimization
"""

import os
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator, Union, TYPE_CHECKING
from dataclasses import dataclass

try:
    import litellm
    from litellm import acompletion
    LITELLM_AVAILABLE = True
except Exception:
    LITELLM_AVAILABLE = False
    litellm = None

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, hook
from webagents.utils.logging import get_logger, log_skill_event, log_tool_execution, timer


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    provider: str
    max_tokens: int = 4096
    supports_tools: bool = True
    supports_streaming: bool = True


class LiteLLMSkill(Skill):
    """
    Cross-provider LLM skill using LiteLLM for unified access
    
    Supports multiple providers with automatic routing, fallbacks,
    streaming, tool calling, and comprehensive error handling.
    """
    
    # Default model configurations
    DEFAULT_MODELS = {
        # OpenAI
        "gpt-4o": ModelConfig("gpt-4o", "openai", 4096, True, True),
        "gpt-4o-mini": ModelConfig("gpt-4o-mini", "openai", 16384, True, True),
        "gpt-4.1": ModelConfig("gpt-4.1", "openai", 4096, True, True),
        "text-embedding-3-small": ModelConfig("text-embedding-3-small", "openai", 8192, False, False),
        
        # Anthropic
        "claude-3-5-sonnet": ModelConfig("claude-3-5-sonnet", "anthropic", 8192, True, True),
        "claude-3-5-haiku": ModelConfig("claude-3-5-haiku", "anthropic", 4096, True, True),
        "claude-3-opus": ModelConfig("claude-3-opus", "anthropic", 4096, True, True),
        "claude-4-opus": ModelConfig("claude-4-opus", "anthropic", 8192, True, True),
        
        # XAI/Grok
        "xai/grok-4": ModelConfig("xai/grok-4", "xai", 8192, True, True),
        "grok-4": ModelConfig("grok-4", "xai", 8192, True, True),
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        
        # Store full configuration
        self.config = config or {}
        
        # Configuration
        self.model = config.get('model', 'gpt-4o-mini') if config else 'gpt-4o-mini'
        self.temperature = config.get('temperature', 0.7) if config else 0.7
        self.max_tokens = config.get('max_tokens') if config else None
        self.fallback_models = config.get('fallback_models', []) if config else []
        
        # API configuration
        self.api_keys = self._load_api_keys(config)
        self.model_configs = {**self.DEFAULT_MODELS}
        if config and 'custom_models' in config:
            self.model_configs.update(config['custom_models'])
        
        # Runtime state
        self.current_model = self.model
        self.error_counts = {}
        
        # Validate LiteLLM availability
        if not LITELLM_AVAILABLE:
            raise ImportError("LiteLLM not available. Install with: pip install litellm")
    
    def _load_api_keys(self, config: Dict[str, Any] = None) -> Dict[str, str]:
        """Load API keys from config and environment - CONFIG HAS PRIORITY"""
        keys = {}
        
        # Load from environment variables first
        env_keys = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'xai': 'XAI_API_KEY',
            'google': 'GOOGLE_API_KEY',
        }
        
        for provider, env_var in env_keys.items():
            if env_var in os.environ:
                keys[provider] = os.environ[env_var]
        
        # Override with config keys (config has priority)
        if config and 'api_keys' in config:
            keys.update(config['api_keys'])
        
        return keys
    
    async def initialize(self, agent: 'BaseAgent') -> None:
        """Initialize LiteLLM skill"""
        from webagents.utils.logging import get_logger, log_skill_event
        
        self.agent = agent
        self.logger = get_logger('skill.llm.litellm', agent.name)
        
        # Configure LiteLLM
        if litellm:
            # Note: API keys are now passed directly to completion calls rather than set globally
            
            # Configure base URL if provided (for proxy usage)
            if self.config and 'base_url' in self.config:
                litellm.api_base = self.config['base_url']
                os.environ['OPENAI_API_BASE'] = self.config['base_url']
                self.logger.info(f"LiteLLM configured with base URL: {self.config['base_url']}")
            
            # Configure LiteLLM settings
            litellm.set_verbose = False  # We handle logging ourselves
            litellm.drop_params = True   # Drop unsupported parameters
        
        log_skill_event(agent.name, 'litellm', 'initialized', {
            'model': self.model,
            'temperature': self.temperature,
            'available_providers': list(self.api_keys.keys()),
            'fallback_models': self.fallback_models,
            'total_models': len(self.model_configs)
        })
    
    
    
    # Core LLM functionality
    
    async def chat_completion(self, messages: List[Dict[str, Any]], 
                            model: Optional[str] = None,
                            tools: Optional[List[Dict[str, Any]]] = None,
                            stream: bool = False,
                            **kwargs: Any) -> Dict[str, Any]:
        """
        Create a chat completion using LiteLLM
        
        Args:
            messages: OpenAI-format messages
            model: Override model (defaults to skill's current model)
            tools: OpenAI-format tool definitions  
            stream: Whether to stream (handled by chat_completion_stream)
            **kwargs: Additional LLM parameters
        """
        
        if stream:
            raise ValueError("Use chat_completion_stream() for streaming responses")
        
        target_model = model or self.current_model
        
        with timer(f"chat_completion_{target_model}", self.agent.name):
            try:
                response = await self._execute_completion(
                    messages=messages,
                    model=target_model,
                    tools=tools,
                    stream=False,
                    **kwargs
                )
                # Log token usage to context.usage if available
                try:
                    usage_obj = None
                    if hasattr(response, 'usage'):
                        usage_obj = getattr(response, 'usage')
                    elif isinstance(response, dict):
                        usage_obj = response.get('usage')
                    if usage_obj:
                        prompt_tokens = int(getattr(usage_obj, 'prompt_tokens', None) or usage_obj.get('prompt_tokens') or 0)
                        completion_tokens = int(getattr(usage_obj, 'completion_tokens', None) or usage_obj.get('completion_tokens') or 0)
                        total_tokens = int(getattr(usage_obj, 'total_tokens', None) or usage_obj.get('total_tokens') or (prompt_tokens + completion_tokens))
                        self._append_usage_record(model=target_model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens, streaming=False)
                except Exception:
                    # Never fail the call on logging issues
                    pass

                return response
                
            except Exception as e:
                self.logger.error(f"Chat completion failed for {target_model}: {e}")
                
                # Try fallback models
                if self.fallback_models:
                    for fallback_model in self.fallback_models:
                        try:
                            self.logger.info(f"Trying fallback model: {fallback_model}")
                            response = await self._execute_completion(
                                messages=messages,
                                model=fallback_model,
                                tools=tools,
                                stream=False,
                                **kwargs
                            )
                            
                            return response
                            
                        except Exception as fallback_error:
                            self.logger.warning(f"Fallback {fallback_model} also failed: {fallback_error}")
                            continue
                
                # All models failed
                self._track_error(target_model)
                raise e
    
    async def chat_completion_stream(self, messages: List[Dict[str, Any]],
                                   model: Optional[str] = None,
                                   tools: Optional[List[Dict[str, Any]]] = None,
                                   **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Create a streaming chat completion using LiteLLM
        """
        
        target_model = model or self.current_model
        
        try:
            async for chunk in self._execute_completion_stream(
                messages=messages,
                model=target_model,
                tools=tools,
                **kwargs
            ):
                yield chunk
            
            # Usage logging handled via final usage chunk during streaming
            
        except Exception as e:
            self.logger.error(f"Streaming completion failed for {target_model}: {e}")
            
            # Try fallback models
            if self.fallback_models:
                for fallback_model in self.fallback_models:
                    try:
                        self.logger.info(f"Trying fallback streaming with: {fallback_model}")
                        async for chunk in self._execute_completion_stream(
                            messages=messages,
                            model=fallback_model,
                            tools=tools,
                            **kwargs
                        ):
                            yield chunk
                        
                        self._track_usage(fallback_model)
                        return
                        
                    except Exception as fallback_error:
                        self.logger.warning(f"Fallback streaming {fallback_model} failed: {fallback_error}")
                        continue
            
            # All models failed
            self._track_error(target_model)
            raise e
    
    # Private helper methods
    
    def _get_api_key_for_model(self, model: str) -> Optional[str]:
        """Get the appropriate API key based on the model provider"""
        # Determine provider from model name
        if model.startswith('azure/'):
            return self.api_keys.get('azure')
        elif model.startswith('openai/') or model in ['gpt-4', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'text-embedding-3-small']:
            return self.api_keys.get('openai')
        elif model.startswith('anthropic/') or model.startswith('claude'):
            return self.api_keys.get('anthropic')
        elif model.startswith('xai/') or model.startswith('grok') or model == 'grok-4':
            return self.api_keys.get('xai')
        elif model.startswith('google/') or model.startswith('gemini'):
            return self.api_keys.get('google')
        else:
            # Try to find a matching provider from model configs
            model_config = self.model_configs.get(model)
            if model_config:
                return self.api_keys.get(model_config.provider)
            # Fallback to default
            return self.api_keys.get('openai')
    
    async def _execute_completion(self, messages: List[Dict[str, Any]],
                                model: str,
                                tools: Optional[List[Dict[str, Any]]] = None,
                                stream: bool = False,
                                **kwargs) -> Dict[str, Any]:
        """Execute a single completion request"""
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "stream": stream,
            # Ensure usage is available when streaming is requested later
            "stream_options": {"include_usage": True} if stream else None,
        }
        
        # Add base URL if configured (for proxy support)
        if hasattr(self, 'config') and self.config and 'base_url' in self.config:
            params["api_base"] = self.config['base_url']
        
        # Add max_tokens if specified
        if self.max_tokens or 'max_tokens' in kwargs:
            params["max_tokens"] = kwargs.get('max_tokens', self.max_tokens)
        
        # Add tools if provided - most modern models support tools
        # Only skip tools for models explicitly marked as non-supporting
        model_config = self.model_configs.get(model)
        skip_tools = model_config and not model_config.supports_tools
        
        if tools is not None and tools and not skip_tools:
            params["tools"] = tools
        
        # Add other parameters
        for param in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop']:
            if param in kwargs:
                params[param] = kwargs[param]
        
        # Add API key based on model provider
        api_key = self._get_api_key_for_model(model)
        if api_key:
            params["api_key"] = api_key
        
        self.logger.debug(f"Executing completion with model {model}")
        self.logger.debug(f"Parameters: {params}")
        
        # Validate parameters before calling LiteLLM
        if not messages or not isinstance(messages, list):
            raise ValueError(f"Messages must be a non-empty list, got: {type(messages)}")
        
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dict, got: {type(msg)}")
            if 'role' not in msg:
                raise ValueError(f"Message {i} missing required 'role' field")
        
        try:
            # Execute completion
            response = await acompletion(**params)
            
            # Convert LiteLLM response to our format
            return self._normalize_response(response, model)
        except Exception as e:
            self.logger.error(f"LiteLLM completion failed with params: {params}")
            self.logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            raise
    
    async def _execute_completion_stream(self, messages: List[Dict[str, Any]],
                                       model: str,
                                       tools: Optional[List[Dict[str, Any]]] = None,
                                       **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a streaming completion request"""
        
        # Prepare parameters (same as non-streaming)
        params = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "stream": True,
            # Include a final usage chunk before [DONE] per LiteLLM docs
            "stream_options": {"include_usage": True},
        }
        
        # Add base URL if configured (for proxy support)
        if self.config and 'base_url' in self.config:
            params["api_base"] = self.config['base_url']
        
        if self.max_tokens or 'max_tokens' in kwargs:
            params["max_tokens"] = kwargs.get('max_tokens', self.max_tokens)
        
                # Always pass tools if provided - most modern models support tools
        # Only skip tools for models explicitly marked as non-supporting
        model_config = self.model_configs.get(model)
        skip_tools = model_config and not model_config.supports_tools
        
        if tools is not None and tools and not skip_tools:
            params["tools"] = tools

        for param in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop']:
            if param in kwargs:
                params[param] = kwargs[param]
        
        # Add API key based on model provider
        api_key = self._get_api_key_for_model(model)
        if api_key:
            params["api_key"] = api_key
        
        self.logger.debug(f"Executing streaming completion with model {model}")
        
        # Execute streaming completion
        stream = await acompletion(**params)
        
        async for chunk in stream:
            # Normalize and yield chunk
            normalized_chunk = self._normalize_streaming_chunk(chunk, model)
            
            # If LiteLLM sent a final usage chunk, log tokens to context.usage
            try:
                usage = normalized_chunk.get('usage') if isinstance(normalized_chunk, dict) else None
                is_final_usage_chunk = (
                    usage
                    and isinstance(usage, dict)
                    and (not normalized_chunk.get('choices'))
                )
                if is_final_usage_chunk:
                    prompt_tokens = int(usage.get('prompt_tokens') or 0)
                    completion_tokens = int(usage.get('completion_tokens') or 0)
                    total_tokens = int(usage.get('total_tokens') or (prompt_tokens + completion_tokens))
                    self._append_usage_record(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens, streaming=True)
            except Exception:
                # Never break streaming on usage logging
                pass
            yield normalized_chunk
    
    def _normalize_response(self, response: Any, model: str) -> Dict[str, Any]:
        """Normalize LiteLLM response to OpenAI format"""
        
        # LiteLLM already returns OpenAI-compatible format
        # Just ensure model name is correct
        if hasattr(response, 'model'):
            response.model = model
        elif isinstance(response, dict) and 'model' in response:
            response['model'] = model
        
        return response
    
    def _normalize_streaming_chunk(self, chunk: Any, model: str) -> Dict[str, Any]:
        """Normalize LiteLLM streaming chunk to OpenAI format"""
        
        # Convert chunk to dictionary if it's not already
        if hasattr(chunk, 'model_dump'):
            # Pydantic v2
            chunk_dict = chunk.model_dump()
        elif hasattr(chunk, 'dict'):
            # Pydantic v1
            chunk_dict = chunk.dict()
        elif hasattr(chunk, '__dict__'):
            # Generic object with attributes
            chunk_dict = vars(chunk)
        elif isinstance(chunk, dict):
            # Already a dictionary
            chunk_dict = chunk.copy()
        else:
            # Try to convert to dict
            try:
                chunk_dict = dict(chunk)
            except:
                # Fallback - return as-is and hope for the best
                return chunk
        
        # Ensure model name is correct
        chunk_dict['model'] = model
        
        return chunk_dict
    
    def _append_usage_record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        streaming: bool,
    ) -> None:
        """Append a normalized usage record to context.usage"""
        try:
            from webagents.server.context.context_vars import get_context
            context = get_context()
            if not context or not hasattr(context, 'usage'):
                return
            context.usage.append({
                'type': 'llm',
                'skill': 'litellm',
                'model': model,
                'prompt_tokens': int(prompt_tokens or 0),
                'completion_tokens': int(completion_tokens or 0),
                'total_tokens': int(total_tokens or 0),
                'streaming': bool(streaming),
                'timestamp': time.time(),
            })
        except Exception:
            # Do not raise from logging
            return
    
    def _track_error(self, model: str):
        """Track model error statistics"""
        if model not in self.error_counts:
            self.error_counts[model] = 0
        self.error_counts[model] += 1
        
        self.logger.warning(f"Model error tracked: {model} ({self.error_counts[model]} total errors)")
    
    # Compatibility methods for BaseAgent integration
    
    def get_dependencies(self) -> List[str]:
        """Get skill dependencies"""
        return []  # LiteLLM skill is self-contained
    
    async def query_litellm(self, prompt: str, model: Optional[str] = None, **kwargs: Any) -> str:
        """Simple query interface for compatibility"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.chat_completion(messages, model=model, **kwargs)
        
        if isinstance(response, dict) and 'choices' in response:
            return response['choices'][0]['message']['content']
        
        return str(response)
    
    async def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embeddings (placeholder for V2.1)"""
        # This would use LiteLLM's embedding support in V2.1
        self.logger.info("Embedding generation requested - will be implemented in V2.1")
        return [0.0] * 1536  # Placeholder embedding 