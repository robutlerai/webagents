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
import base64
import uuid
import hashlib
import tempfile
from typing import Dict, Any, List, Optional, AsyncGenerator, Union, TYPE_CHECKING
import re
from dataclasses import dataclass

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

try:
    from webagents.agents.skills.robutler.payments import pricing
    PRICING_AVAILABLE = True
except ImportError:
    # Fallback: create a no-op decorator if pricing is not available
    def pricing(**kwargs):
        def decorator(func):
            return func
        return decorator
    PRICING_AVAILABLE = False


try:
    import litellm
    from litellm import acompletion, token_counter, register_model
    LITELLM_AVAILABLE = True
except Exception:
    LITELLM_AVAILABLE = False
    litellm = None
    token_counter = None
    register_model = None

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
        
        # Google Vertex AI (Gemini)
        "vertex_ai/gemini-2.5-pro": ModelConfig("vertex_ai/gemini-2.5-pro", "google", 8192, True, True),
        "vertex_ai/gemini-2.5-flash": ModelConfig("vertex_ai/gemini-2.5-flash", "google", 8192, True, True),
        "vertex_ai/gemini-2.5-flash-image": ModelConfig("vertex_ai/gemini-2.5-flash-image", "google", 8192, True, True),
        "gemini-2.5-pro": ModelConfig("gemini-2.5-pro", "google", 8192, True, True),
        "gemini-2.5-flash": ModelConfig("gemini-2.5-flash", "google", 8192, True, True),
        "gemini-2.5-flash-image": ModelConfig("gemini-2.5-flash-image", "google", 8192, True, True),
        "gemini-pro": ModelConfig("gemini-pro", "google", 8192, True, True),
        "gemini-flash": ModelConfig("gemini-flash", "google", 8192, True, True),
        "gemini-image-preview": ModelConfig("gemini-image-preview", "google", 8192, True, True),
        "gemini-flash-image": ModelConfig("gemini-flash-image", "google", 8192, True, True),
        
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
        self.custom_llm_provider = config.get('custom_llm_provider') if config else None
        self.disable_streaming = bool(config.get('disable_streaming')) if config else False
        
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
        
        # Register Gemini 2.5 experimental models pricing
        # Official pricing from https://ai.google.dev/gemini-api/docs/models
        # IMPORTANT: Register both with and without vertex_ai/ prefix for compatibility
        if LITELLM_AVAILABLE and register_model:
            try:
                gemini_models = {
                    # Gemini Flash - base model (alias for gemini-2.5-flash-thinking)
                    "gemini-flash": {
                        "max_tokens": 65535,
                        "max_input_tokens": 1048576,
                        "max_output_tokens": 65535,
                        "input_cost_per_token": 0.0000003,      # $0.30 per 1M tokens
                        "output_cost_per_token": 0.0000025,     # $2.50 per 1M tokens
                        "cache_read_input_token_cost": 0.000000075,  # $0.075 per 1M cached tokens
                        "litellm_provider": "vertex_ai",
                        "mode": "chat",
                        "supports_function_calling": True,
                        "supports_vision": True
                    },
                    # Gemini 2.5 Flash Thinking - standard reasoning model
                    "gemini-2.5-flash-thinking": {
                        "max_tokens": 65535,
                        "max_input_tokens": 1048576,
                        "max_output_tokens": 65535,
                        "input_cost_per_token": 0.0000003,      # $0.30 per 1M tokens
                        "output_cost_per_token": 0.0000025,     # $2.50 per 1M tokens
                        "cache_read_input_token_cost": 0.000000075,  # $0.075 per 1M cached tokens
                        "litellm_provider": "vertex_ai",
                        "mode": "chat",
                        "supports_function_calling": True,
                        "supports_vision": True
                    },
                    # Gemini 2.5 Flash Image Preview - experimental image model (more expensive output)
                    "gemini-2.5-flash-image": {
                        "max_tokens": 65535,
                        "max_input_tokens": 1048576,
                        "max_output_tokens": 65535,
                        "input_cost_per_token": 0.0000003,      # $0.30 per 1M tokens
                        "output_cost_per_token": 0.00003,       # $30 per 1M tokens (image model premium)
                        "cache_read_input_token_cost": 0.000000075,  # $0.075 per 1M cached tokens
                        "litellm_provider": "vertex_ai",
                        "mode": "chat",
                        "supports_function_calling": True,
                        "supports_vision": True
                    },
                    # Alias for gemini-flash-image (same pricing as standard flash)
                    "gemini-flash-image": {
                        "max_tokens": 65535,
                        "max_input_tokens": 1048576,
                        "max_output_tokens": 65535,
                        "input_cost_per_token": 0.0000003,      # $0.30 per 1M tokens
                        "output_cost_per_token": 0.0000025,     # $2.50 per 1M tokens
                        "cache_read_input_token_cost": 0.000000075,  # $0.075 per 1M cached tokens
                        "litellm_provider": "vertex_ai",
                        "mode": "chat",
                        "supports_function_calling": True,
                        "supports_vision": True
                    }
                }
                
                # Register models with and without vertex_ai/ prefix
                models_to_register = {}
                for model_name, config in gemini_models.items():
                    models_to_register[model_name] = config
                    models_to_register[f"vertex_ai/{model_name}"] = config.copy()
                
                register_model(models_to_register)
            except Exception:
                # Silent fail - not critical
                pass
    
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
        """Initialize LiteLLM skill and register as handoff"""
        from webagents.utils.logging import get_logger, log_skill_event
        from webagents.agents.skills.base import Handoff
        
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
        
        # Register as handoff (completion handler)
        # Priority=10 (high priority - likely to be the default for local LLMs)
        # NOTE: We register the STREAMING function so it works in both modes:
        # - Streaming: Returns generator directly
        # - Non-streaming: Agent consumes generator and reconstructs response
        agent.register_handoff(
            Handoff(
                target=f"litellm_{self.model.replace('/', '_')}",
                description=f"LiteLLM completion handler using {self.model}",
                scope="all",
                metadata={
                    'function': self.chat_completion_stream,
                    'priority': 10,
                    'is_generator': True  # chat_completion_stream is async generator
                }
            ),
            source="litellm"
        )
        
        self.logger.info(f"📨 Registered LiteLLM as handoff with model: {self.model}")
        
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
        # If streaming is disabled for this skill, fallback to non-streaming and yield once
        if self.disable_streaming:
            non_stream_response = await self.chat_completion(messages, model=model, tools=tools, stream=False, **kwargs)
            # Normalize into a single streaming-style chunk
            normalized = self._normalize_response(non_stream_response, model or self.current_model)
            yield normalized
            return

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
        elif model.startswith('google/') or model.startswith('gemini') or model.startswith('vertex_ai/'):
            return self.api_keys.get('google')
        else:
            # Try to find a matching provider from model configs
            model_config = self.model_configs.get(model)
            if model_config:
                return self.api_keys.get(model_config.provider)
            # Fallback to default
            return self.api_keys.get('openai')
    
    
    async def _upload_image_to_content_api(self, image_base64_url: str, model: str) -> str:
        """
        Upload base64 image data to content API and return a public URL.
        Similar to openlicense skill approach.
        """
        if not HTTPX_AVAILABLE:
            self.logger.warning("httpx not available, cannot upload image to content API")
            return image_base64_url
            
        try:
            # Extract base64 data from data URL
            if not image_base64_url.startswith('data:image/'):
                self.logger.warning(f"Invalid image URL format: {image_base64_url[:50]}...")
                return image_base64_url  # Return as-is if not a data URL
            
            # Parse the data URL: data:image/png;base64,<base64_data>
            header, base64_data = image_base64_url.split(',', 1)
            image_format = 'png'  # Default to PNG
            
            # Extract format from header if available
            if 'image/' in header:
                try:
                    format_part = header.split('image/')[1].split(';')[0]
                    if format_part in ['png', 'jpeg', 'jpg', 'webp']:
                        image_format = format_part
                except:
                    pass  # Use default PNG
            
            # Decode base64 data
            image_data = base64.b64decode(base64_data)
            self.logger.debug(f"Decoded image data: {len(image_data)} bytes, format: {image_format}")
            
            # Generate a short filename
            short_id = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]
            filename = f"gemini_{short_id}.{image_format}"
            
            # Get portal URL from environment (same as openlicense skill)
            portal_url = os.getenv("ROBUTLER_INTERNAL_API_URL", "https://robutler.ai")
            upload_url = f"{portal_url}/api/content"
            
            # Prepare metadata for upload
            description = f"AI-generated image from {model}"
            tags = ['ai-generated', 'gemini', 'litellm']
            
            # Create form data for upload
            files = {
                'file': (filename, image_data, f'image/{image_format}')
            }
            
            data = {
                'description': description,
                'tags': ','.join(tags),
                'userId': 'gemini-agent',  # Store under agent account like openlicense
                'visibility': 'public'
            }
            
            # Get API key from context (similar to openlicense approach)
            try:
                from webagents.server.context.context_vars import get_context
                context = get_context()
                api_key = None
                
                if context:
                    # Try multiple possible key names
                    api_key = (context.get("api_key") or 
                              context.get("robutler_api_key") or 
                              context.get("agent_api_key") or
                              getattr(context, 'api_key', None))
                    
                    # Also try to get from identity info or token info
                    if not api_key:
                        identity_info = context.get("identity_info")
                        if identity_info and isinstance(identity_info, dict):
                            api_key = identity_info.get("api_key")
                        
                        if not api_key:
                            token_info = context.get("token_info")
                            if token_info and isinstance(token_info, dict):
                                api_key = token_info.get("api_key")
                
                # Fallback to skill config
                if not api_key and hasattr(self, 'config') and self.config:
                    api_key = self.config.get('robutler_api_key')
                
                # Try environment variables as last resort
                if not api_key:
                    api_key = os.getenv('ROBUTLER_API_KEY') or os.getenv('API_KEY')
                
                if not api_key:
                    self.logger.warning("No API key found for content upload, trying without authentication")
                    # Don't return early - try the upload anyway, it might work without auth in dev mode
                
                headers = {}
                if api_key:
                    headers['Authorization'] = f'Bearer {api_key}'
                
                # Upload the image
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(upload_url, files=files, data=data, headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        public_url = result.get('publicUrl')
                        
                        if public_url:
                            # Rewrite URL to chat server if needed (like openlicense)
                            chat_base = (os.getenv('ROBUTLER_CHAT_URL') or 'http://localhost:3001').rstrip('/')
                            if public_url.startswith('/api/content/public'):
                                public_url = f"{chat_base}{public_url}"
                            
                            self.logger.info(f"Successfully uploaded image: {filename} -> {public_url}")
                            return public_url
                        else:
                            self.logger.error(f"Upload successful but no publicUrl in response: {result}")
                            return image_base64_url
                    else:
                        self.logger.error(f"Failed to upload image: {response.status_code} - {response.text}")
                        return image_base64_url
                        
            except Exception as e:
                self.logger.error(f"Error during image upload: {e}")
                return image_base64_url
                
        except Exception as e:
            self.logger.error(f"Failed to process image for upload: {e}")
            return image_base64_url

    def _truncate_data_urls_in_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate data URLs in params for safe logging"""
        import copy
        safe_params = copy.deepcopy(params)
        
        messages = safe_params.get('messages', [])
        for msg in messages:
            content = msg.get('content')
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'image_url':
                        url = part.get('image_url', {}).get('url', '')
                        if url.startswith('data:') and len(url) > 100:
                            # Truncate data URL
                            prefix = url.split(',', 1)[0] if ',' in url else url[:50]
                            part['image_url']['url'] = f"{prefix},...[TRUNCATED {len(url)} bytes]"
            elif isinstance(content, str) and content.startswith('data:') and len(content) > 100:
                prefix = content.split(',', 1)[0] if ',' in content else content[:50]
                msg['content'] = f"{prefix},...[TRUNCATED {len(content)} bytes]"
        
        return safe_params
    
    def _optimize_vertex_ai_params(self, params: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Optimize parameters for Vertex AI models"""
        optimized_params = params.copy()
        
        # Check if this is a Vertex AI model
        is_vertex_model = (
            model.startswith('vertex_ai/') or 
            model.startswith('gemini-') or 
            'vertex' in model.lower()
        )
        
        if is_vertex_model:
            is_image_model = "image" in model.lower()
            has_tools = "tools" in optimized_params and optimized_params.get("tools")

            # Always include usage for streaming requests (tools + images supported per latest guidance)
            if optimized_params.get('stream'):
                optimized_params["stream_options"] = {"include_usage": True}

            # If image model and tools are provided in OpenAI format, convert to Vertex function_declarations
            if is_image_model and has_tools:
                try:
                    tools_value = optimized_params.get("tools")
                    # If tools is already an object with function_declarations, keep as-is
                    if isinstance(tools_value, dict) and "function_declarations" in tools_value:
                        pass
                    else:
                        # Expect OpenAI-format list -> convert
                        if isinstance(tools_value, list):
                            fdecls = []
                            for t in tools_value:
                                if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
                                    fdecls.append(t["function"])
                            if fdecls:
                                optimized_params["tools"] = {"function_declarations": fdecls}
                                self.logger.debug(f"Converted tools to function_declarations for {model}")
                except Exception as e:
                    self.logger.debug(f"Tool conversion skipped due to error: {e}")

            # Set optimal temperature for Vertex AI if not specified
            if 'temperature' not in params or params['temperature'] is None:
                optimized_params['temperature'] = 0.7
            
            # Ensure reasonable token limits for Vertex AI
            if not optimized_params.get('max_tokens') and not self.max_tokens:
                optimized_params['max_tokens'] = 8192
        
        return optimized_params
    
    def _convert_markdown_images_to_multimodal(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert markdown image links to multimodal format for vision models.
        This allows LLMs to see images visually while preserving URLs as text.
        
        Pattern: ![alt](url) -> multimodal content with both text and image
        """
        markdown_image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        processed_messages = []
        
        for message in messages:
            msg_copy = dict(message)
            content = msg_copy.get('content')
            
            # Only process user/assistant messages with string content containing markdown images
            if msg_copy.get('role') not in ('user', 'assistant') or not isinstance(content, str):
                processed_messages.append(msg_copy)
                continue
            
            # Check if there are markdown images
            markdown_images = markdown_image_pattern.findall(content)
            if not markdown_images:
                processed_messages.append(msg_copy)
                continue
            
            # Convert to multimodal format
            content_parts = []
            last_end = 0
            
            for match in markdown_image_pattern.finditer(content):
                # Add text before the image (including the markdown link for URL extraction)
                text_chunk = content[last_end:match.end()].strip()
                if text_chunk:
                    content_parts.append({
                        "type": "text",
                        "text": text_chunk
                    })
                
                # Add the image part
                alt_text, image_url = match.groups()
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                })
                
                last_end = match.end()
            
            # Add any remaining text after the last image
            text_after = content[last_end:].strip()
            if text_after:
                content_parts.append({
                    "type": "text",
                    "text": text_after
                })
            
            # Update message with multimodal content
            if content_parts:
                msg_copy['content'] = content_parts
                self.logger.info(f"🖼️  Converted {len(markdown_images)} markdown image(s) to multimodal format")
            
            processed_messages.append(msg_copy)
        
        return processed_messages
    
    async def _execute_completion(self, messages: List[Dict[str, Any]],
                                model: str,
                                tools: Optional[List[Dict[str, Any]]] = None,
                                stream: bool = False,
                                **kwargs) -> Dict[str, Any]:
        """Execute a single completion request"""
        
        # Convert markdown images to multimodal format for vision models
        messages = self._convert_markdown_images_to_multimodal(messages)
        
        # For Vertex AI image models, use direct HTTP to preserve custom fields
        is_vertex_image_model = (
            'image' in model.lower() and 
            (model.startswith('vertex_ai/') or model.startswith('gemini-'))
        )
        
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "stream": stream,
            # Ensure usage is available when streaming is requested later
            # Note: stream_options will be set by _optimize_vertex_ai_params for supported models
            "stream_options": {"include_usage": True} if stream else None,
        }

        # Force a specific provider routing when using an OpenAI-compatible proxy
        # For image models, always use 'openai' to prevent response filtering and disable caching
        if self.custom_llm_provider or is_vertex_image_model:
            params["custom_llm_provider"] = self.custom_llm_provider or 'openai'
            if is_vertex_image_model:
                params["caching"] = False  # Disable caching for image models
                self.logger.debug(f"Using custom_llm_provider='openai' and disabled caching for image model {model}")
        
        # Add base URL if configured (for proxy support)
        if self.config and 'base_url' in self.config:
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
        
        # Optimize parameters for Vertex AI models
        params = self._optimize_vertex_ai_params(params, model)
        
        self.logger.debug(f"Executing completion with model {model}")
        
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
            # Log params summary without huge data URLs
            message_summary = []
            for msg in params.get('messages', []):
                role = msg.get('role', '?')
                content = msg.get('content', '')
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if part.get('type') == 'image_url':
                            url = part.get('image_url', {}).get('url', '')
                            if url.startswith('data:'):
                                parts.append('[data:image]')
                            else:
                                parts.append(f'[image:{url[:30]}...]')
                        elif part.get('type') == 'text':
                            parts.append(f'"{part.get("text", "")[:50]}..."')
                    message_summary.append(f"{role}: [{', '.join(parts)}]")
                else:
                    message_summary.append(f"{role}: {str(content)[:100]}...")
            
            self.logger.error(f"LiteLLM completion failed for model={params.get('model')}")
            self.logger.error(f"Messages: {'; '.join(message_summary)}")
            self.logger.error(f"Tools: {len(params.get('tools', []))} tool(s)")
            self.logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            raise
    
    async def _execute_completion_stream(self, messages: List[Dict[str, Any]],
                                       model: str,
                                       tools: Optional[List[Dict[str, Any]]] = None,
                                       **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a streaming completion request"""
        
        # Convert markdown images to multimodal format for vision models
        messages = self._convert_markdown_images_to_multimodal(messages)
        
        # For Vertex AI image models, use custom_llm_provider='openai' to prevent response filtering
        is_vertex_image_model = (
            'image' in model.lower() and 
            (model.startswith('vertex_ai/') or model.startswith('gemini-'))
        )
        
        # Prepare parameters (same as non-streaming)
        params = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "stream": True,
            # Include a final usage chunk before [DONE] per LiteLLM docs
            # Note: stream_options will be optimized by _optimize_vertex_ai_params for model compatibility
            "stream_options": {"include_usage": True},
        }

        # Force a specific provider routing when using an OpenAI-compatible proxy
        # For image models, always use 'openai' to prevent response filtering and disable caching
        if self.custom_llm_provider or is_vertex_image_model:
            params["custom_llm_provider"] = self.custom_llm_provider or 'openai'
            if is_vertex_image_model:
                params["caching"] = False  # Disable caching for image models
                self.logger.debug(f"Using custom_llm_provider='openai' and disabled caching for streaming image model {model}")
        
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
        
        # Optimize parameters for Vertex AI models
        params = self._optimize_vertex_ai_params(params, model)
        
        self.logger.debug(f"Executing streaming completion with model {model}")
        
        # Add special handling for Vertex AI models
        is_vertex_model = (
            model.startswith('vertex_ai/') or 
            model.startswith('gemini-') or 
            'vertex' in model.lower()
        )
        
        # Execute streaming completion
        stream = await acompletion(**params)
        
        chunk_count = 0
        content_chunks = 0
        
        async for chunk in stream:
            chunk_count += 1
            
            # Normalize and yield chunk
            normalized_chunk = self._normalize_streaming_chunk(chunk, model)
            # After normalization, upload any data:image content inside the same loop/task
            try:
                normalized_chunk = await self._upload_and_rewrite_chunk_images(normalized_chunk, model)
            except Exception:
                pass
            
            # Debug Vertex AI streaming
            if is_vertex_model and chunk_count <= 3:  # Log first few chunks for debugging
                if isinstance(normalized_chunk, dict):
                    choices = normalized_chunk.get('choices', [])
                    if choices and len(choices) > 0:
                        delta = choices[0].get('delta', {})
                        if 'content' in delta and delta['content']:
                            content_chunks += 1
                            self.logger.debug(f"Vertex AI streaming chunk {chunk_count}: got content ({len(delta['content'])} chars)")
            
            # If LiteLLM sent a final usage chunk, log tokens to context.usage
            try:
                usage = normalized_chunk.get('usage') if isinstance(normalized_chunk, dict) else None
                if usage and isinstance(usage, dict):
                    prompt_tokens = int(usage.get('prompt_tokens') or 0)
                    completion_tokens = int(usage.get('completion_tokens') or 0)
                    total_tokens = int(usage.get('total_tokens') or (prompt_tokens + completion_tokens))
                    
                    # DEBUG: Log all usage fields to see what Gemini sends
                    self.logger.info(f"🔍 USAGE CHUNK RECEIVED: {usage}")
                    
                    self._append_usage_record(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens, streaming=True)
            except Exception as e:
                # Never break streaming on usage logging
                self.logger.warning(f"💰 LiteLLM streaming: Failed to log usage: {e}")
                pass
            yield normalized_chunk
            
        # Log streaming completion stats for Vertex AI models
        if is_vertex_model:
            self.logger.debug(f"Vertex AI streaming completed: {chunk_count} total chunks, {content_chunks} with content")
    
    def _normalize_response(self, response: Any, model: str) -> Dict[str, Any]:
        """Normalize LiteLLM response to OpenAI format and handle images"""
        
        self.logger.info(f"🔍 _normalize_response called for model: {model}")
        
        # Log the raw response object type and attributes
        self.logger.debug(f"🔍 Raw response type: {type(response)}")
        if hasattr(response, '__dict__'):
            all_attrs = list(vars(response).keys())
            self.logger.debug(f"🔍 Raw response attributes: {all_attrs[:20]}")
        
        # Convert response to dict if needed
        if hasattr(response, 'model_dump'):
            response_dict = response.model_dump()
        elif hasattr(response, 'dict'):
            response_dict = response.dict()
        elif isinstance(response, dict):
            response_dict = response.copy()
        else:
            response_dict = dict(response) if response else {}
        
        self.logger.debug(f"🔍 Response dict keys: {response_dict.keys()}")
        
        # Check for response data in dict format
        if 'choices' in response_dict and response_dict['choices']:
            self.logger.debug(f"🔍 First choice keys: {response_dict['choices'][0].keys()}")
            if 'message' in response_dict['choices'][0]:
                msg = response_dict['choices'][0]['message']
                self.logger.debug(f"🔍 Message keys: {msg.keys()}")
                self.logger.debug(f"🔍 Message content length: {len(str(msg.get('content', '')))}")
        
        # Ensure model name is correct and includes provider prefix for cost lookup
        # Add vertex_ai/ prefix if it's a Gemini model without a provider prefix
        if model and not model.startswith(('vertex_ai/', 'openai/', 'anthropic/', 'xai/')):
            if 'gemini' in model.lower() or 'flash' in model.lower():
                response_dict['model'] = f"vertex_ai/{model}"
            else:
                response_dict['model'] = model
        else:
            response_dict['model'] = model
        
        # Handle custom image field from Gemini models
        if 'choices' in response_dict and response_dict['choices']:
            for choice in response_dict['choices']:
                if 'message' in choice and choice['message']:
                    message = choice['message']
                    
                    # Check for custom image field
                    if 'image' in message and message['image']:
                        image_data = message['image']
                        
                        self.logger.info(f"Found image field in non-streaming response for {model}")
                        
                        # Convert image to markdown format for display
                        if 'url' in image_data and image_data['url']:
                            image_url = image_data['url']
                            
                            # Upload all base64 images to content API for better performance and reliability
                            if image_url.startswith('data:image/'):
                                self.logger.info(f"Base64 image detected ({len(image_url)} chars), uploading to content API")
                                try:
                                    # Handle async upload in sync context
                                    import asyncio
                                    import concurrent.futures
                                    
                                    def run_upload():
                                        return asyncio.run(self._upload_image_to_content_api(image_url, model))
                                    
                                    try:
                                        loop = asyncio.get_event_loop()
                                        if loop.is_running():
                                            # If loop is already running, run in a separate thread
                                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                                future = executor.submit(run_upload)
                                                uploaded_url = future.result(timeout=30)
                                        else:
                                            uploaded_url = loop.run_until_complete(self._upload_image_to_content_api(image_url, model))
                                    except RuntimeError:
                                        # Fallback: run in new event loop
                                        uploaded_url = run_upload()
                                    
                                    if uploaded_url != image_url:  # Only update if upload was successful
                                        image_url = uploaded_url
                                        self.logger.info(f"Successfully uploaded image to: {uploaded_url}")
                                except Exception as e:
                                    self.logger.error(f"Failed to upload image: {e}, using original URL")
                            
                            # Create markdown image syntax
                            image_markdown = f"![Generated Image]({image_url})"
                            
                            # Append to content or replace if no content
                            current_content = message.get('content') or ''
                            if current_content:
                                message['content'] = f"{current_content}\n\n{image_markdown}"
                            else:
                                message['content'] = image_markdown
                            
                            self.logger.info(f"Converted image field to markdown for model {model}")
                            
                            # Remove the custom image field since we've converted it
                            del message['image']
        
        return response_dict
    
    def _normalize_streaming_chunk(self, chunk: Any, model: str) -> Dict[str, Any]:
        """Normalize LiteLLM streaming chunk to OpenAI format and handle images"""
        
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
        
        # Ensure model name is correct and includes provider prefix for cost lookup
        # Add vertex_ai/ prefix if it's a Gemini model without a provider prefix
        if model and not model.startswith(('vertex_ai/', 'openai/', 'anthropic/', 'xai/')):
            if 'gemini' in model.lower() or 'flash' in model.lower():
                chunk_dict['model'] = f"vertex_ai/{model}"
            else:
                chunk_dict['model'] = model
        else:
            chunk_dict['model'] = model
        
        # Handle custom image field from Gemini models in streaming
        if 'choices' in chunk_dict and chunk_dict['choices']:
            for choice in chunk_dict['choices']:
                # Check both delta and message for image data
                for message_key in ['delta', 'message']:
                    if message_key in choice and choice[message_key]:
                        message = choice[message_key]
                        
                        # Check for custom image field
                        if 'image' in message and message['image']:
                            self.logger.info(f"🔍 STREAMING: Found 'image' field in {message_key}")
                            image_data = message['image']
                            
                            # Convert image to markdown format for display (do not upload here)
                            if 'url' in image_data and image_data['url']:
                                image_url = image_data['url']
                                self.logger.info(f"🔍 STREAMING: Image URL: {image_url[:100]}...")
                                
                                # Create markdown image syntax
                                image_markdown = f"\n\n![Generated Image]({image_url})"
                                
                                # For streaming, replace any existing content with the image markdown
                                message['content'] = image_markdown
                                
                                self.logger.info(f"Converted streaming image field to markdown for model {model}")
                                
                                # Remove the custom image field since we've converted it
                                del message['image']
        
        return chunk_dict

    async def _upload_and_rewrite_chunk_images(self, chunk: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        Detect data:image URLs in chunk content, upload to content API, rewrite URLs,
        and log pricing. Runs inside the streaming loop so context is available.
        """
        try:
            if not chunk or 'choices' not in chunk:
                return chunk
            data_url_pattern = re.compile(r"data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+")
            image_logged = False
            for choice in chunk['choices']:
                for key in ['delta', 'message']:
                    if key in choice and choice[key] and isinstance(choice[key], dict):
                        msg = choice[key]
                        content = msg.get('content')
                        if isinstance(content, str):
                            matches = list(data_url_pattern.finditer(content))
                            if not matches:
                                continue
                            new_content = content
                            for m in matches:
                                data_url = m.group(0)
                                try:
                                    public_url = await self._upload_image_to_content_api(data_url, model)
                                    if public_url and public_url != data_url:
                                        new_content = new_content.replace(data_url, public_url)
                                        # Always log one pricing record per uploaded image URL
                                        self._log_image_upload_pricing()
                                        image_logged = True
                                except Exception as e:
                                    self.logger.error(f"Image upload failed during streaming rewrite: {e}")
                            if new_content != content:
                                msg['content'] = new_content
            return chunk
        except Exception as e:
            self.logger.error(f"Error rewriting streaming image URLs: {e}")
            return chunk
    
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
            
            record = {
                'type': 'llm',
                'skill': 'litellm',
                'model': model,
                'prompt_tokens': int(prompt_tokens or 0),
                'completion_tokens': int(completion_tokens or 0),
                'total_tokens': int(total_tokens or 0),
                'streaming': bool(streaming),
                'timestamp': time.time(),
            }
            context.usage.append(record)
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
    
    async def _upload_image_to_content_api(self, image_base64_url: str, model: str) -> str:
        """
        Upload base64 image data to content API and return a public URL.
        Uses the same approach as the openlicense skill.
        Charges 5 cents per image upload.
        """
        if not HTTPX_AVAILABLE:
            self.logger.warning("httpx not available, cannot upload image to content API")
            return image_base64_url
            
        try:
            # Extract base64 data from data URL
            if not image_base64_url.startswith('data:image/'):
                self.logger.warning(f"Invalid image URL format: {image_base64_url[:50]}...")
                return image_base64_url  # Return as-is if not a data URL
            
            # Parse the data URL: data:image/png;base64,<data>
            try:
                header, data = image_base64_url.split(',', 1)
                image_format = header.split('/')[1].split(';')[0]  # Extract format (png, jpeg, etc.)
                image_data = base64.b64decode(data)
            except Exception as e:
                self.logger.error(f"Failed to parse base64 image data: {e}")
                return image_base64_url
            
            # Get API key - try multiple sources with comprehensive fallbacks
            agent_api_key = None
            
            # Method 1: Try context first (same approach as openlicense skill)
            try:
                from webagents.server.context.context_vars import get_context
                context = get_context()
                
                if context:
                    # Try multiple possible keys (same as openlicense)
                    agent_api_key = (context.get("api_key") or 
                                    context.get("robutler_api_key") or 
                                    context.get("agent_api_key") or
                                    getattr(context, 'api_key', None))
                    
                    if agent_api_key:
                        self.logger.info(f"Found agent API key from context: {agent_api_key[:10]}...{agent_api_key[-4:] if len(agent_api_key) > 14 else ''}")
                    else:
                        self.logger.debug("No agent API key found in context")
                else:
                    self.logger.debug("No context available")
            except Exception as e:
                self.logger.debug(f"Error accessing context: {e}")
            
            # Method 2: Try skill's own config (from dynamic factory)
            if not agent_api_key and hasattr(self, 'config') and self.config:
                self.logger.debug("Trying to get API key from skill config...")
                try:
                    # For r-banana, the API key should be in api_keys dict
                    if isinstance(self.config, dict):
                        # Try robutler_api_key first
                        agent_api_key = self.config.get('robutler_api_key')
                        
                        # Try api_keys dict (from dynamic factory)
                        if not agent_api_key and 'api_keys' in self.config:
                            api_keys = self.config['api_keys']
                            if isinstance(api_keys, dict):
                                # Try different provider keys
                                agent_api_key = (api_keys.get('azure') or 
                                               api_keys.get('openai') or 
                                               api_keys.get('anthropic') or 
                                               api_keys.get('google'))
                        
                        if agent_api_key:
                            self.logger.info(f"Found agent API key from skill config: {agent_api_key[:10]}...{agent_api_key[-4:] if len(agent_api_key) > 14 else ''}")
                        else:
                            self.logger.debug(f"No API key found in skill config. Config keys: {list(self.config.keys())}")
                            if 'api_keys' in self.config:
                                self.logger.debug(f"api_keys dict keys: {list(self.config['api_keys'].keys()) if isinstance(self.config['api_keys'], dict) else 'not a dict'}")
                except (KeyError, TypeError, AttributeError) as e:
                    self.logger.debug(f"Could not access skill config: {e}")
            
            # Method 3: Try environment variables as last resort
            if not agent_api_key:
                self.logger.debug("Trying environment variables as fallback...")
                agent_api_key = (os.getenv('ROBUTLER_API_KEY') or 
                               os.getenv('API_KEY') or
                               os.getenv('AGENT_API_KEY'))
                if agent_api_key:
                    self.logger.info(f"Found agent API key from environment: {agent_api_key[:10]}...{agent_api_key[-4:] if len(agent_api_key) > 14 else ''}")
                else:
                    self.logger.debug("No API key found in environment variables")
            
            if not agent_api_key:
                self.logger.error("No agent API key found anywhere - content upload will fail")
                return image_base64_url  # Fallback to original URL
            
            # Get portal URL (same as openlicense)
            portal_url = os.getenv('ROBUTLER_INTERNAL_API_URL', 'http://localhost:3000')
            upload_url = f"{portal_url}/api/content"
            
            # Get agent ID for access scope
            agent_id = None
            if context:
                agent_id = context.get("agent_id") or context.get("current_agent_id")
                if agent_id:
                    self.logger.debug(f"Found agent_id in context for access scope: {agent_id}")
            
            # Prepare file data
            filename = f"ai_image_{uuid.uuid4().hex[:8]}.{image_format}"
            files = {
                'file': (filename, image_data, f'image/{image_format}')
            }
            
            # Prepare metadata
            description = f"AI-generated image from {model}"
            tags = ['ai-generated', 'litellm-skill', model.replace('/', '-')]
            
            # Prepare agent access
            grant_agent_access = []
            if agent_id:
                grant_agent_access.append(agent_id)
                self.logger.debug(f"Granting agent access to: {agent_id}")
            
            data = {
                'visibility': 'public',
                'description': description,
                'tags': json.dumps(tags),
                'grantAgentAccess': json.dumps(grant_agent_access) if grant_agent_access else None
            }
            
            # Make authenticated request to upload API (same as openlicense)
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Prepare headers with proper API key authentication
                headers = {'User-Agent': 'LiteLLM-Skill/1.0'}
                
                if agent_api_key:
                    # Use Bearer token format as expected by the content API
                    headers['Authorization'] = f'Bearer {agent_api_key}'
                    self.logger.debug("Added Authorization header with Bearer token")
                else:
                    self.logger.warning("No API key available for authentication - upload may fail")
                
                response = await client.post(
                    upload_url,
                    files=files,
                    data=data,
                    headers=headers
                )
                
                if response.status_code in [200, 201]:
                    result = response.json()
                    content_url = result.get('url')
                    if content_url:
                        # Replace portal URL with chat URL for public access (same as openlicense skill)
                        chat_base = (os.getenv('ROBUTLER_CHAT_URL') or 'http://localhost:3001').rstrip('/')
                        portal_base = (os.getenv('ROBUTLER_INTERNAL_API_URL') or 'http://localhost:3000').rstrip('/')
                        
                        if content_url.startswith(portal_base):
                            # Replace portal base with chat base for public URL
                            public_url = content_url.replace(portal_base, chat_base, 1)
                            self.logger.info(f"Successfully uploaded image to content API: {public_url}")
                            
                            return public_url
                        else:
                            self.logger.info(f"Successfully uploaded image to content API: {content_url}")
                            return content_url
                    else:
                        self.logger.error(f"Upload successful but no URL returned: {result}")
                        return image_base64_url
                else:
                    self.logger.error(f"Failed to upload image to content API: {response.status_code} - {response.text}")
                    return image_base64_url
                    
        except Exception as e:
            self.logger.error(f"Error uploading image to content API: {e}")
            return image_base64_url

    def _log_image_upload_pricing(self) -> None:
        """
        FIXME: this is a temporary hack to log image upload pricing to context for PaymentSkill to process.
        This should be removed in V0.3.0
        Log image upload pricing to context for PaymentSkill to process.
        Charges 5 cents per image upload.
        """
        try:
            from webagents.server.context.context_vars import get_context
            context = get_context()
            
            if context:
                # Initialize usage list if not present
                if not hasattr(context, 'usage') or context.usage is None:
                    context.usage = []
                
                # Add tool usage record in the format PaymentSkill expects
                usage_record = {
                    'type': 'tool',
                    'tool_name': 'image_upload',
                    'pricing': {
                        'credits': 0.05,
                        'reason': 'AI image upload to content API',
                        'metadata': {
                            'service': 'image_upload',
                            'model': 'content_api'
                        }
                    }
                }
                
                context.usage.append(usage_record)
                self.logger.info(f"💰 Logged image upload pricing: 0.05 credits - AI image upload to content API")
            else:
                self.logger.warning("No context available to log image upload pricing")
                
        except Exception as e:
            self.logger.error(f"Error logging image upload pricing: {e}")

    async def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embeddings (placeholder for V2.1)"""
        # This would use LiteLLM's embedding support in V2.1
        self.logger.info("Embedding generation requested - will be implemented in V2.1")
        return [0.0] * 1536  # Placeholder embedding 