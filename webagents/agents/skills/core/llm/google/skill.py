"""
Google AI Skill - WebAgents V2.0

Native integration with Google's GenAI API (Gemini).
Uses the official google-genai SDK for direct API access.

Features:
- Direct Gemini API access (no proxy required)
- Streaming and non-streaming support
- Tool calling with function declarations
- Multi-modal support (text, images, video)
- Automatic retries and error handling
- Token usage tracking

Supported Models:
- gemini-2.5-pro / gemini-2.5-flash
- gemini-2.0-flash
- gemini-1.5-pro / gemini-1.5-flash
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator, Union, TYPE_CHECKING
from dataclasses import dataclass

try:
    from google import genai
    from google.genai import types
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None
    types = None

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, hook
from webagents.utils.logging import get_logger, log_skill_event, timer


@dataclass
class GeminiModelConfig:
    """Configuration for a specific Gemini model"""
    name: str
    max_output_tokens: int = 8192
    max_input_tokens: int = 1048576
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = True
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0


class GoogleAISkill(Skill):
    """
    Native Google Gemini skill using the official google-genai SDK.
    
    Direct access to Google's GenAI API with full feature support
    including streaming, tool calling, and multi-modal inputs.
    """
    
    # Default model configurations
    DEFAULT_MODELS = {
        # Gemini 3 series
        "gemini-3-flash-preview": GeminiModelConfig(
            "gemini-3-flash-preview", 8192, 1048576, True, True, True,
            0.0, 0.0
        ),
        "gemini-3-pro-preview": GeminiModelConfig(
            "gemini-3-pro-preview", 8192, 1048576, True, True, True,
            0.0, 0.0
        ),

        # Gemini 2.5 series
        "gemini-2.5-pro": GeminiModelConfig(
            "gemini-2.5-pro", 8192, 1048576, True, True, True,
            1.25, 5.00
        ),
        "gemini-2.5-flash": GeminiModelConfig(
            "gemini-2.5-flash", 8192, 1048576, True, True, True,
            0.075, 0.30
        ),
        "gemini-2.5-flash-lite": GeminiModelConfig(
            "gemini-2.5-flash-lite", 8192, 1048576, True, True, True,
            0.0375, 0.15
        ),
        
        # Gemini 2.0 series
        "gemini-2.0-flash": GeminiModelConfig(
            "gemini-2.0-flash", 8192, 1048576, True, True, True,
            0.0, 0.0
        ),
        "gemini-2.0-flash-exp": GeminiModelConfig(
            "gemini-2.0-flash-exp", 8192, 1048576, True, True, True,
            0.0, 0.0
        ),
        "gemini-2.0-flash-lite": GeminiModelConfig(
            "gemini-2.0-flash-lite", 8192, 1048576, True, True, True,
            0.0, 0.0
        ),
        "gemini-2.0-flash-thinking-exp": GeminiModelConfig(
            "gemini-2.0-flash-thinking-exp", 8192, 1048576, True, True, True,
            0.0, 0.0
        ),
        
        # Gemini 1.5 series
        "gemini-1.5-pro": GeminiModelConfig(
            "gemini-1.5-pro", 8192, 1048576, True, True, True,
            0.0, 0.0
        ),
        "gemini-1.5-flash": GeminiModelConfig(
            "gemini-1.5-flash", 8192, 1048576, True, True, True,
            0.0, 0.0
        ),
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        
        self.config = config or {}
        
        # Configuration
        self.model = config.get('model', 'gemini-2.5-flash') if config else 'gemini-2.5-flash'
        self.temperature = config.get('temperature', 0.7) if config else 0.7
        self.max_tokens = config.get('max_tokens') if config else None
        
        # Thinking config
        self.thinking_config = config.get('thinking', {})
        
        self.fallback_models = config.get('fallback_models', []) if config else []
        
        # API configuration
        self.api_key = self._load_api_key(config)
        self.model_configs = {**self.DEFAULT_MODELS}
        if config and 'custom_models' in config:
            self.model_configs.update(config['custom_models'])
        
        # Runtime state
        self.current_model = self.model
        self._client = None
        self.agent = None
        self.logger = get_logger('skill.llm.google', 'init')
        
        # State for thinking detection
        self._in_thinking_block = False
        
        # Validate availability
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError(
                "Google GenAI SDK not available. "
                "Install with: pip install google-genai"
            )
            
    def _get_google_tools(self) -> Optional[List[types.Tool]]:
        """Get configured Google built-in tools"""
        google_tools = []
        
        # Check config for enabled tools
        # Structure could be:
        # google_tools: ["google_search", "code_execution"]
        # or
        # google_tools: [{"google_search": {}}, {"code_execution": {}}]
        
        configured_tools = self.config.get('tools', []) or self.config.get('google_tools', [])
        
        for tool_config in configured_tools:
            tool_name = tool_config if isinstance(tool_config, str) else list(tool_config.keys())[0]
            
            if tool_name in ['google_search', 'web_search']:
                google_tools.append(types.Tool(google_search=types.GoogleSearch()))
            elif tool_name in ['code_execution', 'code_interpreter']:
                google_tools.append(types.Tool(code_execution=types.CodeExecution()))
            # Add other tools as needed (e.g. google_search_retrieval)
            
        return google_tools if google_tools else None

    def _load_api_key(self, config: Dict[str, Any] = None) -> str:
        """Load API key from config or environment"""
        if config and 'api_key' in config:
            return config['api_key']
        if config and 'api_keys' in config and 'google' in config['api_keys']:
            return config['api_keys']['google']
        
        return (
            os.environ.get('GOOGLE_GEMINI_API_KEY') or
            os.environ.get('GOOGLE_API_KEY') or
            os.environ.get('GEMINI_API_KEY', '')
        )
    
    def _get_client(self):
        """Get or create the GenAI client"""
        if self._client is None:
            if self.api_key:
                self._client = genai.Client(api_key=self.api_key)
            else:
                # Uses GEMINI_API_KEY env var
                self._client = genai.Client()
        return self._client
    
    async def initialize(self, agent: 'BaseAgent') -> None:
        """Initialize Google AI skill and register as handoff"""
        from webagents.agents.skills.base import Handoff
        
        self.agent = agent
        self.logger = get_logger('skill.llm.google', agent.name)
        
        # Initialize client
        self._get_client()
        self.logger.info(f"Google GenAI client initialized")
        
        # Register as handoff (completion handler)
        agent.register_handoff(
            Handoff(
                target=f"google_{self.model.replace('/', '_').replace('-', '_')}",
                description=f"Google Gemini completion handler using {self.model}",
                scope="all",
                metadata={
                    'function': self.chat_completion_stream,
                    'priority': 10,
                    'is_generator': True
                }
            ),
            source="google"
        )
        
        self.logger.info(f"Registered Google AI as handoff with model: {self.model}")
        
        log_skill_event(agent.name, 'google', 'initialized', {
            'model': self.model,
            'temperature': self.temperature,
            'fallback_models': self.fallback_models,
        })
    
    # === Core LLM functionality ===
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Create a chat completion using Google Gemini."""
        if stream:
            raise ValueError("Use chat_completion_stream() for streaming responses")
        
        target_model = model or self.current_model
        agent_name = getattr(self, 'agent', None) and self.agent.name or 'unknown'
        
        with timer(f"chat_completion_{target_model}", agent_name):
            try:
                response = await self._execute_completion(
                    messages=messages,
                    model=target_model,
                    tools=tools,
                    **kwargs
                )
                return response
                
            except Exception as e:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.error(f"Chat completion failed for {target_model}: {e}")
                
                for fallback_model in self.fallback_models:
                    try:
                        self.logger.info(f"Trying fallback model: {fallback_model}")
                        response = await self._execute_completion(
                            messages=messages,
                            model=fallback_model,
                            tools=tools,
                            **kwargs
                        )
                        return response
                    except Exception as fallback_error:
                        self.logger.warning(f"Fallback {fallback_model} failed: {fallback_error}")
                        continue
                
                raise e
    
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Create a streaming chat completion using Google Gemini."""
        target_model = model or self.current_model
        
        try:
            async for chunk in self._execute_completion_stream(
                messages=messages,
                model=target_model,
                tools=tools,
                **kwargs
            ):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Streaming completion failed for {target_model}: {e}")
            
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
                    return
                except Exception as fallback_error:
                    self.logger.warning(f"Fallback streaming {fallback_model} failed: {fallback_error}")
                    continue
            
            raise e
    
    # === Private helpers ===
    
    def _convert_messages_to_gemini(
        self,
        messages: List[Dict[str, Any]]
    ) -> tuple[Optional[str], List[types.Content]]:
        """Convert OpenAI-format messages to Gemini format."""
        system_instruction = None
        contents = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Handle system messages
            if role == 'system':
                if isinstance(content, str):
                    system_instruction = (system_instruction or "") + content + "\n"
                continue
            
            # Map roles
            gemini_role = 'model' if role == 'assistant' else 'user'
            
            # Handle content
            if isinstance(content, str):
                contents.append(types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)]
                ))
            elif isinstance(content, list):
                # Multi-modal content
                parts = []
                for part in content:
                    if part.get('type') == 'text':
                        parts.append(types.Part.from_text(text=part.get('text', '')))
                    elif part.get('type') == 'image_url':
                        image_url = part.get('image_url', {}).get('url', '')
                        if image_url.startswith('data:'):
                            parts.append(self._parse_data_url(image_url))
                        else:
                            parts.append(types.Part.from_text(text=f"[Image: {image_url}]"))
                
                if parts:
                    contents.append(types.Content(role=gemini_role, parts=parts))
            
            # Handle tool calls in assistant messages
            if role == 'assistant' and 'tool_calls' in msg:
                for tool_call in (msg.get('tool_calls') or []):
                    if tool_call.get('type') == 'function':
                        func = tool_call.get('function', {})
                        parts = [types.Part.from_function_call(
                            name=func.get('name', ''),
                            args=json.loads(func.get('arguments', '{}'))
                        )]
                        contents.append(types.Content(role='model', parts=parts))
            
            # Handle tool response messages
            if role == 'tool':
                contents.append(types.Content(
                    role='user',
                    parts=[types.Part.from_function_response(
                        name=msg.get('name', msg.get('tool_call_id', 'unknown')),
                        response={'result': content}
                    )]
                ))
        
        return system_instruction, contents
    
    def _parse_data_url(self, data_url: str) -> types.Part:
        """Parse a data URL into Gemini Part format"""
        try:
            header, data = data_url.split(',', 1)
            mime_type = header.split(':')[1].split(';')[0]
            import base64
            return types.Part.from_bytes(
                data=base64.b64decode(data),
                mime_type=mime_type
            )
        except Exception:
            return types.Part.from_text(text='[Invalid image data]')
    
    def _convert_tools_to_gemini(
        self,
        tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[types.Tool]]:
        """Convert OpenAI-format tools to Gemini function declarations"""
        if not tools:
            return None
        
        function_declarations = []
        
        for tool_def in tools:
            if tool_def.get('type') != 'function':
                continue
            
            func = tool_def.get('function', {})
            parameters = func.get('parameters', {})
            
            function_declarations.append(types.FunctionDeclaration(
                name=func.get('name', ''),
                description=func.get('description', ''),
                parameters=parameters
            ))
        
        if function_declarations:
            return [types.Tool(function_declarations=function_declarations)]
        return None
    
    async def _execute_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a single completion request"""
        
        model_config = self.model_configs.get(model)
        model_name = model_config.name if model_config else model
        
        client = self._get_client()
        system_instruction, contents = self._convert_messages_to_gemini(messages)
        gemini_tools = self._convert_tools_to_gemini(tools)
        
        # Add Google built-in tools if configured
        built_in_tools = self._get_google_tools()
        if built_in_tools:
            if gemini_tools:
                gemini_tools.extend(built_in_tools)
            else:
                gemini_tools = built_in_tools
        
        # Support for thinking mode - Enable for all capable models
        thinking_config = None
        
        # Check if model supports thinking (Gemini 2.5+)
        is_thinking_capable = "gemini-2.5" in model_name.lower() or "gemini-3" in model_name.lower()
        should_think = kwargs.get('thinking', True) if is_thinking_capable else False # Default to True for capable models
        
        if should_think and hasattr(types, 'ThinkingConfig'):
            budget = self.thinking_config.get('budget_tokens')
            
            if not budget:
                # Map 'effort' to token budget if explicit tokens not provided
                effort = self.thinking_config.get('effort', 'low')
                if effort == 'low':
                    budget = 1024
                elif effort == 'medium':
                    budget = 4096 
                elif effort == 'high':
                    budget = 8192
                else:
                    budget = 1024

            if "gemini-2.5" in model_name.lower():
                # Gemini 2.5 Flash requires explicit budget > 0 to guarantee thinking
                # Using budget from config or default
                thinking_budget = budget
                thinking_config = types.ThinkingConfig(include_thoughts=True, thinking_budget=thinking_budget)
            elif "gemini-3" in model_name.lower():
                # Gemini 3 uses include_thoughts and optionally thinking_level (default HIGH)
                thinking_config = types.ThinkingConfig(include_thoughts=True)
            else:
                thinking_config = types.ThinkingConfig(include_thoughts=True)
        elif should_think and is_thinking_capable:
             self.logger.warning("ThinkingConfig not available in installed google-genai SDK version")

        config = types.GenerateContentConfig(
            temperature=kwargs.get('temperature', self.temperature),
            max_output_tokens=kwargs.get('max_tokens', self.max_tokens) or 8192,
            top_p=kwargs.get('top_p'),
            system_instruction=system_instruction,
            tools=gemini_tools,
            thinking_config=thinking_config,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
        
        self.logger.debug(f"Executing completion with model {model_name}")
        
        # Use async client
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )
        
        return self._normalize_response(response, model)
    
    async def _execute_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a streaming completion request"""
        
        # Reset thinking state for new request
        self._in_thinking_block = False
        
        model_config = self.model_configs.get(model)
        model_name = model_config.name if model_config else model
        
        client = self._get_client()
        system_instruction, contents = self._convert_messages_to_gemini(messages)
        gemini_tools = self._convert_tools_to_gemini(tools)
        
        # Add Google built-in tools if configured
        built_in_tools = self._get_google_tools()
        if built_in_tools:
            if gemini_tools:
                gemini_tools.extend(built_in_tools)
            else:
                gemini_tools = built_in_tools
        
        # Support for thinking mode - Enable for all capable models
        thinking_config = None
        
        # Check if model supports thinking (Gemini 2.5+)
        is_thinking_capable = "gemini-2.5" in model_name.lower() or "gemini-3" in model_name.lower()
        should_think = kwargs.get('thinking', True) if is_thinking_capable else False # Default to True for capable models
        
        if should_think and hasattr(types, 'ThinkingConfig'):
            budget = self.thinking_config.get('budget_tokens')
            
            if not budget:
                # Map 'effort' to token budget if explicit tokens not provided
                effort = self.thinking_config.get('effort', 'low')
                if effort == 'low':
                    budget = 1024
                elif effort == 'medium':
                    budget = 4096 
                elif effort == 'high':
                    budget = 8192
                else:
                    budget = 1024

            if "gemini-2.5" in model_name.lower():
                # Gemini 2.5 Flash requires explicit budget > 0 to guarantee thinking
                # Using budget from config or default
                thinking_budget = budget
                thinking_config = types.ThinkingConfig(include_thoughts=True, thinking_budget=thinking_budget)
            elif "gemini-3" in model_name.lower():
                # Gemini 3 uses include_thoughts and optionally thinking_level (default HIGH)
                thinking_config = types.ThinkingConfig(include_thoughts=True)
            else:
                thinking_config = types.ThinkingConfig(include_thoughts=True)
        elif should_think and is_thinking_capable:
             self.logger.warning("ThinkingConfig not available in installed google-genai SDK version")

        config = types.GenerateContentConfig(
            temperature=kwargs.get('temperature', self.temperature),
            max_output_tokens=kwargs.get('max_tokens', self.max_tokens) or 8192,
            top_p=kwargs.get('top_p'),
            system_instruction=system_instruction,
            tools=gemini_tools,
            thinking_config=thinking_config,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
        
        self.logger.debug(f"Executing streaming completion with model {model_name}")
        
        # Use async streaming - must await the coroutine to get the async iterator
        chunk_count = 0
        stream = await client.aio.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=config
        )
        async for chunk in stream:
            chunk_count += 1
            yield self._normalize_streaming_chunk(chunk, model, chunk_count)
            
        # Ensure thinking block is closed at the end of stream
        if self._in_thinking_block:
            self._in_thinking_block = False
            # Yield a final dummy chunk to close the tag
            yield {
                'id': f'gemini-stream-final',
                'object': 'chat.completion.chunk',
                'model': model,
                'choices': [{
                    'index': 0,
                    'delta': {'content': '</think>'},
                    'finish_reason': 'stop'
                }]
            }
    
    def _normalize_response(
        self,
        response: Any,
        model: str
    ) -> Dict[str, Any]:
        """Normalize Gemini response to OpenAI format"""
        
        try:
            candidate = response.candidates[0] if response.candidates else None
            
            if not candidate:
                return {
                    'id': f'gemini-{id(response)}',
                    'object': 'chat.completion',
                    'model': model,
                    'choices': [{
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': 'No response generated'
                        },
                        'finish_reason': 'stop'
                    }],
                    'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                }
            
            content = ""
            tool_calls = []
            
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    content += part.text
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    tool_calls.append({
                        'id': f'call_{len(tool_calls)}',
                        'type': 'function',
                        'function': {
                            'name': fc.name,
                            'arguments': json.dumps(dict(fc.args) if fc.args else {})
                        }
                    })
            
            message = {'role': 'assistant', 'content': content}
            if tool_calls:
                message['tool_calls'] = tool_calls
            
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                um = response.usage_metadata
                usage = {
                    'prompt_tokens': getattr(um, 'prompt_token_count', 0),
                    'completion_tokens': getattr(um, 'candidates_token_count', 0),
                    'total_tokens': getattr(um, 'total_token_count', 0)
                }
            
            return {
                'id': f'gemini-{id(response)}',
                'object': 'chat.completion',
                'model': model,
                'choices': [{
                    'index': 0,
                    'message': message,
                    'finish_reason': self._map_finish_reason(candidate.finish_reason)
                }],
                'usage': usage
            }
            
        except Exception as e:
            self.logger.error(f"Failed to normalize response: {e}")
            return {
                'id': f'gemini-error',
                'object': 'chat.completion',
                'model': model,
                'choices': [{
                    'index': 0,
                    'message': {'role': 'assistant', 'content': f'Error: {e}'},
                    'finish_reason': 'error'
                }],
                'usage': {}
            }
    
    def _normalize_streaming_chunk(
        self,
        chunk: Any,
        model: str,
        chunk_index: int
    ) -> Dict[str, Any]:
        """Normalize Gemini streaming chunk to OpenAI format"""
        
        try:
            candidate = chunk.candidates[0] if chunk.candidates else None
            
            delta = {}
            if candidate and candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # Handle internal reasoning (thoughts)
                    if hasattr(part, 'thought') and part.thought:
                        if not self._in_thinking_block:
                            self._in_thinking_block = True
                            delta['content'] = delta.get('content', '') + f"<think>{part.text}"
                        else:
                            delta['content'] = delta.get('content', '') + part.text
                    elif hasattr(part, 'text') and part.text:
                        if self._in_thinking_block:
                            self._in_thinking_block = False
                            delta['content'] = delta.get('content', '') + f"</think>{part.text}"
                        else:
                            delta['content'] = delta.get('content', '') + part.text
                    
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        delta['tool_calls'] = [{
                            'index': 0,
                            'id': f'call_0',
                            'type': 'function',
                            'function': {
                                'name': fc.name,
                                'arguments': json.dumps(dict(fc.args) if fc.args else {})
                            }
                        }]
            
            finish_reason = None
            if candidate and candidate.finish_reason:
                finish_reason = self._map_finish_reason(candidate.finish_reason)
            
            result = {
                'id': f'gemini-stream-{chunk_index}',
                'object': 'chat.completion.chunk',
                'model': model,
                'choices': [{
                    'index': 0,
                    'delta': delta,
                    'finish_reason': finish_reason
                }]
            }
            
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                um = chunk.usage_metadata
                result['usage'] = {
                    'prompt_tokens': getattr(um, 'prompt_token_count', 0),
                    'completion_tokens': getattr(um, 'candidates_token_count', 0),
                    'total_tokens': getattr(um, 'total_token_count', 0)
                }
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to normalize streaming chunk: {e}")
            return {
                'id': f'gemini-stream-{chunk_index}',
                'object': 'chat.completion.chunk',
                'model': model,
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': None
                }]
            }
    
    def _map_finish_reason(self, gemini_reason: Any) -> str:
        """Map Gemini finish reason to OpenAI format"""
        if gemini_reason is None:
            return None
        
        reason_str = str(gemini_reason).upper()
        
        mapping = {
            'STOP': 'stop',
            'MAX_TOKENS': 'length',
            'SAFETY': 'content_filter',
            'RECITATION': 'content_filter',
            'OTHER': 'stop',
            'FINISH_REASON_UNSPECIFIED': 'stop',
        }
        
        return mapping.get(reason_str, 'stop')
    
    # === Tool methods ===
    
    @tool(scope="all")
    async def analyze_image(
        self,
        image_url: str,
        prompt: str = "Describe this image in detail.",
        model: Optional[str] = None
    ) -> str:
        """
        Analyze an image using Google Gemini's vision capabilities.
        
        Args:
            image_url: URL or base64 data URL of the image
            prompt: What to analyze about the image
            model: Optional model override (must support vision)
        
        Returns:
            Analysis of the image
        """
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': image_url}},
                {'type': 'text', 'text': prompt}
            ]
        }]
        
        response = await self.chat_completion(
            messages=messages,
            model=model or 'gemini-2.5-flash'
        )
        
        return response['choices'][0]['message']['content']
