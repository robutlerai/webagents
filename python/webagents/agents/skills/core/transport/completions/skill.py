"""
Completions Transport Skill - WebAgents V2.0

OpenAI-compatible /chat/completions endpoint as a transport skill.
This wraps the existing completions behavior and routes through handoffs.

Transport is independent from session management - session logging is
handled by SessionManagerSkill hooks (on_connection, on_message, etc.)

Uses UAMP (Universal Agentic Message Protocol) for internal message representation.
"""

import json
from typing import Dict, Any, List, Optional, AsyncGenerator, TYPE_CHECKING

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import http
from webagents.uamp import (
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ContentDelta,
    ResponseOutput,
    UsageStats,
)
from .uamp_adapter import CompletionsUAMPAdapter

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent


class CompletionsTransportSkill(Skill):
    """
    OpenAI-compatible chat completions transport.
    
    Exposes /chat/completions endpoint that routes through the agent's
    handoff system for LLM processing.
    
    This transport is independent from session management. Session
    logging is handled automatically by SessionManagerSkill hooks.
    
    Example:
        agent = BaseAgent(
            name="my-agent",
            skills=[CompletionsTransportSkill()]
        )
        
        # POST /agents/my-agent/chat/completions
        # {
        #     "messages": [{"role": "user", "content": "Hello"}],
        #     "stream": true
        # }
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self._adapter = CompletionsUAMPAdapter()
    
    async def initialize(self, agent: 'BaseAgent') -> None:
        """Initialize the completions transport"""
        self.agent = agent
    
    @http("/capabilities", method="get")
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get agent/model capabilities.
        
        Returns UAMP ModelCapabilities for the active LLM.
        Enables clients to discover supported features before making requests.
        
        Example response:
        {
            "model_id": "gpt-4o",
            "provider": "openai",
            "modalities": ["text", "image"],
            "supports_streaming": true,
            "image": { "formats": ["jpeg", "png"], "supports_pdf": true }
        }
        """
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        return self._get_model_capabilities(agent)
    
    @http("/models", method="get")
    async def list_models(self) -> Dict[str, Any]:
        """
        List available models (OpenAI-compatible).
        
        Returns models with their capabilities.
        """
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        caps = self._get_model_capabilities(agent)
        
        return {
            "object": "list",
            "data": [{
                "id": caps.get("model_id", "default"),
                "object": "model",
                "created": 0,
                "owned_by": caps.get("provider", "webagents"),
                # Extension: include capabilities
                "capabilities": caps
            }]
        }
    
    def _get_model_capabilities(self, agent) -> Dict[str, Any]:
        """Get UAMP model capabilities from agent's LLM skills."""
        if not agent or not hasattr(agent, 'skills'):
            return {"modalities": ["text"], "supports_streaming": True}
        
        # Look for LLM skills with UAMP adapters
        for skill in agent.skills.values():
            if hasattr(skill, '_adapter') and hasattr(skill._adapter, 'get_capabilities'):
                caps = skill._adapter.get_capabilities()
                return {
                    "model_id": caps.model_id,
                    "provider": caps.provider,
                    "modalities": caps.modalities,
                    "supports_streaming": caps.supports_streaming,
                    "supports_thinking": caps.supports_thinking,
                    "context_window": caps.context_window,
                    "max_output_tokens": caps.max_output_tokens,
                    "image": {
                        "formats": caps.image.formats,
                        "detail_levels": caps.image.detail_levels,
                    } if caps.image else None,
                    "audio": {
                        "input_formats": caps.audio.input_formats,
                        "output_formats": caps.audio.output_formats,
                        "supports_realtime": caps.audio.supports_realtime,
                    } if caps.audio else None,
                    "file": {
                        "supports_pdf": caps.file.supports_pdf,
                        "supported_mime_types": caps.file.supported_mime_types,
                    } if caps.file else None,
                    "tools": {
                        "supports_tools": caps.tools.supports_tools,
                        "built_in_tools": caps.tools.built_in_tools,
                    } if caps.tools else None,
                }
        
        return {"modalities": ["text"], "supports_streaming": True}
    
    # `stream` DEFAULTS TO FALSE, per the OpenAI API (2026-09-24). It was True
    # here after the static route was fixed to False on 2026-09-23, and this
    # route OVERRIDES that one for every daemon-served agent, so the fix never
    # reached them: the official SDKs omit `stream` on a plain call and got SSE.
    # Every in-repo caller sends it explicitly (DaemonClient, both NLI skills,
    # the test runner), so nothing relied on the old default.
    @http("/chat/completions", method="post")
    async def chat_completions(
        self,
        messages: List[Dict[str, Any]] = None,
        stream: bool = False,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Dict[str, Any]] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Any] = None,
        n: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        user: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        OpenAI-compatible chat completions endpoint.
        
        Supports all standard OpenAI parameters:
        - messages: List of messages (required)
        - stream: Whether to stream (default: True)
        - model: Model to use
        - temperature: Sampling temperature (0-2)
        - max_tokens: Maximum tokens to generate
        - tools: Tool definitions
        - tool_choice: Tool selection mode
        - response_format: Response format (e.g., {"type": "json_object"})
        - top_p: Nucleus sampling parameter
        - frequency_penalty: Frequency penalty (-2 to 2)
        - presence_penalty: Presence penalty (-2 to 2)
        - stop: Stop sequences
        - n: Number of completions to generate
        - logprobs: Whether to return log probabilities
        - top_logprobs: Number of top log probabilities
        - user: End-user identifier for tracking
        - seed: Seed for deterministic outputs
        """
        messages = messages or []
        
        # Build kwargs for handoff, passing through all parameters
        handoff_kwargs = {}
        if model is not None:
            handoff_kwargs['model'] = model
        if temperature is not None:
            handoff_kwargs['temperature'] = temperature
        if max_tokens is not None:
            handoff_kwargs['max_tokens'] = max_tokens
        if tool_choice is not None:
            handoff_kwargs['tool_choice'] = tool_choice
        if response_format is not None:
            handoff_kwargs['response_format'] = response_format
        if top_p is not None:
            handoff_kwargs['top_p'] = top_p
        if frequency_penalty is not None:
            handoff_kwargs['frequency_penalty'] = frequency_penalty
        if presence_penalty is not None:
            handoff_kwargs['presence_penalty'] = presence_penalty
        if stop is not None:
            handoff_kwargs['stop'] = stop
        if n is not None:
            handoff_kwargs['n'] = n
        if logprobs is not None:
            handoff_kwargs['logprobs'] = logprobs
        if top_logprobs is not None:
            handoff_kwargs['top_logprobs'] = top_logprobs
        if user is not None:
            handoff_kwargs['user'] = user
        if seed is not None:
            handoff_kwargs['seed'] = seed
        
        # Merge any additional kwargs
        handoff_kwargs.update(kwargs)
        
        if stream:
            full_text_parts = []
            async for chunk in self.execute_handoff(messages, tools=tools, **handoff_kwargs):
                yield f"data: {json.dumps(chunk)}\n\n"
                # Accumulate text for response signing
                try:
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            full_text_parts.append(content)
                except Exception:
                    pass
            
            yield "data: [DONE]\n\n"
            
            # Emit optional response signature (RS256 JWT for non-repudiation)
            sig = self._sign_response("".join(full_text_parts), json.dumps(messages or []))
            if sig:
                yield f"event: response_signature\ndata: {json.dumps({'signature': sig})}\n\n"
        else:
            # ONE JSON BODY, NOT SSE (2026-09-24). This branch yielded the
            # merged completion as a `data:` event followed by `[DONE]`, so a
            # `stream: false` request, which is the OpenAI default and what the
            # official SDKs send when streaming is not asked for, received
            # `text/event-stream` and could not parse it. The server returns a
            # Response yielded first as the whole answer (`server/core/app.py`,
            # "A FINISHED RESPONSE"). Every chunk is collected BEFORE the yield,
            # so the handoff, including any payment settlement inside it,
            # completes in full; nothing after the yield is expected to run.
            from starlette.responses import JSONResponse

            chunks = []
            async for chunk in self.execute_handoff(messages, tools=tools, **handoff_kwargs):
                chunks.append(chunk)

            merged = self._merge_streaming_chunks(chunks) if chunks else {
                # Still a completion a client can parse, not `{}`.
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }],
                "usage": None,
            }
            yield JSONResponse(content=merged)
    
    @http("/uamp/completions", method="post")
    async def uamp_completions(
        self,
        messages: List[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        UAMP-native completions endpoint.
        
        Unlike /chat/completions which returns OpenAI-format chunks,
        this endpoint uses full UAMP flow:
        1. Convert OpenAI messages to UAMP events
        2. Process through agent.process_uamp()
        3. Return UAMP server events directly
        
        Useful for clients that want UAMP-native communication.
        
        Example request:
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [...]
        }
        
        Example response (SSE stream of UAMP events):
        data: {"type": "response.created", "response_id": "..."}
        data: {"type": "response.delta", "delta": {"type": "text", "text": "Hi"}}
        data: {"type": "response.done", "response_id": "..."}
        """
        from webagents.uamp import InputTextEvent, InputImageEvent, ResponseCreateEvent
        
        messages = messages or []
        
        # Get agent reference
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        if not agent:
            yield f"data: {json.dumps({'type': 'error', 'error': 'No agent available'})}\n\n"
            return
        
        # Convert OpenAI messages to UAMP events
        uamp_events = self._adapter.to_uamp({"messages": messages})
        
        # Process through agent's native UAMP method
        async for uamp_event in agent.process_uamp(uamp_events, tools=tools, **kwargs):
            # Yield UAMP event as SSE
            yield f"data: {json.dumps(uamp_event.to_dict())}\n\n"
        
        yield "data: [DONE]\n\n"
    
    def _merge_streaming_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge streaming chunks into a complete response"""
        if not chunks:
            return {}
        
        # Start with the structure from the first chunk
        result = {
            "id": chunks[0].get("id", ""),
            "object": "chat.completion",
            "created": chunks[0].get("created", 0),
            "model": chunks[0].get("model", ""),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "finish_reason": None
            }],
            "usage": None
        }
        
        # Accumulate content from all chunks
        content_parts = []
        tool_calls = []
        
        for chunk in chunks:
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    content_parts.append(delta["content"])
                # `.get`, not `in` (2026-09-24). The OpenAI SDK's chunks carry
                # the key with a null value (`"tool_calls": null`) on every
                # plain-text delta, and `extend(None)` raised "'NoneType' object
                # is not iterable". The route answered 500 for EVERY
                # non-streaming chat with a daemon-served agent, a plain text
                # reply included; streaming never merges, so only stream=false
                # broke. Found by the sandbox end-to-end run.
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])
                if choices[0].get("finish_reason"):
                    result["choices"][0]["finish_reason"] = choices[0]["finish_reason"]
            
            # Capture usage if present (usually in last chunk)
            if chunk.get("usage"):
                result["usage"] = chunk["usage"]
        
        result["choices"][0]["message"]["content"] = "".join(content_parts)
        if tool_calls:
            result["choices"][0]["message"]["tool_calls"] = tool_calls
        
        return result
