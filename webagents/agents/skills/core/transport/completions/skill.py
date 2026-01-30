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
    
    @http("/chat/completions", method="post")
    async def chat_completions(
        self,
        messages: List[Dict[str, Any]] = None,
        stream: bool = True,
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
            # Stream raw SSE response unchanged - client parses everything
            async for chunk in self.execute_handoff(messages, tools=tools, **handoff_kwargs):
                yield f"data: {json.dumps(chunk)}\n\n"
            
            yield "data: [DONE]\n\n"
        else:
            # Collect all chunks and return as single response
            chunks = []
            async for chunk in self.execute_handoff(messages, tools=tools, **handoff_kwargs):
                chunks.append(chunk)
            
            # Merge chunks into final response
            if chunks:
                final_response = self._merge_streaming_chunks(chunks)
                yield f"data: {json.dumps(final_response)}\n\n"
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
                if "tool_calls" in delta:
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
