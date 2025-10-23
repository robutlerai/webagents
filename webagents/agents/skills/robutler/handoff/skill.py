"""
Agent Handoff Skill - WebAgents V2.0

Enables handoff to remote agents via NLI (Network LLM Interface).
Provides streaming support for remote agent responses.
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator

from webagents.agents.skills.base import Skill, Handoff
from webagents.agents.tools.decorators import handoff
from webagents.utils.logging import get_logger


class AgentHandoffSkill(Skill):
    """
    Skill for handing off to remote agents with streaming support
    
    Uses NLI skill to communicate with remote agents and stream responses.
    Automatically normalizes NLI responses to OpenAI-compatible format.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self.default_agent_url = config.get('agent_url') if config else None
    
    async def initialize(self, agent):
        """Initialize agent handoff skill"""
        self.agent = agent
        self.logger = get_logger('skill.handoff.agent', agent.name)
        
        # Get NLI skill for remote communication
        self.nli_skill = None  # Will be set when needed
        
        # Register handoff using decorator
        # Priority=20 (lower than local LLM at priority=10)
        agent.register_handoff(
            Handoff(
                target="remote_agent",
                description="Hand off to remote specialist agent for tasks requiring specific expertise",
                scope="all",
                metadata={
                    'function': self.remote_agent_handoff,
                    'priority': 20,
                    'is_generator': True  # Async generator for streaming
                }
            ),
            source="agent_handoff"
        )
        
        self.logger.info(f"📨 Registered remote agent handoff (priority=20)")
    
    def _ensure_nli_skill(self):
        """Ensure NLI skill is available"""
        if not self.nli_skill:
            self.nli_skill = self.agent.skills.get('nli')
            if not self.nli_skill:
                raise ValueError("NLI skill required for remote agent handoffs")
    
    @handoff(
        name="remote_agent",
        prompt="Hand off to remote specialist agent for complex or specialized tasks",
        priority=20
    )
    async def remote_agent_handoff(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        agent_url: Optional[str] = None,
        context=None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream responses from remote agent via NLI
        
        Args:
            messages: Conversation messages
            tools: Available tools to pass to remote agent
            agent_url: URL of remote agent (overrides default)
            context: Request context (auto-injected)
            **kwargs: Additional arguments
        
        Yields:
            OpenAI-compatible streaming chunks
        """
        target_url = agent_url or self.default_agent_url
        if not target_url:
            # Try to get from context or raise error
            if context:
                target_url = context.get('handoff_agent_url')
            
            if not target_url:
                raise ValueError(
                    "agent_url required for remote handoff. "
                    "Provide via config, parameter, or context."
                )
        
        self._ensure_nli_skill()
        
        self.logger.info(f"🔄 Starting remote agent handoff to: {target_url}")
        
        # Stream from remote agent via NLI
        try:
            async for chunk in self.nli_skill.stream_message(
                agent_url=target_url,
                messages=messages,
                tools=tools
            ):
                # Normalize NLI chunk to OpenAI format
                normalized_chunk = self._normalize_nli_chunk(chunk)
                yield normalized_chunk
        
        except Exception as e:
            self.logger.error(f"Remote handoff failed: {e}")
            # Yield error chunk
            yield self._create_error_chunk(str(e))
    
    def _normalize_nli_chunk(self, nli_chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Convert NLI streaming chunk to OpenAI format
        
        Args:
            nli_chunk: Chunk from NLI stream
        
        Returns:
            OpenAI-compatible streaming chunk
        """
        # If already in OpenAI format, pass through
        if 'choices' in nli_chunk and isinstance(nli_chunk.get('choices'), list):
            if nli_chunk['choices'] and 'delta' in nli_chunk['choices'][0]:
                return nli_chunk
        
        # Convert from NLI format to OpenAI streaming format
        # NLI format might be: {"content": "...", "finish_reason": "stop", ...}
        content = nli_chunk.get('content', '')
        finish_reason = nli_chunk.get('finish_reason')
        tool_calls = nli_chunk.get('tool_calls')
        
        chunk = {
            "id": nli_chunk.get('id') or f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": nli_chunk.get('created') or int(time.time()),
            "model": nli_chunk.get('model', 'remote-agent'),
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant" if content or tool_calls else None,
                },
                "finish_reason": finish_reason
            }]
        }
        
        # Add content if present
        if content:
            chunk["choices"][0]["delta"]["content"] = content
        
        # Add tool calls if present
        if tool_calls:
            chunk["choices"][0]["delta"]["tool_calls"] = tool_calls
        
        # Add usage if present (final chunk)
        if 'usage' in nli_chunk:
            chunk['usage'] = nli_chunk['usage']
        
        return chunk
    
    def _create_error_chunk(self, error_message: str) -> Dict[str, Any]:
        """Create error chunk in OpenAI format
        
        Args:
            error_message: Error message to include
        
        Returns:
            OpenAI-compatible error chunk
        """
        return {
            "id": f"chatcmpl-error-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "remote-agent-error",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": f"Error communicating with remote agent: {error_message}"
                },
                "finish_reason": "stop"
            }]
        }

