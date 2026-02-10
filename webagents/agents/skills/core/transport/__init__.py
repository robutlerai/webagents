"""
Transport Skills - WebAgents V2.0

Transport skills provide protocol adapters for different agent communication standards.
Each transport exposes endpoints via @http or @websocket decorators and routes
messages through the handoff system via execute_handoff().

Available Transports:
- CompletionsTransportSkill: OpenAI-compatible /chat/completions
- A2ATransportSkill: Google Agent2Agent Protocol
- RealtimeTransportSkill: OpenAI Realtime API (WebSocket)
- ACPTransportSkill: Agent Client Protocol (IDE integration)
- UAMPTransportSkill: Native UAMP protocol (WebSocket)
"""

from .completions.skill import CompletionsTransportSkill
from .a2a.skill import A2ATransportSkill
from .realtime.skill import RealtimeTransportSkill
from .acp.skill import ACPTransportSkill
from .uamp.skill import UAMPTransportSkill

__all__ = [
    'CompletionsTransportSkill',
    'A2ATransportSkill', 
    'RealtimeTransportSkill',
    'ACPTransportSkill',
    'UAMPTransportSkill',
]
