"""
WebAgents V2.0 Core Agent Module

Contains the core agent implementation and supporting systems:
- BaseAgent: Main agent class with tool/hook/handoff registration  
- LocalAgentHandoff: Same-instance agent handoff system
- MessageRouter: Central hub for capability-based message routing

Note: DynamicAgentFactory has been moved to the agents server repository.
"""

from .base_agent import BaseAgent
from .handoffs import LocalAgentHandoff, HandoffExecution, create_local_handoff_system
from .router import (
    MessageRouter,
    UAMPEvent,
    RouterContext,
    Handler,
    Observer,
    TransportSink,
    SystemEvents,
    UAMPEventTypes,
    CallbackSink,
    BufferSink,
    matches_subscription,
)

__all__ = [
    "BaseAgent",
    "LocalAgentHandoff",
    "HandoffExecution", 
    "create_local_handoff_system",
    # Router
    "MessageRouter",
    "UAMPEvent",
    "RouterContext",
    "Handler",
    "Observer",
    "TransportSink",
    "SystemEvents",
    "UAMPEventTypes",
    "CallbackSink",
    "BufferSink",
    "matches_subscription",
]
