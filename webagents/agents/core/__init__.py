"""
WebAgents V2.0 Core Agent Module

Contains the core agent implementation and supporting systems:
- BaseAgent: Main agent class with tool/hook/handoff registration  
- LocalAgentHandoff: Same-instance agent handoff system

Note: DynamicAgentFactory has been moved to the agents server repository.
"""

from .base_agent import BaseAgent
from .handoffs import LocalAgentHandoff, HandoffExecution, create_local_handoff_system

__all__ = [
    "BaseAgent",
    "LocalAgentHandoff",
    "HandoffExecution", 
    "create_local_handoff_system"
]
