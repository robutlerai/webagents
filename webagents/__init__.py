"""
WebAgents V2.0 - AI Agent Framework

A comprehensive framework for building AI agents with skills, tools, handoffs, 
and seamless integration with LLM providers.
"""

__version__ = "2.0.0"

# Main exports for easy imports
from .agents.core.base_agent import BaseAgent
from .agents.skills.base import Skill

__all__ = [
    "BaseAgent",
    "Skill",
    "__version__",
] 