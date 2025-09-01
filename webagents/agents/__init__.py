"""
WebAgents V2.0 Agents Module

Core agent system including BaseAgent, skills, tools, handoffs, and lifecycle management.
"""

from .core.base_agent import BaseAgent
from .skills.base import Skill

__all__ = [
    "BaseAgent", 
    "Skill",
] 