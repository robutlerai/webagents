"""
WebAgents - AI Agent Framework

A comprehensive framework for building AI agents with skills, tools, handoffs, 
and seamless integration with LLM providers.

CLI: Premium command-line interface for AI agents.
"""

__version__ = "0.3.0"

# Main exports for easy imports
from .agents.core.base_agent import BaseAgent
from .agents.skills.base import Skill
from .agents.tools.decorators import tool, prompt, hook, http, handoff, widget, command, observe, websocket
from .agents.widgets import WidgetTemplateRenderer

# CLI exports
from .cli import app as cli_app

# Loader exports (from cli.loader)
from .cli.loader import AgentFile, AgentMetadata, AgentLoader, MergedAgent

# State exports (from cli.state)
from .cli.state import LocalState, LocalRegistry, SessionManager

__all__ = [
    # Core
    "BaseAgent",
    "Skill",
    "tool",
    "prompt",
    "hook",
    "http",
    "handoff",
    "widget",
    "command",
    "observe",
    "websocket",
    "WidgetTemplateRenderer",
    # CLI
    "cli_app",
    # Loader
    "AgentFile",
    "AgentMetadata",
    "AgentLoader",
    "MergedAgent",
    # State
    "LocalState",
    "LocalRegistry",
    "SessionManager",
    # Version
    "__version__",
]
