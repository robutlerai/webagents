"""
WebAgents - AI Agent Framework

A comprehensive framework for building AI agents with skills, tools, handoffs, 
and seamless integration with LLM providers.

CLI: Premium command-line interface for AI agents.
"""

# Single-source the version on the installed distribution metadata.
# A hardcoded literal here drifted from pyproject (0.3.4 vs 0.3.5) because the
# release workflow's sed only rewrote pyproject. When running from a source
# tree that was never installed, fall back to the pyproject value.
def _resolve_version() -> str:
    try:
        from importlib.metadata import version as _dist_version
        return _dist_version("webagents")
    except Exception:
        return "0.3.5"


__version__ = _resolve_version()

# Main exports for easy imports
from .agents.core.base_agent import BaseAgent
from .agents.skills.base import Skill
# The server factory, re-exported at the package root. A shorter IMPORT is not
# obfuscation; the wrappers that used to live here (`connect`/`host`) were,
# because they hid the lifecycle. Everything they added — the agent card at the
# origin with metadata.publicKey, the heartbeat, starting an attached
# PortalConnectSkill — now belongs to `create_server` itself.
from .server.core.app import create_server
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
    "create_server",
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
