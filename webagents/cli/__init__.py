"""
WebAgents CLI - Premium command-line interface for AI agents.

Build, run, and discover AI agents from your terminal.

This module contains all CLI-related functionality:
- loader: AGENT.md parsing and context hierarchy
- state: Local state management (.webagents/)
- daemon: Background daemon (webagentsd)
- platform: robutler.ai integration
- templates: Agent templates
- commands: CLI commands
- repl: Interactive REPL
- ui: Rich terminal UI
"""

from .main import app, cli

# Re-export submodules
from . import loader
from . import state
from . import daemon
from . import platform
from . import templates
from . import commands
from . import repl
from . import ui

__all__ = [
    "app",
    "cli",
    "loader",
    "state", 
    "daemon",
    "platform",
    "templates",
    "commands",
    "repl",
    "ui",
]
