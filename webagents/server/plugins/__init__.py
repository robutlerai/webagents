"""
WebAgents Plugin System

Plugin interfaces and base classes for extending webagentsd.
"""

from .interface import AgentSource, WebAgentsPlugin
from .loader import load_plugins
from .local_file_source import LocalFileSource

__all__ = [
    "AgentSource",
    "WebAgentsPlugin",
    "load_plugins",
    "LocalFileSource",
]
