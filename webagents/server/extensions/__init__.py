"""
WebAgents Extension System

Extension interfaces and base classes for extending webagentsd.

Note: Previously called "plugins", renamed to "extensions" to avoid
confusion with Claude Code plugins (supported via SkillPlugin).
"""

from .interface import AgentSource, WebAgentsExtension, WebAgentsPlugin
from .loader import load_extensions, load_plugins
from .local_file_source import LocalFileSource
from .cache_invalidation import RedisCacheInvalidator

__all__ = [
    # Core interfaces
    "AgentSource",
    "WebAgentsExtension",
    "load_extensions",
    "LocalFileSource",
    # Cache invalidation
    "RedisCacheInvalidator",
    # Deprecated aliases (for backwards compatibility)
    "WebAgentsPlugin",
    "load_plugins",
]
