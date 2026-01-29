"""
Plugin Skill - Claude Code Compatible Plugin System

Provides plugin management with marketplace discovery, fuzzy search,
GitHub star ranking, and dynamic tool registration.

Features:
- Claude Code plugin.json compatibility
- Marketplace discovery from claudemarketplaces.com
- Fuzzy search with rapidfuzz and GitHub star ranking
- SKILL.md parser with frontmatter and $ARGUMENTS substitution
- Hook and command execution
- Periodic background marketplace refresh
"""

from .skill import PluginSkill
from .loader import PluginLoader, Plugin
from .marketplace import MarketplaceClient
from .executor import PluginExecutor
from .schema import PluginManifest, validate_manifest

__all__ = [
    "PluginSkill",
    "PluginLoader",
    "Plugin",
    "MarketplaceClient",
    "PluginExecutor",
    "PluginManifest",
    "validate_manifest",
]
