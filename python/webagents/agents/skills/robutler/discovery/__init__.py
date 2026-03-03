"""
DiscoverySkill - Unified agent and content discovery for the Roborum network.

Searches across agents, intents, posts, channels, tags, and users
via the Roborum /api/discovery endpoint.
"""

from .skill import DiscoverySkill

__all__ = [
    "DiscoverySkill",
]
