"""
DiscoverySkill - Unified agent and content discovery for the Robutler network.

Searches across agents, intents, posts, channels, tags, and users
via the Robutler /api/discovery endpoint.
"""

from .skill import DiscoverySkill

__all__ = [
    "DiscoverySkill",
]
