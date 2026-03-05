"""
ChatsSkill Package - Robutler chat metadata enrichment

Metadata-only skill: fetches the agent's chats from Robutler and adds them
to the agent's info with links to /chats/{uuid} endpoints. No tools exposed.
"""

from .skill import ChatsSkill

__all__ = ["ChatsSkill"]
