"""
DiscoverySkill Package - Simplified WebAgents Platform Integration

Agent discovery skill for WebAgents platform.
Provides intent-based agent discovery and intent publishing capabilities.
"""

from .skill import (
    DiscoverySkill,
    DiscoveryResult
)

__all__ = [
    "DiscoverySkill",
    "DiscoveryResult"
]
