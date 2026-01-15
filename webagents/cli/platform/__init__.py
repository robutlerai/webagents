"""
WebAgents Platform Integration

Connect to robutler.ai for discovery, namespaces, and sync.
"""

from .auth import login, logout, is_authenticated, get_current_user
from .api import RobutlerAPI

__all__ = [
    "login",
    "logout", 
    "is_authenticated",
    "get_current_user",
    "RobutlerAPI",
]
