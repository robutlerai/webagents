"""
WebAgents CLI Client

Client for communicating with a running `webagents daemon`.
"""

from .daemon_client import DaemonClient

__all__ = ["DaemonClient"]
