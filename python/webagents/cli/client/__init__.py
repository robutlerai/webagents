"""
WebAgents CLI Client

Client for communicating with webagentsd.
"""

from .daemon_client import DaemonClient
from .auto_start import ensure_daemon_running

__all__ = ["DaemonClient", "ensure_daemon_running"]
