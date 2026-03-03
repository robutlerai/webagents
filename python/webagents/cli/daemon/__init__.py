"""
WebAgents Daemon (webagentsd)

Background daemon for running, scheduling, and exposing agents.
"""

from .server import WebAgentsDaemon, create_daemon
from .manager import AgentManager
from .registry import DaemonRegistry

__all__ = [
    "WebAgentsDaemon",
    "create_daemon",
    "AgentManager",
    "DaemonRegistry",
]
