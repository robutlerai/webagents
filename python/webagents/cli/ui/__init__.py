"""
WebAgents CLI UI Components

Rich terminal UI with splash screen, panels, and streaming.
"""

from .splash import print_splash, print_status_bar
from .console import WebAgentsConsole

__all__ = ["print_splash", "print_status_bar", "WebAgentsConsole"]
