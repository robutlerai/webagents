"""
Browser Skills Module

Skills for browser automation, web search, and DOM manipulation.
"""

from .search import WebSearchSkill
from .automation import BrowserAutomationSkill

__all__ = [
    'WebSearchSkill',
    'BrowserAutomationSkill',
]
