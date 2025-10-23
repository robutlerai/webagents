"""
LiteLLM Skill Package - WebAgents V2.0

Cross-provider LLM routing with support for OpenAI, Anthropic, XAI/Grok, and more.
Provides unified interface for multiple LLM providers with automatic fallbacks.
"""

from .skill import LiteLLMSkill

__all__ = ["LiteLLMSkill"]
