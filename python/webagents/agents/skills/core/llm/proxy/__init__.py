"""
LLM Proxy Skill Package - WebAgents V2.0

Connects to the UAMP LLM proxy via WebSocket for daemon agents
running inside the cluster. No API keys needed - the proxy handles
provider selection and payment.
"""

from .skill import LLMProxySkill

__all__ = ["LLMProxySkill"]
