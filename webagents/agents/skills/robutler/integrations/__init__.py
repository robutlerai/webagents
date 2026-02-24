"""
Integrations Skill - MCP Bridge Integration for External Providers

Provides dynamic tool registration from external integrations (Google, X, etc.)
via the Roborum MCP bridge.
"""

from .skill import IntegrationsSkill, IntegrationTool

__all__ = [
    'IntegrationsSkill',
    'IntegrationTool',
]
