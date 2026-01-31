"""
MCP Skill - Model Context Protocol Integration

Provides integration with external MCP (Model Context Protocol) servers,
enabling dynamic tool discovery and execution from the MCP ecosystem.
"""

from .skill import MCPSkill, MCPServerConfig, MCPTransport, MCPExecution

__all__ = [
    'MCPSkill',
    'MCPServerConfig',
    'MCPTransport',
    'MCPExecution'
]
