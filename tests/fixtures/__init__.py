"""
Test fixtures for WebAgents testing

This package contains mock servers, test data, and utilities for testing
various WebAgents components, particularly the MCP integration.
"""

from .mock_mcp_servers import (
    MathMCPServer,
    FileMCPServer,
    DatabaseMCPServer,
    MockTool,
    MockResource,
    MockPrompt
)

__all__ = [
    'MathMCPServer',
    'FileMCPServer', 
    'DatabaseMCPServer',
    'MockTool',
    'MockResource',
    'MockPrompt'
] 