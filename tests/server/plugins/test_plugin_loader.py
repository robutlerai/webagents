"""
Tests for plugin loader
"""

import pytest
from webagents.server.plugins.loader import load_plugins
from webagents.server.plugins.interface import WebAgentsPlugin, AgentSource


class MockPlugin(WebAgentsPlugin):
    """Mock plugin for testing"""
    
    def __init__(self, config):
        self.config = config
    
    def get_name(self):
        return "mock"
    
    async def initialize(self, server):
        pass
    
    def get_agent_sources(self):
        return []
    
    def get_skills(self):
        return {}


def test_load_plugins_empty_config():
    """Test loading with empty config"""
    config = {"plugins": []}
    plugins = load_plugins(config)
    assert len(plugins) == 0


def test_load_plugins_missing_module():
    """Test loading with non-existent module"""
    config = {
        "plugins": [
            {
                "module": "nonexistent_module",
                "class": "NonexistentPlugin",
                "config": {}
            }
        ]
    }
    plugins = load_plugins(config)
    # Should not raise, just log warning and return empty list
    assert len(plugins) == 0


def test_load_plugins_invalid_class():
    """Test loading with invalid class name"""
    config = {
        "plugins": [
            {
                "module": "webagents.server.plugins.loader",
                "class": "NonexistentClass",
                "config": {}
            }
        ]
    }
    plugins = load_plugins(config)
    # Should not raise, just log warning
    assert len(plugins) == 0
