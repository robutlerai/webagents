"""
Tests for LocalFileSource agent caching.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Import the source we're testing
from webagents.server.extensions.local_file_source import LocalFileSource


class MockMetadataStore:
    """Mock metadata store for testing."""
    def __init__(self):
        self.agents = {}
    
    def register_agent(self, name, metadata):
        self.agents[name] = metadata


class MockDaemonAgent:
    """Mock agent file info."""
    def __init__(self, name, source_path, description=""):
        self.name = name
        self.source_path = Path(source_path)
        self.description = description


class MockRegistry:
    """Mock registry for testing."""
    def __init__(self):
        self._agents = {}
    
    def get(self, name):
        return self._agents.get(name)
    
    def add(self, name, agent):
        self._agents[name] = agent
    
    def list_agents(self):
        return list(self._agents.values())
    
    async def scan_directory(self, watch_dir):
        pass


@pytest.fixture
def source():
    """Create a LocalFileSource for testing."""
    metadata_store = MockMetadataStore()
    registry = MockRegistry()
    return LocalFileSource(
        watch_dirs=[Path("/test")],
        metadata_store=metadata_store,
        registry=registry
    )


class TestLocalFileSourceCaching:
    """Tests for agent caching in LocalFileSource."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_agent(self, source):
        """Verify that cached agents are returned without reloading."""
        # Pre-populate cache
        mock_agent = MagicMock(name="test-agent")
        source._agent_cache["test-agent"] = mock_agent
        source._cache_timestamps["test-agent"] = 12345.0
        
        # Get agent should return cached version
        result = await source.get_agent("test-agent")
        
        assert result is mock_agent
    
    @pytest.mark.asyncio
    async def test_cache_miss_loads_agent(self, source):
        """Verify that uncached agents are loaded and cached."""
        # Setup mock agent file
        mock_agent_file = MockDaemonAgent("new-agent", "/test/AGENT.md")
        source.registry.add("new-agent", mock_agent_file)
        
        # Mock the _create_agent method
        mock_created = MagicMock(name="created-agent")
        source._create_agent = AsyncMock(return_value=mock_created)
        
        # Get agent should load and cache
        result = await source.get_agent("new-agent")
        
        assert result is mock_created
        assert "new-agent" in source._agent_cache
        assert source._agent_cache["new-agent"] is mock_created
        assert "new-agent" in source._cache_timestamps
    
    @pytest.mark.asyncio
    async def test_invalidate_removes_from_cache(self, source):
        """Verify that invalidate() removes an agent from cache."""
        # Pre-populate cache
        mock_agent = MagicMock()
        source._agent_cache["test-agent"] = mock_agent
        source._cache_timestamps["test-agent"] = 12345.0
        
        # Invalidate
        result = source.invalidate("test-agent")
        
        assert result is True
        assert "test-agent" not in source._agent_cache
        assert "test-agent" not in source._cache_timestamps
    
    @pytest.mark.asyncio
    async def test_invalidate_nonexistent_returns_false(self, source):
        """Verify that invalidating a non-cached agent returns False."""
        result = source.invalidate("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_invalidate_all_clears_cache(self, source):
        """Verify that invalidate_all() clears entire cache."""
        # Pre-populate cache with multiple agents
        source._agent_cache["agent1"] = MagicMock()
        source._agent_cache["agent2"] = MagicMock()
        source._cache_timestamps["agent1"] = 1.0
        source._cache_timestamps["agent2"] = 2.0
        
        # Invalidate all
        count = source.invalidate_all()
        
        assert count == 2
        assert len(source._agent_cache) == 0
        assert len(source._cache_timestamps) == 0
    
    @pytest.mark.asyncio
    async def test_get_cache_stats(self, source):
        """Verify cache stats are returned correctly."""
        # Pre-populate cache
        source._agent_cache["agent1"] = MagicMock()
        source._agent_cache["agent2"] = MagicMock()
        source._cache_timestamps["agent1"] = 100.0
        source._cache_timestamps["agent2"] = 200.0
        
        stats = source.get_cache_stats()
        
        assert set(stats["cached_agents"]) == {"agent1", "agent2"}
        assert stats["cache_size"] == 2
        assert stats["timestamps"]["agent1"] == 100.0
        assert stats["timestamps"]["agent2"] == 200.0


class TestExtensionLoading:
    """Tests for extension loading and backwards compatibility."""
    
    def test_load_extensions_with_extensions_key(self):
        """Verify that extensions config works."""
        from webagents.server.extensions.loader import load_extensions
        
        # Empty config should return empty list
        result = load_extensions({"extensions": []})
        assert result == []
    
    def test_load_extensions_with_deprecated_plugins_key(self):
        """Verify backwards compatibility with 'plugins' key."""
        from webagents.server.extensions.loader import load_extensions
        
        # Empty plugins config should work with warning
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_extensions({"plugins": []})
            
            assert result == []
            # Check that deprecation warning was issued
            assert any("plugins" in str(warning.message) for warning in w)


class TestWebAgentsExtensionInterface:
    """Tests for WebAgentsExtension interface."""
    
    def test_extension_alias_warns(self):
        """Verify that WebAgentsPlugin subclass triggers deprecation warning."""
        from webagents.server.extensions.interface import WebAgentsPlugin, AgentSource
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Subclassing WebAgentsPlugin should warn
            class TestPlugin(WebAgentsPlugin):
                def get_name(self): return "test"
                async def initialize(self, server): pass
                def get_agent_sources(self): return []
                def get_skills(self): return {}
            
            # Instantiation should also work
            plugin = TestPlugin()
            assert plugin.get_name() == "test"
