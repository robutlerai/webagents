"""
Local Registry Tests

Test agent registry in .webagents/registry.json.
"""

import pytest
from pathlib import Path
import tempfile

from webagents.cli.state.registry import LocalRegistry, RegisteredAgent
from webagents.cli.loader.agent_md import AgentFile


class TestLocalRegistry:
    """Test local registry."""
    
    def test_register_agent(self):
        """Test registering an agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.json"
            
            # Create agent file
            agent_path = Path(tmpdir) / "AGENT.md"
            agent_path.write_text("""---
name: test
description: Test agent
intents:
  - test things
---
Test
""")
            agent_file = AgentFile(agent_path)
            
            registry = LocalRegistry(registry_path)
            registered = registry.register(agent_file)
            
            assert registered.name == "test"
            assert "test" in registry.agents
    
    def test_persistence(self):
        """Test registry persists to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.json"
            
            # Create and register
            agent_path = Path(tmpdir) / "AGENT.md"
            agent_path.write_text("---\nname: persisted\n---\nTest")
            agent_file = AgentFile(agent_path)
            
            registry1 = LocalRegistry(registry_path)
            registry1.register(agent_file)
            
            # Load new registry from same file
            registry2 = LocalRegistry(registry_path)
            
            assert "persisted" in registry2.agents
    
    def test_unregister(self):
        """Test unregistering an agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.json"
            
            agent_path = Path(tmpdir) / "AGENT.md"
            agent_path.write_text("---\nname: to-remove\n---\nTest")
            agent_file = AgentFile(agent_path)
            
            registry = LocalRegistry(registry_path)
            registry.register(agent_file)
            
            result = registry.unregister("to-remove")
            
            assert result == True
            assert "to-remove" not in registry.agents
    
    def test_scan_directory(self):
        """Test scanning directory for agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "AGENT.md").write_text("---\nname: a\n---\nA")
            (tmppath / "AGENT-b.md").write_text("---\nname: b\n---\nB")
            
            subdir = tmppath / "sub"
            subdir.mkdir()
            (subdir / "AGENT.md").write_text("---\nname: c\n---\nC")
            
            registry = LocalRegistry(tmppath / "registry.json")
            paths = registry.scan_directory(tmppath, recursive=True)
            
            assert len(paths) == 3
    
    def test_sync_from_directory(self):
        """Test syncing registry from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "AGENT.md").write_text("---\nname: synced\n---\nTest")
            
            registry = LocalRegistry(tmppath / "registry.json")
            count = registry.sync_from_directory(tmppath)
            
            assert count == 1
            assert "synced" in registry.agents


class TestRegisteredAgent:
    """Test RegisteredAgent model."""
    
    def test_creation(self):
        """Test creating a registered agent."""
        agent = RegisteredAgent(
            name="test",
            namespace="local",
            source_path="/path/to/agent.md",
        )
        
        assert agent.name == "test"
        assert agent.status == "registered"
        assert agent.sync_status == "local"
    
    def test_timestamps(self):
        """Test automatic timestamps."""
        agent = RegisteredAgent(
            name="test",
            source_path="/path",
        )
        
        assert agent.registered_at is not None
