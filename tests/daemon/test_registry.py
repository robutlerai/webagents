"""
Daemon Registry Tests

Test agent registry functionality.
"""

import pytest
from pathlib import Path
import tempfile

from webagents.cli.daemon.registry import DaemonRegistry, DaemonAgent
from webagents.cli.loader.agent_md import AgentFile


class TestDaemonRegistry:
    """Test daemon registry."""
    
    def test_register_agent(self):
        """Test registering an agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "AGENT.md"
            agent_path.write_text("""---
name: test-agent
description: Test description
intents:
  - test things
---
Test instructions
""")
            agent_file = AgentFile(agent_path)
            
            registry = DaemonRegistry()
            registered = registry.register(agent_file)
            
            assert registered.name == "test-agent"
            assert registered.description == "Test description"
            assert "test things" in registered.intents
            assert registered.status == "registered"
    
    def test_unregister_agent(self):
        """Test unregistering an agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "AGENT.md"
            agent_path.write_text("---\nname: test\n---\nTest")
            agent_file = AgentFile(agent_path)
            
            registry = DaemonRegistry()
            registry.register(agent_file)
            
            result = registry.unregister("test")
            
            assert result == True
            assert registry.get("test") is None
    
    def test_get_agent(self):
        """Test getting agent by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "AGENT.md"
            agent_path.write_text("---\nname: my-agent\n---\nTest")
            agent_file = AgentFile(agent_path)
            
            registry = DaemonRegistry()
            registry.register(agent_file)
            
            agent = registry.get("my-agent")
            
            assert agent is not None
            assert agent.name == "my-agent"
    
    def test_list_agents(self):
        """Test listing agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["a", "b", "c"]:
                path = Path(tmpdir) / f"AGENT-{name}.md"
                path.write_text(f"---\nname: {name}\n---\n{name}")
                registry = DaemonRegistry()
            
            for name in ["a", "b", "c"]:
                path = Path(tmpdir) / f"AGENT-{name}.md"
                agent_file = AgentFile(path)
                registry.register(agent_file)
            
            agents = registry.list_agents()
            
            assert len(agents) == 3
    
    def test_filter_by_namespace(self):
        """Test filtering agents by namespace."""
        registry = DaemonRegistry()
        
        # Add agents with different namespaces
        agent1 = DaemonAgent(name="a", namespace="ns1", source_path="/a")
        agent2 = DaemonAgent(name="b", namespace="ns2", source_path="/b")
        agent3 = DaemonAgent(name="c", namespace="ns1", source_path="/c")
        
        registry.agents["a"] = agent1
        registry.agents["b"] = agent2
        registry.agents["c"] = agent3
        
        ns1_agents = registry.list_agents(namespace="ns1")
        
        assert len(ns1_agents) == 2
        assert all(a.namespace == "ns1" for a in ns1_agents)


class TestDaemonAgent:
    """Test DaemonAgent model."""
    
    def test_agent_creation(self):
        """Test creating a daemon agent."""
        agent = DaemonAgent(
            name="test",
            namespace="local",
            source_path="/path/to/agent.md",
            intents=["do things"],
            cron="0 9 * * *",
        )
        
        assert agent.name == "test"
        assert agent.namespace == "local"
        assert agent.status == "registered"
        assert agent.cron == "0 9 * * *"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        agent = DaemonAgent(
            name="test",
            namespace="local",
            source_path="/path",
        )
        
        data = agent.to_dict()
        
        assert data["name"] == "test"
        assert data["namespace"] == "local"
        assert "status" in data
