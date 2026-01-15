"""
AGENT.md Parser Tests

Test parsing of AGENT.md and AGENT-*.md files.
"""

import pytest
from pathlib import Path
import tempfile

from webagents.cli.loader.agent_md import (
    AgentFile,
    parse_frontmatter,
    parse_agent_file,
    find_agent_files,
    create_agent_file,
)
from webagents.cli.loader.schema import AgentMetadata


class TestParseFrontmatter:
    """Test YAML frontmatter parsing."""
    
    def test_parse_valid_frontmatter(self):
        """Test parsing valid YAML frontmatter."""
        content = """---
name: test-agent
description: A test agent
model: openai/gpt-4o
---

# Test Agent

Instructions here.
"""
        yaml_data, body = parse_frontmatter(content)
        
        assert yaml_data["name"] == "test-agent"
        assert yaml_data["description"] == "A test agent"
        assert yaml_data["model"] == "openai/gpt-4o"
        assert "# Test Agent" in body
    
    def test_parse_no_frontmatter(self):
        """Test parsing content without frontmatter."""
        content = "# Just markdown\n\nNo YAML header."
        
        yaml_data, body = parse_frontmatter(content)
        
        assert yaml_data == {}
        assert "# Just markdown" in body
    
    def test_parse_empty_frontmatter(self):
        """Test parsing empty frontmatter."""
        content = """---
---

Body content.
"""
        yaml_data, body = parse_frontmatter(content)
        
        assert yaml_data == {}
        assert "Body content" in body


class TestAgentFile:
    """Test AgentFile class."""
    
    def test_parse_agent_file(self):
        """Test parsing an agent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "AGENT.md"
            agent_path.write_text("""---
name: my-agent
description: My test agent
namespace: local
model: openai/gpt-4o-mini
intents:
  - help with testing
  - run tests
skills:
  - cron
---

# My Agent

You are a helpful test agent.
""")
            agent = AgentFile(agent_path)
            
            assert agent.name == "my-agent"
            assert agent.metadata.description == "My test agent"
            assert agent.metadata.namespace == "local"
            assert agent.metadata.model == "openai/gpt-4o-mini"
            assert "help with testing" in agent.metadata.intents
            assert "cron" in agent.metadata.skills
            assert "# My Agent" in agent.instructions
    
    def test_named_agent(self):
        """Test parsing named agent (AGENT-*.md)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "AGENT-planner.md"
            agent_path.write_text("""---
name: planner
---

Planning agent.
""")
            agent = AgentFile(agent_path)
            
            assert agent.is_named == True
            assert agent.name == "planner"
    
    def test_default_values(self):
        """Test default values when fields are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "AGENT.md"
            agent_path.write_text("""---
name: simple
---

Simple agent.
""")
            agent = AgentFile(agent_path)
            
            assert agent.metadata.model == "openai/gpt-4o-mini"
            assert agent.metadata.namespace == "local"
            assert agent.metadata.visibility == "local"
            assert agent.metadata.skills == []
    
    def test_add_skill(self):
        """Test adding a skill to agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "AGENT.md"
            agent_path.write_text("""---
name: test
skills: []
---

Test agent.
""")
            agent = AgentFile(agent_path)
            agent.add_skill("cron")
            
            # Reload and check
            agent2 = AgentFile(agent_path)
            assert "cron" in agent2.metadata.skills


class TestFindAgentFiles:
    """Test agent file discovery."""
    
    def test_find_in_directory(self):
        """Test finding agents in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "AGENT.md").write_text("---\nname: a\n---\nA")
            (tmppath / "AGENT-b.md").write_text("---\nname: b\n---\nB")
            (tmppath / "AGENT-c.md").write_text("---\nname: c\n---\nC")
            (tmppath / "README.md").write_text("Not an agent")
            
            agents = find_agent_files(tmppath, recursive=False)
            
            assert len(agents) == 3
            names = [a.name for a in agents]
            assert "a" in names or "default" in names
            assert "b" in names
            assert "c" in names
    
    def test_find_recursive(self):
        """Test recursive agent discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "AGENT.md").write_text("---\nname: root\n---\nRoot")
            
            subdir = tmppath / "subdir"
            subdir.mkdir()
            (subdir / "AGENT.md").write_text("---\nname: sub\n---\nSub")
            
            agents = find_agent_files(tmppath, recursive=True)
            
            assert len(agents) == 2


class TestCreateAgentFile:
    """Test agent file creation."""
    
    def test_create_default(self):
        """Test creating default AGENT.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            agent = create_agent_file(tmppath)
            
            assert agent.path == tmppath / "AGENT.md"
            assert agent.path.exists()
    
    def test_create_named(self):
        """Test creating named agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            agent = create_agent_file(tmppath, name="writer")
            
            assert agent.path == tmppath / "AGENT-writer.md"
            assert agent.path.exists()
            assert agent.metadata.name == "writer"
    
    def test_create_with_metadata(self):
        """Test creating agent with custom metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            agent = create_agent_file(
                tmppath,
                metadata={
                    "name": "custom",
                    "model": "anthropic/claude-3-opus",
                    "intents": ["custom intent"],
                }
            )
            
            assert agent.metadata.name == "custom"
            assert agent.metadata.model == "anthropic/claude-3-opus"
            assert "custom intent" in agent.metadata.intents
