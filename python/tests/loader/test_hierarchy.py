"""
Context Hierarchy Tests

Test AGENTS.md context inheritance.
"""

import pytest
from pathlib import Path
import tempfile

from webagents.cli.loader.context import (
    ContextFile,
    ContextHierarchy,
    find_context_file,
    create_context_file,
)
from webagents.cli.loader.hierarchy import (
    AgentLoader,
    MergedAgent,
    find_default_agent,
    load_agent,
)


class TestContextFile:
    """Test AGENTS.md parsing."""
    
    def test_parse_context(self):
        """Test parsing context file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx_path = Path(tmpdir) / "AGENTS.md"
            ctx_path.write_text("""---
namespace: ai.myorg
model: openai/gpt-4o
skills:
  - mcp
  - memory
---

# Project Context

All agents in this project should be helpful.
""")
            ctx = ContextFile(ctx_path)
            
            assert ctx.metadata.namespace == "ai.myorg"
            assert ctx.metadata.model == "openai/gpt-4o"
            assert "mcp" in ctx.metadata.skills
            assert "memory" in ctx.metadata.skills
            assert "# Project Context" in ctx.content


class TestContextHierarchy:
    """Test context hierarchy resolution."""
    
    def test_resolve_single(self):
        """Test resolving single context file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "AGENTS.md").write_text("---\nnamespace: test\n---\nContext")
            
            hierarchy = ContextHierarchy()
            contexts = hierarchy.resolve(tmppath)
            
            assert len(contexts) == 1
            assert contexts[0].metadata.namespace == "test"
    
    def test_resolve_nested(self):
        """Test resolving nested context hierarchy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "AGENTS.md").write_text("""---
namespace: root
model: openai/gpt-4o
---

Root context.
""")
            subdir = tmppath / "subdir"
            subdir.mkdir()
            (subdir / "AGENTS.md").write_text("""---
namespace: sub
skills:
  - cron
---

Sub context.
""")
            hierarchy = ContextHierarchy()
            contexts = hierarchy.resolve(subdir)
            
            # Should have both (root first, sub last)
            assert len(contexts) == 2
            assert contexts[0].metadata.namespace == "root"
            assert contexts[1].metadata.namespace == "sub"
    
    def test_merge_contexts(self):
        """Test merging multiple contexts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "AGENTS.md").write_text("""---
namespace: root
model: openai/gpt-4o
skills:
  - mcp
---
Root
""")
            subdir = tmppath / "sub"
            subdir.mkdir()
            (subdir / "AGENTS.md").write_text("""---
namespace: sub
skills:
  - cron
---
Sub
""")
            hierarchy = ContextHierarchy()
            contexts = hierarchy.resolve(subdir)
            merged = hierarchy.merge_contexts(contexts)
            
            # Namespace should be overridden by sub
            assert merged["namespace"] == "sub"
            # Skills should be accumulated
            assert "mcp" in merged["skills"]
            assert "cron" in merged["skills"]


class TestAgentLoader:
    """Test agent loading with context."""
    
    def test_load_with_context(self):
        """Test loading agent with context inheritance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create context
            (tmppath / "AGENTS.md").write_text("""---
namespace: ai.test
skills:
  - memory
---

Project guidelines.
""")
            # Create agent
            (tmppath / "AGENT.md").write_text("""---
name: agent1
skills:
  - cron
---

# Agent 1

Specific instructions.
""")
            loader = AgentLoader()
            merged = loader.load(tmppath / "AGENT.md")
            
            assert merged.name == "agent1"
            assert merged.metadata.namespace == "ai.test"
            # Skills should include both context and agent
            assert "memory" in merged.metadata.skills
            assert "cron" in merged.metadata.skills
            # Instructions should include context and agent
            assert "Project guidelines" in merged.instructions or "Specific instructions" in merged.instructions
    
    def test_load_all(self):
        """Test loading all agents in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "AGENT.md").write_text("---\nname: a\n---\nA")
            (tmppath / "AGENT-b.md").write_text("---\nname: b\n---\nB")
            
            loader = AgentLoader()
            agents = loader.load_all(tmppath)
            
            assert len(agents) == 2


class TestFindDefaultAgent:
    """Test default agent resolution."""
    
    def test_find_agent_md(self):
        """Test finding AGENT.md as default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "AGENT.md").write_text("---\nname: default\n---\nDefault")
            (tmppath / "AGENT-other.md").write_text("---\nname: other\n---\nOther")
            
            agent = find_default_agent(tmppath)
            
            assert agent is not None
            assert agent.name == "default"
    
    def test_find_single_named(self):
        """Test finding single named agent as default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "AGENT-only.md").write_text("---\nname: only\n---\nOnly")
            
            agent = find_default_agent(tmppath)
            
            assert agent is not None
            assert agent.name == "only"
    
    def test_no_default_multiple(self):
        """Test no default when multiple named agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "AGENT-a.md").write_text("---\nname: a\n---\nA")
            (tmppath / "AGENT-b.md").write_text("---\nname: b\n---\nB")
            
            agent = find_default_agent(tmppath)
            
            assert agent is None
    
    def test_no_agents(self):
        """Test no agents found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = find_default_agent(Path(tmpdir))
            assert agent is None
