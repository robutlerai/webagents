"""
Tests for LocalFileSource
"""

import pytest
from pathlib import Path
from webagents.server.plugins.local_file_source import LocalFileSource
from webagents.server.storage.json_store import JSONMetadataStore


@pytest.mark.asyncio
async def test_local_file_source_list_agents(tmp_path):
    """Test listing local agents"""
    # Create test AGENT.md file
    agent_file = tmp_path / "AGENT.md"
    agent_file.write_text("""---
name: test-agent
description: Test agent
---

You are a test agent.
""")
    
    # Create source
    metadata_store = JSONMetadataStore(data_dir=tmp_path / "data")
    source = LocalFileSource(
        watch_dirs=[tmp_path],
        metadata_store=metadata_store
    )
    
    # List agents
    agents = await source.list_agents()
    assert len(agents) >= 1
    assert any(a["name"] == "test-agent" for a in agents)


@pytest.mark.asyncio
async def test_local_file_source_search_agents(tmp_path):
    """Test searching local agents"""
    # Create test agents
    (tmp_path / "AGENT-customer-support.md").write_text("""---
name: customer-support
---
Support agent
""")
    
    (tmp_path / "AGENT-customer-onboarding.md").write_text("""---
name: customer-onboarding
---
Onboarding agent
""")
    
    (tmp_path / "AGENT-internal-test.md").write_text("""---
name: internal-test
---
Test agent
""")
    
    # Create source
    metadata_store = JSONMetadataStore(data_dir=tmp_path / "data")
    source = LocalFileSource(
        watch_dirs=[tmp_path],
        metadata_store=metadata_store
    )
    
    # Search for customer agents
    results = await source.search_agents("customer-*")
    assert len(results) == 2
    assert all("customer" in a["name"] for a in results)
    
    # Search for test agents
    results = await source.search_agents("*-test")
    assert len(results) == 1
    assert results[0]["name"] == "internal-test"


@pytest.mark.asyncio
async def test_local_file_source_get_agent(tmp_path):
    """Test getting a specific agent"""
    # Create test agent
    agent_file = tmp_path / "AGENT-myagent.md"
    agent_file.write_text("""---
name: myagent
---
Test agent instructions
""")
    
    # Create source
    metadata_store = JSONMetadataStore(data_dir=tmp_path / "data")
    source = LocalFileSource(
        watch_dirs=[tmp_path],
        metadata_store=metadata_store
    )
    
    # Get agent
    agent = await source.get_agent("myagent")
    assert agent is not None
    assert agent.name == "myagent"
    assert "Test agent instructions" in agent.instructions
