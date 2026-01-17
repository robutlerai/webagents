"""
Tests for SessionManagerSkill
"""

import pytest
from pathlib import Path
from webagents.agents.skills.local.session.skill import SessionManagerSkill


@pytest.mark.asyncio
async def test_session_manager_save_checkpoint(tmp_path):
    """Test saving checkpoint"""
    checkpoint_dir = tmp_path / "checkpoints"
    
    skill = SessionManagerSkill({
        "agent_name": "test-agent"
    })
    # Override checkpoint dir for test
    skill.checkpoint_dir = checkpoint_dir
    skill.agent_checkpoint_dir = checkpoint_dir / "test-agent"
    skill.agent_checkpoint_dir.mkdir(parents=True)
    
    # Save checkpoint
    result = await skill.save_checkpoint("test", {"history": ["msg1", "msg2"]})
    assert "saved" in result.lower()
    
    # Check file exists
    checkpoint_file = skill.agent_checkpoint_dir / "test.json"
    assert checkpoint_file.exists()


@pytest.mark.asyncio
async def test_session_manager_load_checkpoint(tmp_path):
    """Test loading checkpoint"""
    checkpoint_dir = tmp_path / "checkpoints"
    
    skill = SessionManagerSkill({
        "agent_name": "test-agent"
    })
    skill.checkpoint_dir = checkpoint_dir
    skill.agent_checkpoint_dir = checkpoint_dir / "test-agent"
    skill.agent_checkpoint_dir.mkdir(parents=True)
    
    # Save then load
    await skill.save_checkpoint("test", {"key": "value"})
    result = await skill.load_checkpoint("test")
    
    # Result should be JSON string
    import json
    data = json.loads(result)
    assert data["data"]["key"] == "value"


@pytest.mark.asyncio
async def test_session_manager_list_checkpoints(tmp_path):
    """Test listing checkpoints"""
    checkpoint_dir = tmp_path / "checkpoints"
    
    skill = SessionManagerSkill({
        "agent_name": "test-agent"
    })
    skill.checkpoint_dir = checkpoint_dir
    skill.agent_checkpoint_dir = checkpoint_dir / "test-agent"
    skill.agent_checkpoint_dir.mkdir(parents=True)
    
    # Save multiple checkpoints
    await skill.save_checkpoint("cp1", {})
    await skill.save_checkpoint("cp2", {})
    
    # List checkpoints
    result = await skill.list_checkpoints()
    assert "cp1" in result
    assert "cp2" in result


@pytest.mark.asyncio
async def test_session_manager_delete_checkpoint(tmp_path):
    """Test deleting checkpoint"""
    checkpoint_dir = tmp_path / "checkpoints"
    
    skill = SessionManagerSkill({
        "agent_name": "test-agent"
    })
    skill.checkpoint_dir = checkpoint_dir
    skill.agent_checkpoint_dir = checkpoint_dir / "test-agent"
    skill.agent_checkpoint_dir.mkdir(parents=True)
    
    # Save and delete
    await skill.save_checkpoint("test", {})
    checkpoint_file = skill.agent_checkpoint_dir / "test.json"
    assert checkpoint_file.exists()
    
    result = await skill.delete_checkpoint("test")
    assert "deleted" in result.lower()
    assert not checkpoint_file.exists()
