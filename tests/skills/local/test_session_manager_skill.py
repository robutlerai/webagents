"""
Tests for SessionManagerSkill

Note: SessionManagerSkill uses @command decorators for session management,
not checkpoint methods. For checkpoint tests, see test_checkpoint_skill.py.
"""

import pytest
import json
from pathlib import Path
from webagents.agents.skills.local.session.skill import SessionManagerSkill, Session
from datetime import datetime


@pytest.mark.asyncio
async def test_session_manager_save_session(tmp_path):
    """Test saving session via command"""
    skill = SessionManagerSkill({
        "agent_name": "test-agent",
        "agent_path": str(tmp_path)
    })
    
    # Add some messages to the current session
    skill.add_message("user", "Hello")
    skill.add_message("assistant", "Hi there!")
    
    # Save session
    result = await skill.save_session(name="test-session")
    assert result["status"] == "saved"
    assert result["message_count"] == 2


@pytest.mark.asyncio
async def test_session_manager_load_session(tmp_path):
    """Test loading session via command"""
    skill = SessionManagerSkill({
        "agent_name": "test-agent",
        "agent_path": str(tmp_path)
    })
    
    # Save then load
    skill.add_message("user", "Test message")
    await skill.save_session()
    
    # Get the session ID
    session_id = skill._current_session.session_id
    
    # Clear and reload
    skill._current_session = None
    result = await skill.load_session(session_id)
    
    assert result["status"] == "loaded"
    assert result["message_count"] == 1


@pytest.mark.asyncio
async def test_session_manager_list_sessions(tmp_path):
    """Test listing sessions"""
    skill = SessionManagerSkill({
        "agent_name": "test-agent",
        "agent_path": str(tmp_path)
    })
    
    # Create multiple sessions
    await skill.new_session()
    skill.add_message("user", "Session 1")
    await skill.save_session(name="session1")
    
    await skill.new_session()
    skill.add_message("user", "Session 2")
    await skill.save_session(name="session2")
    
    # List sessions
    result = await skill.list_sessions()
    assert result["total"] >= 2


@pytest.mark.asyncio
async def test_session_manager_new_session(tmp_path):
    """Test creating new session"""
    skill = SessionManagerSkill({
        "agent_name": "test-agent",
        "agent_path": str(tmp_path)
    })
    
    # Create new session
    result = await skill.new_session()
    assert result["status"] == "created"
    assert "session_id" in result
