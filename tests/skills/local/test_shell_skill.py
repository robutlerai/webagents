"""
Tests for ShellSkill sandboxing
"""

import pytest
from webagents.agents.skills.local.shell.skill import ShellSkill


@pytest.mark.asyncio
async def test_shell_skill_allowed_command():
    """Test running allowed command"""
    skill = ShellSkill({
        "allowed_commands": ["echo", "pwd"]
    })
    
    result = await skill.run_command("echo Hello")
    assert "Hello" in result


@pytest.mark.asyncio
async def test_shell_skill_blocked_command():
    """Test running blocked command"""
    skill = ShellSkill({
        "blocked_commands": ["rm"]
    })
    
    result = await skill.run_command("rm -rf /")
    assert "Access denied" in result
    assert "Blocked command" in result


@pytest.mark.asyncio
async def test_shell_skill_not_in_whitelist():
    """Test running command not in whitelist"""
    skill = ShellSkill({
        "allowed_commands": ["echo"]
    })
    
    result = await skill.run_command("cat /etc/passwd")
    assert "Access denied" in result
    assert "not allowed" in result


@pytest.mark.asyncio
async def test_shell_skill_default_safe_commands():
    """Test default safe commands are allowed"""
    skill = ShellSkill({})  # Default config
    
    # These should be allowed by default
    result = await skill.run_command("echo test")
    assert "test" in result
    
    result = await skill.run_command("pwd")
    assert "Access denied" not in result


@pytest.mark.asyncio
async def test_shell_skill_timeout():
    """Test command timeout"""
    skill = ShellSkill({})
    
    # Test with very short timeout
    result = await skill.run_command("sleep 5", timeout=1)
    assert "timed out" in result.lower()
