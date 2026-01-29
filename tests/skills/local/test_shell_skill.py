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
    # Error message format: "Blocked command" or "Command not allowed"
    assert "Blocked" in result or "not allowed" in result


@pytest.mark.asyncio
async def test_shell_skill_not_in_whitelist():
    """Test running command not in whitelist"""
    skill = ShellSkill({
        "allowed_commands": ["echo"],
        "blocked_commands": []  # Clear default blocked to test whitelist only
    })
    
    # Remove cat from defaults by providing explicit whitelist-only mode
    result = await skill.run_command("whoami")
    # whoami is not in the allowed list (only echo is)
    assert "Access denied" in result or "not allowed" in result.lower()


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
    # sleep is not in default allowed commands, so add it
    skill = ShellSkill({
        "allowed_commands": ["sleep"]
    })
    
    # Test with very short timeout
    result = await skill.run_command("sleep 5", timeout=1)
    assert "timed out" in result.lower()
