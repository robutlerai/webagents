"""
Tests for ShellSkill sandboxing

THE COMMANDS HERE RUN CONFINED (2026-09-27): the sandbox is on by default,
so a bare `ShellSkill({})` runs its commands under srt, the engine that ships
inside this package (the sandbox-engine lane, 2026-09-27; `WEBAGENTS_SRT_CLI`
still wins when set). In a fresh venv they run with nothing else installed:
node comes from PATH or the nodejs-wheel-binaries dependency. Where srt cannot
run, every command here fails closed and the tests skip with the reason rather
than pass on nothing.
"""

import pytest

from webagents.agents.skills.local.shell.skill import ShellSkill
from webagents.sandbox import backend_status, sandbox_available

pytestmark = pytest.mark.skipif(not sandbox_available(), reason=f"no srt here: {backend_status()['reason']}")


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
    """A name the agent file blocks is refused even when confined."""
    skill = ShellSkill({
        "blocked_commands": ["rm"]
    })

    result = await skill.run_command("rm -rf /")
    # One wording in both SDKs (fixture `sandbox/srt.json` `shell_allowlist`).
    assert result == "Access denied: Command 'rm' is blocked"


@pytest.mark.asyncio
async def test_shell_skill_not_in_whitelist():
    """The list gates UNCONFINED commands only (the ptypass-fixes lane,
    2026-09-27): confined, `whoami` runs and the kernel decides; with
    `sandbox: off` it is refused for its name."""
    skill = ShellSkill({
        "allowed_commands": ["echo"],
        "blocked_commands": [],
        "sandbox": "off",
    })

    result = await skill.run_command("whoami")
    assert result == "Access denied: Command 'whoami' is not in the allowlist"


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
    # Confined, `sleep` needs no listing (the ptypass-fixes lane, 2026-09-27).
    skill = ShellSkill({})
    
    # Test with very short timeout
    result = await skill.run_command("sleep 5", timeout=1)
    assert "timed out" in result.lower()
