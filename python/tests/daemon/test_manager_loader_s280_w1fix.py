"""
S-280 (2026-09-26): the legacy Python daemon's `AgentManager.get_or_load_agent`
gave an agent file that listed no skills eight defaults, `filesystem` and
`shell` among them, with no `sandbox:`, and added a `GoogleAISkill` (a
tool-bearing model skill) and the model `google/gemini-2.5-flash` the file
never named. So the model could reach a shell and the filesystem unconfined,
and the owner's own turns (scheduled ones included) could run commands the
file never asked for. (The loader also raised on a dead
`webagents.server.plugins.local_file_source` import, so it had never actually
loaded an agent, which is why the entry was verified by reading; that import
is removed here so the loader works.)

Fixed to load exactly what the file declares, plus a transport, as the
file-source loader does: no default shell/filesystem, and `choose_model`
(which is not Google by default, and fails closed with no key) instead of the
forced `GoogleAISkill`.
"""

from __future__ import annotations

import asyncio
import os

from webagents.cli.daemon.manager import AgentManager
from webagents.cli.daemon.registry import DaemonRegistry
from webagents.cli.loader import AgentFile

NO_SKILLS = """---
name: bare
description: A bare agent that declares no skills.
---

# Bare
"""

ONE_SKILL = """---
name: sessioned
description: An agent that declares one local skill.
skills:
  - session
---

# Sessioned
"""

#: The skills the old default-8 fallback added and the tool-bearing model skill
#: the old GoogleAISkill auto-add added; none of these may appear unasked.
FORBIDDEN_DEFAULTS = ("filesystem", "shell", "web", "rag", "todo", "mcp", "google", "llm")


def _manager(tmp_path, content: str) -> AgentManager:
    (tmp_path / "AGENT.md").write_text(content)
    registry = DaemonRegistry()
    registry.register(AgentFile(tmp_path / "AGENT.md"))
    return AgentManager(registry)


def test_load_skills_loads_exactly_what_is_passed_no_defaults(tmp_path):
    # `_load_skills` builds only the skills it is given (and no longer raises
    # on the dead LocalFileSource import).
    manager = _manager(tmp_path, ONE_SKILL)
    skills = manager._load_skills(["session", "completions"], "sessioned", tmp_path / "AGENT.md")
    assert set(skills.keys()) == {"session", "completions"}
    for forbidden in FORBIDDEN_DEFAULTS:
        assert forbidden not in skills, f"{forbidden} was built though it was not requested"


def test_get_or_load_agent_computes_no_default_eight_for_a_bare_file(tmp_path):
    # The default-eight fallback is gone: a file with no skills reaches
    # `_load_skills` with only the transport, never filesystem/shell/etc.
    manager = _manager(tmp_path, NO_SKILLS)
    captured = {}

    def spy(skills_config, agent_name, agent_path):
        captured["skills"] = list(skills_config)
        return {}

    manager._load_skills = spy  # type: ignore[method-assign]
    asyncio.run(manager.get_or_load_agent("bare"))
    assert captured.get("skills") == ["completions"]
    for forbidden in FORBIDDEN_DEFAULTS:
        assert forbidden not in captured["skills"], f"{forbidden} was injected for a bare file"


def test_get_or_load_agent_adds_no_google_model_and_fails_closed_without_a_key(tmp_path, monkeypatch):
    # No GoogleAISkill is auto-added, and no google/gemini default is forced:
    # a bare agent with no model skill and no provider key does not silently
    # load with Google; `choose_model` fails closed, so the loader returns None
    # rather than a shell-capable agent the file never described.
    # Hermetic: another test can leave a provider key in os.environ (the Python
    # chat's `/keys set` writes there), and with a key `choose_model` rightly
    # picks a model, so the key-less case clears them itself (full-suite order
    # failure, 2026-09-26).
    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_TOKEN")) or name.startswith(("WEBAGENTS_", "ROBUTLER_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    manager = _manager(tmp_path, NO_SKILLS)
    seen = {}

    def spy(skills_config, agent_name, agent_path):
        # A real transport skill so the only thing missing is a model.
        from webagents.agents.skills.core.transport.completions.skill import CompletionsTransportSkill

        skills = {"completions": CompletionsTransportSkill(config={"agent_name": agent_name})}
        seen["skills"] = set(skills.keys())
        return skills

    manager._load_skills = spy  # type: ignore[method-assign]
    agent = asyncio.run(manager.get_or_load_agent("bare"))
    assert agent is None, "a keyless bare agent must fail closed, not load with a forced Google model"
    assert "google" not in seen.get("skills", set()) and "llm" not in seen.get("skills", set())
