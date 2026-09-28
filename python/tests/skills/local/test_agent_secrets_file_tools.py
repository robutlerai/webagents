"""
The file tools guard the agent's secrets and its control files (2026-09-27,
the agent-secrets lane; S-312 and S-314). The names, sentences and cases are
the shared fixture's (`tests/fixtures/agent_secrets/file_tools.json`); the
TypeScript suite runs the same in
`tests/unit/skills/agent-secrets-file-tools.test.ts`. The hole is proved
closed the way the real-model pass exercised it: in process, `write_file`
asked to rewrite `AGENT.md` to `preset: unrestricted`.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional

import pytest

from webagents.access.caller import LOCAL_OWNER, CallerAuth
from webagents.agents.skills.local.filesystem.agent_secrets_guard import (
    CONTROL_HEADER,
    CONTROL_PREFIXES,
    CONTROL_QUESTION,
    SECRET_FOLDERS,
    SECRET_NAMES,
    SECRET_PREFIXES,
    control_declined,
    control_entries,
    control_patterns,
    control_refusal,
    is_control_path,
    is_secret_path,
    secret_refusal,
    unified_diff,
)
from webagents.agents.skills.local.filesystem.skill import FilesystemSkill
from webagents.server.context.context_vars import create_context, set_context

FIXTURE = json.loads((Path(__file__).resolve().parents[2] / "fixtures" / "agent_secrets" / "file_tools.json").read_text())

BEFORE = "---\nname: helper\nsandbox:\n  preset: development\n---\n\nYou help.\n"
AFTER = "---\nname: helper\nsandbox:\n  preset: unrestricted\n---\n\nYou help.\n"


def as_caller(auth: Optional[CallerAuth]) -> None:
    """The turn's caller, as the chat (`run_as_local_owner`) or the server sets it."""
    context = create_context(messages=[], stream=True, agent=None)
    context.auth = auth
    set_context(context)


def owner() -> None:
    as_caller(replace(LOCAL_OWNER, groups=[], principals=[]))


def friend() -> None:
    as_caller(CallerAuth(scope="user", groups=["friends"], provider="portal"))


def run(coro: Awaitable[Any], caller: Callable[[], None] = owner) -> Any:
    async def inner():
        caller()
        return await coro

    return asyncio.run(inner())


# -- the names and the sentences ---------------------------------------------------------------


def test_the_names_and_sentences_are_the_fixture_s():
    assert list(SECRET_NAMES) == FIXTURE["secrets"]["names"]
    assert list(SECRET_PREFIXES) == FIXTURE["secrets"]["prefixes"]
    assert list(SECRET_FOLDERS) == FIXTURE["secrets"]["folders"]
    assert secret_refusal("{path}") == FIXTURE["secrets"]["refusal"]
    assert control_entries() == FIXTURE["control"]["entries"]
    assert control_patterns() == FIXTURE["control"]["patterns"]
    assert list(CONTROL_PREFIXES) == FIXTURE["control"]["prefixes"]
    assert control_refusal("{path}") == FIXTURE["control"]["refusal"]
    assert CONTROL_HEADER == FIXTURE["control"]["header"]
    assert CONTROL_QUESTION == FIXTURE["control"]["question"]
    assert control_declined("{path}") == FIXTURE["control"]["declined"]


def test_list_directory_needs_no_argument_and_says_what_its_path_defaults_to():
    definition = FilesystemSkill.list_directory._webagents_tool_definition["function"]
    assert "path" not in definition["parameters"].get("required", [])
    # The decorator's generic description (the fixture's `note`); the sentence itself is the docstring's.
    assert definition["parameters"]["properties"]["path"]["description"] == FIXTURE["list_directory"]["python_schema_description"]
    assert FIXTURE["list_directory"]["path_description"] in (FilesystemSkill.list_directory.__doc__ or "")


# -- which paths are secrets, and which are control files ----------------------------------------


@pytest.mark.parametrize("case", FIXTURE["secrets"]["cases"], ids=[c["path"] for c in FIXTURE["secrets"]["cases"]])
def test_secret_cases(case, tmp_path):
    assert is_secret_path(tmp_path / case["path"]) is case["refused"]


@pytest.mark.parametrize(
    "case", FIXTURE["control"]["cases"], ids=[("~/" if c.get("home") else "") + c["path"] for c in FIXTURE["control"]["cases"]]
)
def test_control_cases(case, tmp_path):
    root = tmp_path / ("home" if case.get("home") else "project")
    assert is_control_path(root / case["path"]) is case["control"]


def test_a_symbolic_link_to_a_secret_is_a_secret(tmp_path):
    link = FIXTURE["secrets"]["symlink"]
    (tmp_path / link["target"]).write_text("KEY=1\n")
    os.symlink(tmp_path / link["target"], tmp_path / link["link"])
    assert is_secret_path(tmp_path / link["link"]) is link["refused"]


def test_the_running_agents_own_file_is_a_control_file_under_any_name(tmp_path):
    own = tmp_path / FIXTURE["control"]["named_agent_file"]
    assert is_control_path(own) is False
    assert is_control_path(own, own) is True
    assert is_control_path(tmp_path / "notes.md", own) is False


# -- the secrets set is refused for everyone (S-312) ---------------------------------------------


def folder_with_secrets(tmp_path: Path) -> FilesystemSkill:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-live-fixture\n")
    (tmp_path / ".webagents" / "keys").mkdir(parents=True)
    (tmp_path / ".webagents" / "keys" / "reporter.ed25519.jwk.json").write_text('{"d":"secret"}')
    (tmp_path / "README.md").write_text("sk-live-fixture is not here\nhello\n")
    return FilesystemSkill({"base_dir": str(tmp_path), "whitelist": [str(tmp_path)]})


def test_read_write_and_replace_answer_the_sentence_and_the_bytes_stay(tmp_path):
    skill = folder_with_secrets(tmp_path)
    assert run(skill.read_file(".env")) == secret_refusal(".env")
    key = ".webagents/keys/reporter.ed25519.jwk.json"
    assert run(skill.read_file(key)) == secret_refusal(key)
    assert run(skill.write_file(".env.local", "X=1")) == secret_refusal(".env.local")
    assert not (tmp_path / ".env.local").exists()
    assert run(skill.replace(".env", "sk-live", "gone")) == secret_refusal(".env")
    assert (tmp_path / ".env").read_text() == "OPENAI_API_KEY=sk-live-fixture\n"


def test_search_never_reads_them_and_a_link_to_one_is_refused_too(tmp_path):
    skill = folder_with_secrets(tmp_path)
    found = run(skill.search_file_content("sk-live"))
    assert "README.md" in found
    assert ".env" not in found
    assert "OPENAI_API_KEY" not in found
    os.symlink(tmp_path / ".env", tmp_path / "config.txt")
    assert run(skill.read_file("config.txt")) == secret_refusal("config.txt")


def test_a_listing_still_shows_the_names_with_no_argument_at_all(tmp_path):
    skill = folder_with_secrets(tmp_path)
    listing = run(skill.list_directory())
    assert ".env" in listing
    assert "[DIR] .webagents" in listing
    assert run(skill.list_directory(None)) == listing


# -- a control file is written only with the owner's yes in the chat (S-314) ---------------------


def folder_with_agent(tmp_path: Path, confirm: Optional[Callable[[str, str], Awaitable[bool]]] = None) -> FilesystemSkill:
    (tmp_path / "AGENT.md").write_text(BEFORE)
    config: dict = {"base_dir": str(tmp_path), "whitelist": [str(tmp_path)]}
    if confirm is not None:
        config["confirm_control_write"] = confirm
    return FilesystemSkill(config)


def test_with_no_chat_to_ask_the_write_is_refused_and_agent_md_keeps_its_preset(tmp_path):
    skill = folder_with_agent(tmp_path)
    assert run(skill.write_file("AGENT.md", AFTER)) == control_refusal("AGENT.md")
    assert run(skill.replace("AGENT.md", "development", "unrestricted")) == control_refusal("AGENT.md")
    assert (tmp_path / "AGENT.md").read_text() == BEFORE
    for planted in ("AGENT-evil.md", "WEBAGENTS.md", "mcp.json", ".mcp.json", ".agents/skills/x/SKILL.md", ".git/hooks/pre-commit", ".envrc"):
        assert run(skill.write_file(planted, "planted")) == control_refusal(planted)
        assert not (tmp_path / planted).exists()
    assert run(skill.replace("AGENT-evil.md", "", "planted")) == control_refusal("AGENT-evil.md")
    assert not (tmp_path / "AGENT-evil.md").exists()


def test_with_the_chat_asking_the_owner_sees_the_diff_and_a_yes_writes_it(tmp_path):
    asked: List[tuple] = []

    async def confirm(file: str, diff: str) -> bool:
        asked.append((file, diff))
        return True

    skill = folder_with_agent(tmp_path, confirm)
    assert "Successfully overwrote" in run(skill.write_file("AGENT.md", AFTER))
    assert (tmp_path / "AGENT.md").read_text() == AFTER
    assert len(asked) == 1
    assert asked[0][0] == "AGENT.md"
    assert "-  preset: development" in asked[0][1]
    assert "+  preset: unrestricted" in asked[0][1]
    # A new control file shows as all additions.
    assert "Successfully created" in run(skill.write_file("WEBAGENTS.md", "Shared context.\n"))
    assert "+Shared context." in asked[1][1]


def test_a_no_leaves_the_file_as_it_was_and_says_the_owner_declined(tmp_path):
    async def confirm(file: str, diff: str) -> bool:
        return False

    skill = folder_with_agent(tmp_path, confirm)
    assert run(skill.write_file("AGENT.md", AFTER)) == control_declined("AGENT.md")
    assert run(skill.replace("AGENT.md", "development", "unrestricted")) == control_declined("AGENT.md")
    assert (tmp_path / "AGENT.md").read_text() == BEFORE


def test_a_caller_who_is_not_the_owner_is_refused_even_when_the_chat_could_ask(tmp_path):
    asked = 0

    async def confirm(file: str, diff: str) -> bool:
        nonlocal asked
        asked += 1
        return True

    skill = folder_with_agent(tmp_path, confirm)
    assert run(skill.write_file("AGENT.md", AFTER), caller=friend) == control_refusal("AGENT.md")
    assert run(skill.write_file("AGENT.md", AFTER), caller=lambda: as_caller(None)) == control_refusal("AGENT.md")
    assert asked == 0
    assert (tmp_path / "AGENT.md").read_text() == BEFORE


def test_an_ordinary_file_never_asks(tmp_path):
    asked = 0

    async def confirm(file: str, diff: str) -> bool:
        nonlocal asked
        asked += 1
        return False

    skill = folder_with_agent(tmp_path, confirm)
    assert "Successfully created" in run(skill.write_file("notes.md", "plain"))
    assert (tmp_path / "notes.md").read_text() == "plain"
    assert asked == 0


def test_the_loaders_pass_the_agent_file_and_only_the_chat_passes_the_confirm(tmp_path):
    from webagents.cli.agent_builder import load_skills

    (tmp_path / "AGENT.md").write_text(BEFORE)

    async def confirm(file: str, diff: str) -> bool:
        return True

    plain = load_skills(["filesystem"], agent_name="helper", agent_path=tmp_path / "AGENT.md")["filesystem"]
    assert plain.agent_file == (tmp_path / "AGENT.md").resolve()
    assert plain.confirm_control_write is None
    chat = load_skills(["filesystem"], agent_name="helper", agent_path=tmp_path / "AGENT.md", confirm_control_write=confirm)["filesystem"]
    assert chat.confirm_control_write is confirm


def test_the_diff_is_a_unified_diff_and_a_huge_file_is_summarized():
    diff = unified_diff("a\nb\nc\n", "a\nB\nc\n", "x.md")
    assert diff.split("\n")[:2] == ["--- x.md", "+++ x.md"]
    assert "-b" in diff
    assert "+B" in diff
    big = "\n".join(f"line {i}" for i in range(5000))
    assert "too large to show" in unified_diff(big, big + "\nmore", "big.md")
