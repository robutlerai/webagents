"""
The host-reaching tools are owner-only unless the agent file opens them
(S-248, 2026-09-26).

`ShellSkill.run_command` and the six `FilesystemSkill` tools took the
decorator's default scope, `all`, so every caller an agent answered could run
commands and read and write files as the developer's user. They declare
`scope="owner"` now, and an `access: tools:` block replaces that scope with
the named group's. The TypeScript suite pins the same in
`tests/unit/skills/host-tools-owner-only-s248.test.ts`.
"""

from webagents.access.install import apply_access_tools
from webagents.access.policy import parse_access
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.core.scopes import scope_allows
from webagents.agents.skills.local.filesystem.skill import FilesystemSkill
from webagents.agents.skills.local.shell.skill import ShellSkill

FILESYSTEM_TOOLS = ["list_directory", "read_file", "write_file", "glob", "search_file_content", "replace"]


def _agent(tmp_path):
    return BaseAgent(
        name="host",
        instructions="x",
        model="openai/gpt-4o-mini",
        skills={
            "shell": ShellSkill({"base_dir": str(tmp_path)}),
            "filesystem": FilesystemSkill({"base_dir": str(tmp_path)}),
        },
    )


def test_the_tools_declare_the_owner_scope():
    assert ShellSkill.run_command._tool_scope == "owner"
    for name in FILESYSTEM_TOOLS:
        assert getattr(FilesystemSkill, name)._tool_scope == "owner", name


def test_the_agent_offers_them_to_the_owner_and_an_admin_only(tmp_path):
    agent = _agent(tmp_path)
    host_tools = {"run_command", *FILESYSTEM_TOOLS}

    def offered(scope):
        return {t.get("name") or t.get("function", {}).get("name", "") for t in agent.get_tools_for_scope(scope)}

    assert host_tools <= offered("owner")
    assert host_tools <= offered("admin")
    assert not (host_tools & offered("all"))
    assert not (host_tools & offered("user"))


def test_the_registered_scope_is_owner(tmp_path):
    agent = _agent(tmp_path)
    scopes = {t["name"]: t.get("scope") for t in agent.get_all_tools()}
    for name in ["run_command", *FILESYSTEM_TOOLS]:
        assert scopes[name] == "owner", name
        assert scope_allows(scopes[name], {"owner"})
        assert scope_allows(scopes[name], {"admin"})
        assert not scope_allows(scopes[name], {"user"})
        assert not scope_allows(scopes[name], set())


def test_an_access_block_hands_them_to_a_group_and_the_owner_keeps_them(tmp_path):
    agent = _agent(tmp_path)
    policy = parse_access({"groups": {"friends": []}, "tools": {"friends": ["shell", "filesystem"]}})
    apply_access_tools(agent, policy, ["shell", "filesystem"], strict=True)
    scopes = {t["name"]: t.get("scope") for t in agent.get_all_tools()}
    for name in ["run_command", *FILESYSTEM_TOOLS]:
        assert scopes[name] == ["group:friends"], name
        assert scope_allows(scopes[name], {"group:friends"})
        assert scope_allows(scopes[name], {"owner"})
        assert not scope_allows(scopes[name], {"user"})
        assert not scope_allows(scopes[name], set())
