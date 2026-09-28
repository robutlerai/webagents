"""
The SKILL.md skill an agent carries (gap-closure plan item 1.4, 2026-09-26),
against the shared fixture `tests/fixtures/skillmd/skillmd.json`, which the
TypeScript suite reads too (`tests/unit/skills/skillmd-skill.test.ts`): the
three tools and their words, the catalog in the prompt for callers who may
activate, activation once per conversation, reading confined to the skill's
folder, and the owner-only default that `access: tools:` opens. Scripts run
under real srt in `tests/sandbox/test_skillmd_scripts.py`.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from webagents.access.caller import LOCAL_OWNER, CallerAuth
from webagents.access.install import apply_access_tools
from webagents.access.policy import parse_access
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.core.scopes import scope_allows
from webagents.agents.skills.local.skillmd import SKILL_KEY, SkillMdSkill, discover_skills
from webagents.agents.skills.local.skillmd.skillmd_skill import DEFAULT_TIMEOUT, INTERPRETERS, MAX_TIMEOUT, READ_MAX_BYTES, TOOL_NAMES
from webagents.server.context.context_vars import CONTEXT, create_context, set_context

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "skillmd"
FIXTURE = json.loads((FIXTURES / "skillmd.json").read_text())
REPO = FIXTURES / FIXTURE["sample"]["repo"]


@pytest.fixture(autouse=True)
def _no_caller():
    token = CONTEXT.set(None)
    yield
    CONTEXT.reset(token)


def _as(auth, messages=None, agent=None):
    context = create_context(messages=list(messages or []), agent=agent)
    context.auth = auth
    set_context(context)
    return context


def _skill(agent_dir: Path = REPO, sandbox=None) -> SkillMdSkill:
    found = discover_skills(str(agent_dir), FIXTURE["sample"]["explicit"])
    return SkillMdSkill({"skills": found.skills, "skipped": found.skipped, "warnings": found.warnings, "agent_dir": str(agent_dir), "sandbox": sandbox})


def _agent(skill: SkillMdSkill) -> BaseAgent:
    return BaseAgent(name="host", instructions="x", model="openai/gpt-4o-mini", skills={SKILL_KEY: skill})


class TestTheToolsMatchTheFixture:
    def test_the_names_scope_and_limits(self):
        assert list(TOOL_NAMES) == FIXTURE["tools"]["names"]
        assert INTERPRETERS == FIXTURE["scripts"]["interpreters"]
        assert DEFAULT_TIMEOUT == FIXTURE["scripts"]["default_timeout"]
        assert MAX_TIMEOUT == FIXTURE["scripts"]["max_timeout"]
        assert READ_MAX_BYTES == FIXTURE["read"]["max_bytes"]
        assert SKILL_KEY == FIXTURE["layout"]["skill_key"]

    def test_the_definitions_word_for_word(self):
        agent = _agent(_skill())
        by_name = {t["name"]: t for t in agent.get_all_tools()}
        for name in FIXTURE["tools"]["names"]:
            expected = FIXTURE["tools"][name]
            assert by_name[name]["definition"]["function"]["description"] == expected["description"], name
            assert by_name[name]["definition"]["function"]["parameters"] == expected["parameters"], name
            assert by_name[name]["scope"] == FIXTURE["tools"]["scope"], name
            assert by_name[name]["source"] == SKILL_KEY

    def test_owner_and_admin_see_them_and_nobody_else(self):
        agent = _agent(_skill())
        names = set(FIXTURE["tools"]["names"])

        def offered(scope):
            return {t["name"] for t in agent.get_tools_for_scope(scope)}

        assert names <= offered("owner")
        assert names <= offered("admin")
        assert not (names & offered("user"))
        assert not (names & offered("all"))

    def test_an_access_block_opens_the_skill_by_its_name_or_a_tool_by_its_own(self):
        agent = _agent(_skill())
        policy = parse_access({"groups": {"friends": []}, "tools": {"friends": [SKILL_KEY]}})
        apply_access_tools(agent, policy, [SKILL_KEY], strict=True)
        scopes = {t["name"]: t["scope"] for t in agent.get_all_tools()}
        for name in FIXTURE["tools"]["names"]:
            assert scopes[name] == ["group:friends"], name
            assert scope_allows(scopes[name], {"group:friends"})
            assert scope_allows(scopes[name], {"owner"})
            assert not scope_allows(scopes[name], {"user"})

        agent = _agent(_skill())
        policy = parse_access({"groups": {"readers": []}, "tools": {"readers": ["activate_skill", "read_skill_file"]}})
        apply_access_tools(agent, policy, [SKILL_KEY], strict=True)
        scopes = {t["name"]: t["scope"] for t in agent.get_all_tools()}
        assert scopes["activate_skill"] == ["group:readers"]
        assert scopes["read_skill_file"] == ["group:readers"]
        assert scopes["run_skill_script"] == "owner"


class TestTheCatalog:
    def test_it_is_in_the_prompt_for_the_owner(self):
        skill = _skill()
        agent = _agent(skill)
        by_name = {s.name: s for s in skill.skills.values()}
        expected = FIXTURE["sample"]["catalog"].replace("{pdf_location}", by_name["pdf"].location).replace("{xlsx_location}", by_name["xlsx"].location)
        context = _as(LOCAL_OWNER)
        prompts = agent.get_prompts_for_scopes(context.auth_scopes)
        texts = [p["function"]() for p in prompts if p.get("source") == SKILL_KEY]
        assert texts == [expected]

    def test_a_caller_who_cannot_activate_does_not_see_it(self):
        skill = _skill()
        _agent(skill)
        _as(CallerAuth(scope="user"))
        assert skill.skills_catalog() == ""
        _as(LOCAL_OWNER)
        assert skill.skills_catalog().startswith(FIXTURE["catalog"]["preamble"])

    def test_a_group_the_block_opens_sees_it(self):
        skill = _skill()
        agent = _agent(skill)
        policy = parse_access({"groups": {"friends": []}, "tools": {"friends": [SKILL_KEY]}})
        apply_access_tools(agent, policy, [SKILL_KEY], strict=True)
        # Before the first turn starts the skills, the run's context carries the agent.
        _as(CallerAuth(scope="user", groups=["friends"]), agent=agent)
        assert skill.skills_catalog().startswith(FIXTURE["catalog"]["preamble"])
        _as(CallerAuth(scope="user", groups=["others"]), agent=agent)
        assert skill.skills_catalog() == ""
        # And once started, the skill knows its agent itself.
        asyncio.run(agent._ensure_skills_initialized())
        _as(CallerAuth(scope="user", groups=["friends"]))
        assert skill.skills_catalog().startswith(FIXTURE["catalog"]["preamble"])

    def test_no_skills_no_section(self, tmp_path):
        skill = SkillMdSkill({"skills": [], "agent_dir": str(tmp_path)})
        _as(LOCAL_OWNER)
        assert skill.skills_catalog() == FIXTURE["catalog"]["empty"]


class TestActivation:
    def test_returns_the_body_and_lists_the_files(self):
        skill = _skill()
        pdf = skill.skills["pdf"]
        expected = FIXTURE["sample"]["activation"].replace("{pdf_dir}", pdf.directory)
        _as(LOCAL_OWNER)
        assert asyncio.run(skill.activate_skill(name="pdf")) == expected

    def test_never_injects_twice_in_one_conversation(self):
        skill = _skill()
        first = asyncio.run(skill.activate_skill(name="pdf"))
        _as(LOCAL_OWNER, messages=[{"role": "user", "content": "hi"}, {"role": "tool", "tool_call_id": "c1", "content": first}])
        expected = FIXTURE["activation"]["already_active"].replace("{name}", "pdf")
        assert asyncio.run(skill.activate_skill(name="pdf")) == expected
        # Another skill is not affected, and a new conversation starts over.
        assert asyncio.run(skill.activate_skill(name="xlsx")).startswith(FIXTURE["activation"]["marker"].replace("{name}", "xlsx"))
        _as(LOCAL_OWNER, messages=[{"role": "user", "content": "new"}])
        assert asyncio.run(skill.activate_skill(name="pdf")) == first

    def test_a_tool_result_with_text_parts_counts(self):
        skill = _skill()
        first = asyncio.run(skill.activate_skill(name="pdf"))
        _as(LOCAL_OWNER, messages=[{"role": "tool", "content": [{"type": "text", "text": first}]}])
        assert asyncio.run(skill.activate_skill(name="pdf")) == FIXTURE["activation"]["already_active"].replace("{name}", "pdf")

    def test_an_unknown_name(self):
        skill = _skill()
        expected = FIXTURE["activation"]["unknown"].replace("{name}", "nope").replace("{available}", "pdf, xlsx")
        assert asyncio.run(skill.activate_skill(name="nope")) == expected
        empty = SkillMdSkill({"skills": [], "agent_dir": str(REPO)})
        assert asyncio.run(empty.activate_skill(name="nope")) == FIXTURE["activation"]["unknown"].replace("{name}", "nope").replace("{available}", FIXTURE["activation"]["unknown_none"])


class TestReadingFiles:
    def test_reads_a_bundled_file(self):
        skill = _skill()
        text = asyncio.run(skill.read_skill_file(skill="pdf", path="forms.md"))
        assert text == (REPO / "skills" / "pdf" / "forms.md").read_text()
        assert asyncio.run(skill.read_skill_file(skill="pdf", path="scripts/fill_form.py")).startswith('"""Fixture script')

    def test_is_confined_to_the_skill_folder(self, tmp_path):
        skill = _skill()
        refusal = FIXTURE["read"]["refusals"]["outside"]
        for path in ("../xlsx/SKILL.md", "/etc/hosts", "scripts", "missing.md", "../../README.md"):
            assert asyncio.run(skill.read_skill_file(skill="pdf", path=path)) == refusal.replace("{path}", path), path

    def test_a_symbolic_link_out_of_the_folder_is_refused(self, tmp_path):
        directory = tmp_path / ".agents" / "skills" / "linky"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text("---\ndescription: d\n---\n")
        (tmp_path / "secret.txt").write_text("TOP-SECRET")
        os.symlink(tmp_path / "secret.txt", directory / "escape.txt")
        found = discover_skills(str(tmp_path))
        skill = SkillMdSkill({"skills": found.skills, "agent_dir": str(tmp_path)})
        assert asyncio.run(skill.read_skill_file(skill="linky", path="escape.txt")) == FIXTURE["read"]["refusals"]["outside"].replace("{path}", "escape.txt")

    def test_binary_and_large_files_are_refused(self, tmp_path):
        directory = tmp_path / ".agents" / "skills" / "big"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text("---\ndescription: d\n---\n")
        (directory / "blob.bin").write_bytes(b"\x00\x01\x02")
        (directory / "huge.txt").write_bytes(b"x" * (READ_MAX_BYTES + 1))
        found = discover_skills(str(tmp_path))
        skill = SkillMdSkill({"skills": found.skills, "agent_dir": str(tmp_path)})
        assert asyncio.run(skill.read_skill_file(skill="big", path="blob.bin")) == FIXTURE["read"]["refusals"]["binary"].replace("{path}", "blob.bin")
        assert asyncio.run(skill.read_skill_file(skill="big", path="huge.txt")) == FIXTURE["read"]["refusals"]["large"].replace("{path}", "huge.txt")

    def test_an_unknown_skill(self):
        skill = _skill()
        assert asyncio.run(skill.read_skill_file(skill="nope", path="x")) == FIXTURE["activation"]["unknown"].replace("{name}", "nope").replace("{available}", "pdf, xlsx")


class TestScriptRefusalsWithoutSrt:
    """Everything decided before the sandbox is asked."""

    def test_outside_and_not_runnable(self):
        skill = _skill()
        refusals = FIXTURE["scripts"]["refusals"]
        assert asyncio.run(skill.run_skill_script(skill="pdf", script="../xlsx/SKILL.md")) == refusals["outside"].replace("{script}", "../xlsx/SKILL.md")
        assert asyncio.run(skill.run_skill_script(skill="pdf", script="scripts/none.py")) == refusals["outside"].replace("{script}", "scripts/none.py")
        assert asyncio.run(skill.run_skill_script(skill="pdf", script="forms.md")) == refusals["not_runnable"].replace("{script}", "forms.md")

    def test_unrestricted_is_refused(self):
        skill = _skill(sandbox={"preset": "unrestricted"})
        assert asyncio.run(skill.run_skill_script(skill="pdf", script="scripts/fill_form.py")) == FIXTURE["scripts"]["refusals"]["unrestricted"]

    def test_an_invalid_declaration_is_refused(self):
        skill = _skill(sandbox={"preset": "stirct"})
        out = asyncio.run(skill.run_skill_script(skill="pdf", script="scripts/fill_form.py"))
        assert out.startswith(FIXTURE["scripts"]["refusals"]["invalid_declaration"].split("{message}")[0])
        assert "stirct" in out

    def test_the_policy_keeps_the_skill_folder_read_only_and_readable(self, tmp_path):
        skill = _skill(agent_dir=tmp_path)
        pdf = discover_skills(str(REPO), ["skills"]).skills[0]
        skill.skills[pdf.name] = pdf
        undeclared = skill.script_policy(pdf)
        assert undeclared.preset == FIXTURE["scripts"]["policy"]["undeclared"]["preset"]
        assert undeclared.scoped_reads and pdf.directory in undeclared.read_roots
        assert pdf.directory in undeclared.deny_writes
        assert os.path.realpath(tmp_path) not in undeclared.write_roots
        assert undeclared.network_domains == []
        declared = _skill(agent_dir=tmp_path, sandbox={"preset": "development", "network": ["example.com"]})
        declared.skills[pdf.name] = pdf
        policy = declared.script_policy(pdf)
        assert policy.preset == "development" and os.path.realpath(tmp_path) in policy.write_roots
        assert pdf.directory in policy.deny_writes
        assert policy.network_domains == ["example.com"]

    def test_the_command_quotes_every_argument(self):
        skill = _skill()
        target = str(REPO / "skills" / "pdf" / "scripts" / "fill_form.py")
        assert skill.script_command(target, ["a b", "$HOME"]) == f"python3 {target} 'a b' '$HOME'"
