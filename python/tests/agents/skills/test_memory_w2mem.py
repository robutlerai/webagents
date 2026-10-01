"""The memory skill against the fixture both SDKs run
(`tests/fixtures/memory_tool/definition.json`; TypeScript:
`typescript/tests/unit/skills/memory/memory-skill-w2mem.test.ts`): the four
tool definitions, what a `- memory: {...}` entry accepts, the namespace a
caller gets, where the local tier keeps it, the id an entry has everywhere,
the token estimate and the transcript (the agent's compaction counts the same
way, `agents/core/context_compaction.py`), and the frozen notes."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from webagents.agents.skills.local.memory.caller_scoped import MEMORY_TOOL_DEFINITIONS, MemorySkill, parse_memory_config
from webagents.agents.core.context_compaction import estimate_tokens, transcript_of
from webagents.agents.skills.local.memory.memory_namespace import (
    caller_key,
    entry_id_for,
    is_valid_key,
    key_refusal,
    local_dir_of,
    namespace_of,
    readable_namespaces,
    writable_namespaces,
)
from webagents.agents.skills.local.memory.memory_notes import render_notes
from webagents.cli.agent_builder import SKILL_CLASSES, load_skills

FIXTURE = json.loads((Path(__file__).resolve().parents[2] / "fixtures" / "memory_tool" / "definition.json").read_text())


class FakeAgent:
    name = "helper"
    skills = {}

    def __init__(self):
        self.tools = []

    def register_tool(self, fn, source=None, scope=None):
        self.tools.append(fn)


def _auth(case):
    """The caller in this SDK's terms: no `principals` when no access block ran."""
    fields = {"authenticated": True, "provider": "platform", "scope": case["tier"] or "user", "user_id": case["user_id"]}
    if case["principals"] is not None:
        fields["principals"] = case["principals"]
    if not any((case["tier"], case["principals"], case["user_id"])):
        fields["authenticated"] = False
    return SimpleNamespace(**fields)


def test_the_definitions_both_sdks_share(tmp_path):
    assert MEMORY_TOOL_DEFINITIONS == FIXTURE["definitions"]
    skill = MemorySkill({"agent_path": str(tmp_path)})
    agent = FakeAgent()
    asyncio.run(skill.initialize(agent))
    assert [t._webagents_tool_definition for t in agent.tools] == FIXTURE["definitions"]


@pytest.mark.parametrize("shape", FIXTURE["config_shapes"], ids=[s["case"] for s in FIXTURE["config_shapes"]])
def test_the_entry_is_checked_as_typescript_checks_it(shape):
    if "error" in shape:
        with pytest.raises(ValueError) as refused:
            parse_memory_config(shape["config"])
        assert str(refused.value) == shape["error"]
    else:
        parsed = parse_memory_config(shape["config"])
        assert (parsed["local"], parsed["portal"]) == (shape["local"], shape["portal"])


def test_defaults():
    parsed = parse_memory_config({})
    assert parsed["notes_budget"] == FIXTURE["defaults"]["notes_budget"]
    assert parsed["compaction"] == FIXTURE["defaults"]["compaction"]


def test_named_memory_in_an_agent_file(tmp_path):
    assert SKILL_CLASSES["memory"].endswith("caller_scoped.MemorySkill")
    report = {}
    skills = load_skills(["memory"], agent_name="helper", agent_path=tmp_path / "AGENT.md", report=report)
    assert list(skills) == ["memory"] and report == {}
    assert skills["memory"].tiers == {"local": True, "portal": False}
    assert skills["memory"].memory_root == tmp_path / ".webagents" / "memory"
    report = {}
    load_skills([{"memory": {"portal": True, "cloud": True}}], agent_name="helper", agent_path=tmp_path / "AGENT.md", report=report)
    assert report["failed"] == [("memory", 'memory: unknown key "cloud". It takes local, portal, notes_budget and compaction.')]


@pytest.mark.parametrize("case", FIXTURE["namespaces"], ids=[c["case"] for c in FIXTURE["namespaces"]])
def test_the_namespace_comes_from_the_verified_caller(case):
    assert namespace_of(_auth(case)) == case["namespace"]


def test_no_auth_is_nobody():
    assert namespace_of(None) is None


@pytest.mark.parametrize("case", FIXTURE["visibility"], ids=[str(c["namespace"]) for c in FIXTURE["visibility"]])
def test_reads_and_writes(case):
    reads = readable_namespaces(case["namespace"])
    assert ("*" if reads is None else reads) == case["reads"]
    assert writable_namespaces(case["namespace"]) == case["writes"]


@pytest.mark.parametrize("case", FIXTURE["local_dirs"], ids=[c["namespace"] for c in FIXTURE["local_dirs"]])
def test_local_dirs(case):
    assert local_dir_of(case["namespace"]) == case["dir"]


def test_caller_key_is_the_session_skills():
    assert caller_key("user:4c3ad5e6-510d-486d-af50-426ba16b6b6c") == "9691dcadd7ab4f5c3c4e8120264e66e4"


@pytest.mark.parametrize("case", FIXTURE["entry_ids"]["cases"], ids=[f"{c['namespace']}/{c['key']}" for c in FIXTURE["entry_ids"]["cases"]])
def test_entry_ids(case):
    assert entry_id_for(FIXTURE["entry_ids"]["store"], case["namespace"], case["key"]) == case["id"]


def test_keys():
    for key in FIXTURE["keys"]["valid"]:
        assert is_valid_key(key), key
    for key in FIXTURE["keys"]["invalid"]:
        assert not is_valid_key(key), key
        assert key_refusal(key) == FIXTURE["keys"]["refused"].replace("{key}", key)


@pytest.mark.parametrize("case", FIXTURE["tokens"]["cases"])
def test_tokens(case):
    assert estimate_tokens(case["messages"]) == case["tokens"]


@pytest.mark.parametrize("case", FIXTURE["transcript"]["cases"])
def test_transcript(case):
    assert transcript_of(case["messages"]) == case["text"]


@pytest.mark.parametrize("case", FIXTURE["notes"]["cases"], ids=[c["case"] for c in FIXTURE["notes"]["cases"]])
def test_frozen_notes_rendering(case):
    assert render_notes(case["sections"], case["budget"]) == case["text"]
