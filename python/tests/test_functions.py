"""
Python WebAgents — function skill suite tests.

Mirrors a subset of the TypeScript unit tests so cross-language doc
snippets stay honest.
"""

from __future__ import annotations
import pytest

from webagents.agents.skills.platform.functions import (
    FunctionRuntimeSkill,
    StubExecutorClient,
    CronSkill,
    CronScheduleEntry,
    CustomHttpSkill,
    CustomHttpEndpointEntry,
    CustomToolsSkill,
    CustomToolEntry,
    HostSelfEditSkill,
    FunctionManifest,
)


def test_runtime_skill_lists_declared_functions():
    skill = FunctionRuntimeSkill(
        agent_id="agent_a",
        executor=StubExecutorClient(),
        functions={
            "calculator": {"manifest": {"runtime": "js-v1"}, "cacheKey": "abc"},
            "stripe":     {"manifest": {"runtime": "js-v1"}, "cacheKey": "def"},
        },
    )
    assert sorted(skill.list()) == ["calculator", "stripe"]


def test_runtime_skill_has_no_runtime_dependencies():
    # Functions access KV via the executor and secrets via the host
    # resolver; neither path calls into a memory skill instance, so
    # `function-runtime` declares no runtime deps. Mounting memory is
    # an orthogonal opt-in choice for the agent author.
    skill = FunctionRuntimeSkill(agent_id="a")
    assert list(skill.dependencies) == []


def test_consumer_skills_depend_on_function_runtime():
    for cls in (CronSkill, CustomHttpSkill, CustomToolsSkill):
        skill = cls()
        assert "function-runtime" in skill.dependencies


def test_cron_renders_prompt():
    skill = CronSkill([
        CronScheduleEntry(id="nightly", cron="0 9 * * *", use="dailyReport"),
        CronScheduleEntry(id="poll",    cron="*/5 * * * *", use=None),
    ])
    prompt = skill.system_prompt
    assert "0 9 * * *" in prompt
    assert "*/5 * * * *" in prompt


def test_custom_http_renders_prompt():
    skill = CustomHttpSkill([
        CustomHttpEndpointEntry(id="stripe", use="stripeHandler", method="POST", path="/webhooks/stripe", auth="signature"),
    ])
    assert "POST /webhooks/stripe" in skill.system_prompt


def test_custom_tools_renders_prompt():
    skill = CustomToolsSkill([
        CustomToolEntry(id="calc", use="calculator", name="calculate", description="Evaluate math"),
    ])
    assert "calculate" in skill.system_prompt


def test_host_self_edit_only_enabled_when_flag_on():
    skill = HostSelfEditSkill(agent_id="a", owner_id="u", portal_base_url="http://x", feature_flag_enabled=False)
    assert skill.system_prompt == ""

    skill_on = HostSelfEditSkill(agent_id="a", owner_id="u", portal_base_url="http://x", feature_flag_enabled=True)
    assert "declare_function" in skill_on.system_prompt


def test_host_self_edit_rejects_non_owner():
    skill = HostSelfEditSkill(agent_id="a", owner_id="owner", portal_base_url="http://x", feature_flag_enabled=True)

    class FakeAuth:
        userId = "stranger"

    class FakeCtx:
        auth = FakeAuth()

    with pytest.raises(RuntimeError, match="owner"):
        skill._assert_owner(FakeCtx())


def test_manifest_validates_runtime():
    m = FunctionManifest(runtime="js-v1", entrypoint="handler")
    assert m.runtime == "js-v1"
