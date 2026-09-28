"""
The delegate fallback sends the platform credential to the platform's own
origin only (S-308, 2026-09-27, the agent-secrets lane): the origins are the
shared fixture's (`tests/fixtures/agent_secrets/delegate_credentials.json`),
and a non-platform https target receives neither `Authorization` nor
`X-API-Key`, on the one header dict `nli_tool` hands to both transports. The
TypeScript suite runs the same in
`tests/unit/skills/agent-secrets-nli-credential.test.ts`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.nli import NLISkill

FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures" / "agent_secrets" / "delegate_credentials.json").read_text())
PLATFORM_TOKEN = "platform-token-fixture"


def build():
    nli = NLISkill(
        {
            "transport": "http",
            "agent_base_url": FIXTURE["platform_base"],
            "timeout": 5.0,
            "max_retries": 0,
            "robutler_api_key": PLATFORM_TOKEN,
        }
    )
    agent = BaseAgent(name="caller", instructions="Caller", skills={"nli": nli})
    return agent, nli


async def headers_sent_to(target: str, monkeypatch) -> Dict[str, str]:
    """The headers `nli_tool` posts to `target` on the fallback path, with no card served and no platform lookups."""
    caller, nli = build()
    monkeypatch.setattr("webagents.server.context.context_vars.get_context", lambda: None)
    await caller._ensure_skills_initialized()
    nli._resolve_agent_id = AsyncMock(return_value=None)
    nli._mint_owner_assertion = AsyncMock(return_value=None)
    nli._probe_a2a_card = AsyncMock(return_value=None)
    nli.http_client = AsyncMock()
    nli.http_client.post = AsyncMock(return_value=MagicMock(status_code=403, text="no"))
    try:
        await nli.nli_tool(agent=target, message="hi")
        assert nli.http_client.post.await_count == 1, "the fallback did not post"
        return dict(nli.http_client.post.call_args.kwargs["headers"])
    finally:
        nli.http_client = None
        await nli.cleanup()


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["name"] for c in FIXTURE["cases"]])
def test_where_the_platform_credential_may_go(case):
    _agent, nli = build()
    assert nli._sends_credential_to(case["target"]) is case["sends"]


def test_a_target_that_is_not_an_absolute_url_gets_nothing():
    _agent, nli = build()
    assert nli._sends_credential_to("") is False
    assert nli._sends_credential_to("not a url at all") is False
    assert nli._sends_credential_to("/agents/helper") is False


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["name"] for c in FIXTURE["cases"]])
def test_the_fallback_headers(case, monkeypatch):
    sent = asyncio.run(headers_sent_to(case["target"], monkeypatch))
    lower = {name.lower(): value for name, value in sent.items()}
    for header in FIXTURE["credential_headers"]:
        name = header.lower()
        if case["sends"] and name == "authorization":
            assert lower[name] == f"Bearer {PLATFORM_TOKEN}"
        elif case["sends"] and name == "x-api-key":
            assert lower[name] == PLATFORM_TOKEN
        else:
            # X-Forwarded-Auth is the TypeScript skill's header; here it is never set.
            assert name not in lower, header
    # What is not a credential goes as before.
    assert lower["content-type"] == "application/json"
    assert lower["x-origin-agent"] == "caller"
