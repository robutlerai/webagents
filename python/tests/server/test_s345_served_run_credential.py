"""
S-345 (2026-09-29): the legacy daemon class runs every completion with the
request on the context.

`AuthSkill` reads the bearer from `context.request.headers`
(`_extract_api_key_from_context`), and `server/core/app.py`'s routes build one
context per request with `create_context(request=request)` before they run the
agent. The legacy daemon class (`webagents.cli.daemon.WebAgentsDaemon`) took
the parsed body as `request: dict` and ran `agent.run` / `agent.run_streaming`
on a bare context: no request, so an auth skill on the served agent saw no
credential and could neither verify nor refuse one, and a refusal that did
surface was a 500. The TypeScript twin of this file is
`typescript/tests/unit/server/s345-served-run-credential.test.ts`.

Pinned here with a witness that reads exactly where the auth skill reads and
refuses what it would refuse (a token it does not know, and no token at all):

  * the run's `on_connection` hook sees the request (its headers, its path)
    and the body's `metadata` (the platform's sender attribution);
  * a credential the hook refuses is a 401 with the `invalid_token` challenge
    and the model is not reached, streaming or not, on the skill branch and
    on the fallback;
  * on loopback, where the floor is off, the hook still decides.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from webagents.access.caller import CallerAuth
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.skills.robutler.auth.skill import AuthenticationError
from webagents.agents.tools.decorators import handoff, hook
from webagents.cli.daemon.server import WebAgentsDaemon
from webagents.server.core.credential_floor import bearer_challenge

OWNER = "Bearer owner-token"
BAD = "Bearer bad-token"
INVALID_TOKEN = bearer_challenge(refused=True)
PLAIN = bearer_challenge()
COMPLETIONS = "/agents/guarded/chat/completions"
HEADERS = {"X-Api-Key": "key-1", "X-Owner-Assertion": "assert-1", "User-Agent": "s345-probe"}


class Witness(Skill):
    """Stands in for `AuthSkill`, reading where it reads and refusing what it
    refuses: `owner-token` in the request's Authorization header is the
    owner; any other token, and no token at all, is refused with the auth
    skill's own error. It records what it saw."""

    identifies_caller = True

    def __init__(self):
        super().__init__({})
        self.seen: List[Dict[str, Any]] = []

    @hook("on_connection", priority=0, scope="all")
    async def who(self, context):
        request = getattr(context, "request", None)
        headers = dict(request.headers) if request is not None else {}
        self.seen.append(
            {
                "headers": headers,
                "target": f"{request.url.path}?{request.url.query}" if request is not None else None,
                "metadata": getattr(context, "metadata", None),
            }
        )
        bearer = headers.get("authorization")
        if bearer == OWNER:
            context.auth = CallerAuth(scope="owner", provider="test", user_id="owner-1")
            return context
        raise AuthenticationError("this bearer is refused" if bearer == BAD else "no credential seen")


class Model(Skill):
    """A model that answers `ok`, streaming as the OpenAI skill does, and
    counts how often it was reached."""

    def __init__(self):
        super().__init__({})
        self.reached = 0

    @handoff(name="model")
    async def answer(self, messages, tools=None, **kwargs):
        self.reached += 1
        head = {"id": "cmpl-1", "object": "chat.completion.chunk", "model": "model"}
        yield {**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]}
        yield {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield {**head, "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class Completions(Skill):
    """Stands in for `CompletionsTransportSkill`: the daemon's skill branch,
    which it takes when the agent has a skill named `completions` with a
    `chat_completions` method."""

    async def chat_completions(self, messages=None, stream: bool = False, tools=None, **kwargs):
        if stream:
            async for chunk in self.agent.run_streaming(messages, tools=tools):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        else:
            yield await self.agent.run(messages, tools=tools)


def _agent(path: str):
    witness = Witness()
    model = Model()
    skills: Dict[str, Skill] = {"witness": witness, "model": model}
    if path == "skill":
        skills["completions"] = Completions()
    agent = BaseAgent(name="guarded", instructions="Guarded.", skills=skills)
    asyncio.run(agent._ensure_skills_initialized())
    return agent, witness, model


def _daemon(agent, host: Optional[str] = "0.0.0.0") -> TestClient:
    daemon = WebAgentsDaemon(port=0, host=host)
    daemon.manager._loaded_agents["guarded"] = agent
    return TestClient(daemon.app, raise_server_exceptions=False)


def _body(stream: bool, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}], "stream": stream}
    if metadata is not None:
        body["metadata"] = metadata
    return body


@pytest.mark.parametrize("path", ["fallback", "skill"])
@pytest.mark.parametrize("stream", [False, True])
def test_the_hook_sees_the_request_and_the_owner_runs(path, stream):
    agent, witness, model = _agent(path)
    response = _daemon(agent).post(f"{COMPLETIONS}?probe=1", json=_body(stream), headers={"Authorization": OWNER, **HEADERS})
    assert response.status_code == 200, response.text
    assert model.reached == 1
    assert len(witness.seen) == 1
    seen = witness.seen[0]
    # The request, where `AuthSkill` reads the credential headers from.
    assert seen["headers"]["authorization"] == OWNER
    assert seen["headers"]["x-api-key"] == "key-1"
    assert seen["headers"]["x-owner-assertion"] == "assert-1"
    assert seen["headers"]["user-agent"] == "s345-probe"
    assert seen["target"] == f"{COMPLETIONS}?probe=1"
    # And the answer is the model's.
    assert "ok" in response.text
    if stream:
        assert "data: " in response.text


@pytest.mark.parametrize("path", ["fallback", "skill"])
@pytest.mark.parametrize("stream", [False, True])
def test_a_refused_credential_is_401_invalid_token_and_the_model_is_not_reached(path, stream):
    agent, _witness, model = _agent(path)
    response = _daemon(agent).post(COMPLETIONS, json=_body(stream), headers={"Authorization": BAD})
    assert response.status_code == 401, response.text
    assert response.headers["WWW-Authenticate"] == INVALID_TOKEN
    assert response.json() == {"error": {"code": "unauthorized", "message": "this bearer is refused"}}
    assert model.reached == 0


@pytest.mark.parametrize("stream", [False, True])
def test_the_bodys_metadata_reaches_the_hook(stream):
    agent, witness, _model = _agent("fallback")
    metadata = {"chat_id": "chat-1", "sender": {"id": "user-9", "username": "nine"}}
    response = _daemon(agent).post(COMPLETIONS, json=_body(stream, metadata), headers={"Authorization": OWNER})
    assert response.status_code == 200, response.text
    assert witness.seen[0]["metadata"] == metadata


@pytest.mark.parametrize("stream", [False, True])
def test_on_loopback_the_hook_still_decides(stream):
    # No floor on loopback (the CLI sends no credential), so the hook is the
    # only gate there: a token it refuses is `invalid_token`, no token at
    # all is the plain challenge, and neither reaches the model.
    agent, _witness, model = _agent("fallback")
    client = _daemon(agent, host=None)
    refused = client.post(COMPLETIONS, json=_body(stream), headers={"Authorization": BAD})
    assert refused.status_code == 401, refused.text
    assert refused.headers["WWW-Authenticate"] == INVALID_TOKEN
    anonymous = client.post(COMPLETIONS, json=_body(stream))
    assert anonymous.status_code == 401, anonymous.text
    assert anonymous.headers["WWW-Authenticate"] == PLAIN
    assert anonymous.json()["error"]["message"] == "no credential seen"
    assert model.reached == 0


def test_a_body_that_is_not_json_is_refused_without_a_run():
    agent, _witness, model = _agent("fallback")
    response = _daemon(agent).post(COMPLETIONS, content="{not json", headers={"Authorization": OWNER, "Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert model.reached == 0
