"""The discovery skill's `search` tool, the TypeScript skill's call for call (2026-09-25).

An agent file naming `discovery` has to give its model the same tool under
either CLI, so these are the cases `typescript/tests/unit/skills/discovery/search.test.ts`
pins, run against the Python skill: the definition both SDKs are checked
against (`tests/fixtures/discovery_tool/definition.json`), the platform routes
and what each is sent, the result shapes, the order of the answer, and the
sentence when nothing came back. The credential rule is pinned separately in
`tests/test_discovery_credentials.py`.
"""

import asyncio
import json
import logging
from pathlib import Path

import httpx
import pytest

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.discovery.skill import (
    DEFAULT_PLATFORM_URL,
    NO_DISCOVERY_CREDENTIAL,
    NO_DISCOVERY_SIGN_IN,
    DiscoverySkill,
    resolve_platform_url,
)

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures" / "discovery_tool" / "definition.json").read_text()
)
PORTAL = "https://portal.test"
POST_ID = "0b6f7a52-3c1d-4e8f-9a2b-1c2d3e4f5a6b"


class FakeAgent:
    name = "finder"
    api_key = None
    intents = None


@pytest.fixture
def platform(monkeypatch):
    """A stub platform behind `httpx.AsyncClient`. `routes` maps a path to a
    response (or to a coroutine function returning one); everything else is
    404. Every request is recorded in `seen`."""

    class Platform:
        routes = {}
        seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        Platform.seen.append(request)
        answer = Platform.routes.get(request.url.path)
        if answer is None:
            return httpx.Response(404, json={})
        if callable(answer):
            return await answer(request)
        return answer

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **kw: real_client(*a, transport=httpx.MockTransport(handler), **kw)
    )
    for name in ("WEBAGENTS_PUBLIC_URL", "WEBAGENTS_API_KEY", "WEBAGENTS_AGENT_TOKEN", "SERVICE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    return Platform


async def skill(**config):
    s = DiscoverySkill({"robutler_api_url": PORTAL, "robutler_api_key": "test-key", **config})
    await s.initialize(FakeAgent())
    return s


def ok(body):
    return httpx.Response(200, json=body)


def request_to(platform, path):
    return next(r for r in platform.seen if r.url.path == path)


# ---------------------------------------------------------------------------
# The cases the TypeScript suite pins
# ---------------------------------------------------------------------------


def test_offers_the_definition_in_the_shared_fixture():
    discovery = DiscoverySkill({"robutler_api_key": "k"})
    agent = BaseAgent(name="d", instructions="x", skills={"discovery": discovery})
    offered = [t for t in agent.get_all_tools() if getattr(t["function"], "__self__", None) is discovery]
    # One tool, as in TypeScript: publishing is not something the model can do.
    assert [t["name"] for t in offered] == ["search"]
    assert offered[0]["definition"] == FIXTURE["definition"]


def test_the_refusal_without_a_credential_is_the_shared_sentence():
    assert NO_DISCOVERY_CREDENTIAL == FIXTURE["no_credential"]
    assert NO_DISCOVERY_SIGN_IN == FIXTURE["no_sign_in"]


@pytest.mark.asyncio
async def test_intents_are_searched_by_post_and_agents_listed_by_get(platform):
    platform.routes = {
        "/api/intents/search": ok({"results": [{"intent": "generate images", "agentId": "agent-1", "score": 0.9}]}),
        "/api/discovery/agents": ok({"agents": [{"id": "agent-1", "username": "image-gen", "displayName": "Image Generator"}]}),
    }
    s = await skill()
    result = await s.search(query="generate images", types=["intents", "agents"], limit=5)

    intents = request_to(platform, "/api/intents/search")
    assert intents.method == "POST"
    assert intents.content == b'{"query":"generate images","limit":5}'
    agents = request_to(platform, "/api/discovery/agents")
    assert agents.method == "GET"
    assert agents.url.query == b"search=generate+images&type=agent&limit=5"

    assert result == {
        "intents": [{"intent": "generate images", "agentId": "agent-1", "score": 0.9}],
        "agents": [{"username": "image-gen", "display_name": "Image Generator", "reputation": 0, "trust_level": "standard"}],
    }


@pytest.mark.asyncio
async def test_other_types_use_their_own_discovery_route(platform):
    platform.routes = {"/api/discovery/posts": ok({"posts": [{"id": "p1", "title": "AI post"}]})}
    s = await skill()
    result = await s.search(query="artificial intelligence", types=["posts"], limit=20)

    assert request_to(platform, "/api/discovery/posts").url.query == b"q=artificial+intelligence&limit=20"
    # Cut to what the tool promises, never the whole post.
    assert result == {"posts": [{"id": "p1", "title": "AI post", "likes": 0}]}


@pytest.mark.asyncio
async def test_empty_answers_are_empty_lists(platform):
    platform.routes = {
        "/api/intents/search": ok({"results": []}),
        "/api/discovery/agents": ok({"agents": []}),
        "/api/discovery/channels": ok({}),
    }
    s = await skill()
    assert await s.search(query="nonexistent", types=["intents", "agents"]) == {"intents": [], "agents": []}
    assert await s.search(query="empty", types=["channels"]) == {"channels": []}


@pytest.mark.asyncio
async def test_a_failed_type_is_left_out_and_nothing_at_all_says_which(platform):
    platform.routes = {"/api/intents/search": httpx.Response(500, json={"error": "Internal Server Error"})}
    s = await skill()
    assert await s.search(query="fail", types=["intents"]) == {"error": "Search failed: intents 500."}

    platform.routes = {
        "/api/intents/search": ok({"results": []}),
        "/api/discovery/posts": httpx.Response(403, json={"error": "Forbidden"}),
    }
    assert await s.search(query="mixed", types=["intents", "posts"]) == {"intents": []}

    platform.routes = {
        "/api/intents/search": httpx.Response(401, json={"error": "Unauthorized"}),
        "/api/discovery/agents": httpx.Response(401, json={"error": "Unauthorized"}),
    }
    assert await s.search(query="x", types=["intents", "agents"]) == {
        "error": "Search failed: intents 401, agents 401.",
    }


@pytest.mark.asyncio
async def test_reads_a_types_own_key_then_results(platform):
    platform.routes = {"/api/discovery/users": ok({"users": [{"id": "u1", "name": "Alice"}]})}
    s = await skill()
    assert await s.search(query="alice", types=["users"]) == {"users": [{"id": "u1", "name": "Alice"}]}


@pytest.mark.asyncio
async def test_defaults_are_intents_agents_posts_and_ten(platform):
    platform.routes = {
        "/api/intents/search": ok({"results": [{"intent": "default", "agentId": "x", "score": 1}]}),
        "/api/discovery/agents": ok({"agents": [{"id": "x", "username": "agent-x"}]}),
        "/api/discovery/posts": ok({"posts": []}),
    }
    s = await skill()
    result = await s.search(query="test")
    assert list(result) == ["intents", "agents", "posts"]
    assert json.loads(request_to(platform, "/api/intents/search").content)["limit"] == 10
    assert request_to(platform, "/api/discovery/posts").url.query == b"q=test&limit=10"


@pytest.mark.asyncio
async def test_runs_the_types_in_parallel(platform):
    order = []

    def slow(kind, body):
        async def answer(request):
            order.append(f"{kind}_start")
            await asyncio.sleep(0.01)
            order.append(f"{kind}_end")
            return ok(body)

        return answer

    platform.routes = {
        "/api/intents/search": slow("intents", {"results": []}),
        "/api/discovery/agents": slow("agents", {"agents": []}),
        "/api/discovery/posts": slow("posts", {"posts": []}),
    }
    s = await skill()
    await s.search(query="parallel", types=["intents", "agents", "posts"])
    starts = [e for e in order if e.endswith("_start")]
    first_end = next(i for i, e in enumerate(order) if e.endswith("_end"))
    assert len(starts) == 3
    assert first_end >= len(starts)


# ---------------------------------------------------------------------------
# What both SDKs added on 2026-09-25
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answers_in_the_order_asked_for(platform):
    async def late(request):
        await asyncio.sleep(0.015)
        return ok({"channels": [{"slug": "c"}]})

    platform.routes = {"/api/discovery/channels": late, "/api/discovery/tags": ok({"tags": [{"name": "t"}]})}
    s = await skill()
    result = await s.search(query="x", types=["channels", "tags"])
    assert list(result) == ["channels", "tags"]


@pytest.mark.asyncio
async def test_cuts_a_post_to_an_excerpt_with_author_channel_and_likes(platform):
    platform.routes = {
        "/api/discovery/posts": ok({
            "posts": [{
                "id": "p1", "title": "T", "content": "x" * 1000, "humanLikes": 2, "agentLikes": 3,
                "author": {"username": "alice", "avatarUrl": "https://img"}, "channel": {"slug": "news", "name": "News"},
            }],
        }),
    }
    s = await skill()
    result = await s.search(query="x", types=["posts"])
    assert result == {"posts": [{"id": "p1", "title": "T", "content": "x" * 300, "author": "alice", "channel": "news", "likes": 5}]}


@pytest.mark.asyncio
async def test_an_excerpt_is_counted_as_javascript_counts(platform):
    # 299 letters and an emoji (two UTF-16 units): JavaScript's `slice(0, 300)`
    # keeps the letters and half the pair, which a JSON round trip drops.
    platform.routes = {"/api/discovery/posts": ok({"posts": [{"id": "p", "content": "a" * 299 + "\U0001F600" + "b"}]})}
    s = await skill()
    result = await s.search(query="x", types=["posts"])
    assert result["posts"][0]["content"] == "a" * 299


@pytest.mark.asyncio
async def test_gives_an_agent_result_its_url(platform):
    platform.routes = {
        "/api/discovery/agents": ok({
            "agents": [{"username": "jurist", "displayName": "Jurist", "agentUrl": "https://legal.example.com/jurist", "reputationScore": 12}],
        }),
    }
    s = await skill()
    result = await s.search(query="x", types=["agents"])
    assert result == {
        "agents": [{"username": "jurist", "display_name": "Jurist", "url": "https://legal.example.com/jurist", "reputation": 12, "trust_level": "standard"}],
    }


@pytest.mark.asyncio
async def test_fetches_a_post_named_by_its_url_or_id_first(platform):
    platform.routes = {
        f"/api/posts/{POST_ID}": ok({"id": POST_ID, "title": "Named", "content": "c", "humanLikes": 1}),
        "/api/discovery/posts": ok({"posts": [{"id": "other", "title": "Other"}]}),
    }
    s = await skill()
    by_url = await s.search(query=f"what about https://robutler.ai/p/{POST_ID} ?", types=["posts"])
    assert [p["id"] for p in by_url["posts"]] == [POST_ID, "other"]
    by_id = await s.search(query=POST_ID, types=["intents"])
    # Not asked for posts, still given the post.
    assert by_id["posts"] == [{"id": POST_ID, "title": "Named", "content": "c", "likes": 1}]


@pytest.mark.asyncio
async def test_post_filters_and_sort_names(platform):
    platform.routes = {"/api/discovery/posts": ok({"posts": []})}
    s = await skill()
    await s.search(query="video", types=["posts"], channel="marketplace/genai/video", tag="ai", sort="popular")
    assert request_to(platform, "/api/discovery/posts").url.query == (
        b"q=video&limit=10&channel=marketplace%2Fgenai%2Fvideo&tag=ai&sort=top"
    )


@pytest.mark.asyncio
async def test_logs_progress_without_the_query_and_prints_nothing(platform, capsys):
    platform.routes = {"/api/intents/search": httpx.Response(500, json={})}
    s = await skill()
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    # On the skill's logger itself: the `webagents` loggers do not propagate
    # to the root, where pytest's own capture listens.
    target = logging.getLogger("webagents.skill.discovery")
    handler, level = Collect(level=logging.DEBUG), target.level
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
    try:
        await s.search(query="private words", types=["intents", "agents"])
    finally:
        target.removeHandler(handler)
        target.setLevel(level)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    lines = [r.getMessage() for r in records]
    assert any(line.startswith("[search] intents 500 in ") for line in lines), lines
    assert not any("private words" in line for line in lines)


# ---------------------------------------------------------------------------
# Which platform (the TypeScript order)
# ---------------------------------------------------------------------------


def test_the_platform_is_the_config_then_robutler_api_url_then_the_internal_url(monkeypatch):
    monkeypatch.setenv("ROBUTLER_API_URL", "https://api.example.com/")
    monkeypatch.setenv("ROBUTLER_INTERNAL_API_URL", "http://internal:3000")
    assert resolve_platform_url({"robutler_api_url": "https://mine.example.com/"}) == "https://mine.example.com"
    assert resolve_platform_url({}) == "https://api.example.com"
    monkeypatch.delenv("ROBUTLER_API_URL")
    assert resolve_platform_url({}) == "http://internal:3000"


def test_the_platform_is_the_one_the_cli_is_pointed_at_then_robutler_ai(monkeypatch, tmp_path):
    for name in ("ROBUTLER_API_URL", "ROBUTLER_INTERNAL_API_URL", "WEBAGENTS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    (home / ".webagents").mkdir(parents=True)
    (home / ".webagents" / "config.json").write_text(json.dumps({"platform.url": "https://macbook.example.ts.net/"}))
    monkeypatch.setenv("HOME", str(home))
    assert resolve_platform_url({}) == "https://macbook.example.ts.net"

    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    assert resolve_platform_url({}) == DEFAULT_PLATFORM_URL == "https://robutler.ai"
