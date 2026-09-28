"""
A2A v1.0 transport, replayed from the fixture both SDKs share
(`tests/fixtures/a2a/vectors.json`, `config_shapes.json`; plan item 1.3,
2026-09-26). The TypeScript twin is `typescript/tests/unit/transport/a2a-v1.test.ts`.

The agent under test is the fixture's echo agent: a handoff that answers
`echo: ` plus the input texts, one tool anyone may use and one only the owner
may, and an identity skill that turns a bearer into a principal (so tasks are
owned by principal here; the credential-hash key is tested on its own below).
It is served by `WebAgentsServer` under `/agents/<name>`: the static mount the
server dispatches a skill's `@http` routes through, behind the credential
floor, with the signing identity the server mints for a static agent, so the
card comes out signed. One `with TestClient(...)` per test, because the tasks
a send starts keep running between requests only while the client keeps its
event loop; a client used without `with` makes a new loop per request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.skills.core.transport.a2a.a2a_client import (
    A2AClientError,
    call_agent,
    pick_interface,
    read_send_result,
    reply_of_result,
)
from webagents.agents.skills.core.transport.a2a.card import (
    b64url,
    b64url_decode,
    card_signing_bytes,
    card_signing_payload,
    ed25519_card_signer,
    hmac_card_signer,
    sign_agent_card,
    verify_agent_card,
)
from webagents.agents.skills.core.transport.a2a.jcs import canonicalize
from webagents.agents.skills.core.transport.a2a.protocol import read_part
from webagents.agents.skills.core.transport.a2a.skill import (
    A2ATransportSkill,
    peer_token_for,
    resolve_a2a_settings,
)
from webagents.agents.skills.robutler.auth.skill import AuthenticationError
from webagents.agents.tools.decorators import handoff, hook, tool
from webagents.crypto.http_signature import SigningKey
from webagents.server.core.app import WebAgentsServer

FIXTURES = Path(__file__).parent / "fixtures" / "a2a"
VECTORS = json.loads((FIXTURES / "vectors.json").read_text(encoding="utf-8"))
CONFIG_SHAPES = json.loads((FIXTURES / "config_shapes.json").read_text(encoding="utf-8"))

AGENT = VECTORS["agent"]
URL_PREFIX = "/agents"
ORIGIN = "https://agent.example.test"
BASE_PATH = f"{URL_PREFIX}/{AGENT['name']}"
RPC = f"{BASE_PATH}{AGENT['rpc_path']}"
SETTLED = {"TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"}


# ---------------------------------------------------------------------------
# The fixture agent
# ---------------------------------------------------------------------------


class EchoLLM(Skill):
    """The fixture's echo handoff. `delay_seconds` makes a run slow enough to
    be observed WORKING and canceled; `fail_with` makes it raise instead."""

    def __init__(self):
        super().__init__({})
        self.delay_seconds = 0.0
        self.fail_with: Optional[BaseException] = None

    @handoff(name="echo-llm")
    async def echo(self, messages, tools=None, **kwargs):
        texts: List[str] = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(
                    item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
                )
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.fail_with is not None:
            raise self.fail_with
        reply = AGENT["reply_prefix"] + " ".join(texts)
        yield {"choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


class ToolsSkill(Skill):
    @tool(name="public_lookup", description="Anyone may call this")
    async def public_lookup(self, q: str) -> str:
        return "ok"

    @tool(name="owner_only_admin", description="Only the owner may call this", scope="owner")
    async def owner_only_admin(self) -> str:
        return "ok"


class BearerIdentity(Skill):
    """Turns `Bearer <token>` into the principal `user:<token>`; refuses `Bearer refuse-me`."""

    identifies_caller = True

    @hook("on_connection", priority=1)
    async def identify(self, context):
        request = getattr(context, "request", None)
        header = request.headers.get("authorization") if request is not None else None
        token = header[len("Bearer ") :] if isinstance(header, str) and header.startswith("Bearer ") else None
        if not token:
            return context
        if token == "refuse-me":
            raise AuthenticationError("this bearer is refused")
        context.auth = SimpleNamespace(authenticated=True, user_id=f"user:{token}", scope="user", groups=[])
        return context


def build_agent(*, identity: bool = True, skill_config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
    llm = EchoLLM()
    skill = A2ATransportSkill(skill_config or {})
    skills: Dict[str, Skill] = {"echo-llm": llm, "tools": ToolsSkill(), "a2a": skill}
    if identity:
        skills["identity"] = BearerIdentity()
    agent = BaseAgent(name=name or AGENT["name"], instructions="Echo agent", skills=skills)
    agent.description = AGENT["description"]
    asyncio.run(agent._ensure_skills_initialized())
    return SimpleNamespace(agent=agent, llm=llm, skill=skill)


def build_server(tmp_path, *, identity: bool = True, skill_config: Optional[Dict[str, Any]] = None):
    """The fixture agent behind `WebAgentsServer`'s static mount, at
    `{ORIGIN}{BASE_PATH}`, with a signing identity minted into `tmp_path`."""
    built = build_agent(identity=identity, skill_config=skill_config)
    built.server = WebAgentsServer(
        agents=[built.agent],
        url_prefix=URL_PREFIX,
        public_url=ORIGIN,
        keys_dir=str(tmp_path / "keys"),
        enable_monitoring=False,
        enable_prometheus=False,
        enable_rate_limiting=False,
        enable_request_logging=False,
        heartbeat=False,
        quiet=True,
    )
    return built


@pytest.fixture
def served(tmp_path):
    built = build_server(tmp_path)
    with TestClient(built.server.app, raise_server_exceptions=False) as client:
        built.client = client
        yield built
    built.skill._store.clear()


def credential(name: str) -> str:
    return VECTORS["credentials"][name]


def rpc_call(client, caller: str, method: str, params: Any, headers: Optional[Dict[str, str]] = None, rpc_id: Any = None):
    merged = {"content-type": "application/json"}
    merged.update({"a2a-version": "1.0"} if headers is None else headers)
    merged["authorization"] = credential(caller)
    body = {"jsonrpc": "2.0", "id": rpc_id if rpc_id is not None else f"t-{uuid.uuid4()}", "method": method, "params": params}
    res = client.post(RPC, headers=merged, content=json.dumps(body))
    return res.status_code, res.json()


def poll_task(client, caller: str, task_id: str) -> Dict[str, Any]:
    for _ in range(200):
        _status, body = rpc_call(client, caller, "GetTask", {"id": task_id})
        task = (body.get("result") or {}).get("task")
        assert task, body
        if task["status"]["state"] in SETTLED:
            return task
        time.sleep(0.01)
    raise AssertionError("task never settled")


def reply_of(task: Dict[str, Any]) -> str:
    from_artifacts = "".join(p.get("text", "") for a in task["artifacts"] for p in a["parts"])
    if from_artifacts:
        return from_artifacts
    message = task["status"].get("message") or {}
    return "".join(p.get("text", "") for p in message.get("parts", []))


def sse_events(text: str) -> List[Any]:
    return [json.loads(chunk[len("data: ") :]) for chunk in text.split("\n\n") if chunk.startswith("data: ")]


def event_kind(event: Dict[str, Any]) -> str:
    return next(iter(event))


# ---------------------------------------------------------------------------
# JSON-RPC vectors
# ---------------------------------------------------------------------------


class TestJsonRpcVectors:
    @pytest.mark.parametrize("case", VECTORS["jsonrpc"], ids=[c["name"] for c in VECTORS["jsonrpc"]])
    def test_vector(self, served, case):
        client = served.client
        res = client.post(RPC, headers={**case["headers"], "authorization": credential(case["caller"])}, content=json.dumps(case["body"]))
        expect = case["expect"]
        assert res.status_code == expect["status"], res.text
        body = res.json()
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == case["body"].get("id")
        if "error" in expect:
            assert "result" not in body
            assert body["error"]["code"] == expect["error"]["code"]
            if expect["error"].get("reason"):
                info = body["error"]["data"][0]
                assert info["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
                assert info["reason"] == expect["error"]["reason"]
                assert info["domain"] == "a2a-protocol.org"
            return
        assert "error" not in body
        if "result_equals" in expect:
            assert body["result"] == expect["result_equals"]
            return
        assert expect["result_kind"] == "task"
        task = body["result"]["task"]
        assert isinstance(task["id"], str) and isinstance(task["contextId"], str)
        assert task["status"]["state"] in expect["state_in"]
        if "context_id" in expect:
            assert task["contextId"] == expect["context_id"]
        if "reply" in expect:
            assert reply_of(task) == expect["reply"]
        if "status_message_role" in expect:
            assert task["status"]["message"]["role"] == expect["status_message_role"]
        if "history_length" in expect:
            assert len(task["history"]) == expect["history_length"]
        if "history_first_parts" in expect:
            assert task["history"][0]["parts"] == expect["history_first_parts"]
        if "then_get_task" in expect:
            final = poll_task(client, case["caller"], task["id"])
            assert final["status"]["state"] == expect["then_get_task"]["final_state"]
            assert reply_of(final) == expect["then_get_task"]["reply"]
            assert final["artifacts"][0]["parts"][0]["text"] == expect["then_get_task"]["reply"]

    @pytest.mark.parametrize("flow", VECTORS["flows"], ids=[f["name"] for f in VECTORS["flows"]])
    def test_flow(self, served, flow):
        """Per-caller isolation, cancel after completion, and the context as the conversation."""
        client = served.client
        ids: Dict[str, str] = {}
        for step in flow["steps"]:
            if "send" in step:
                message: Dict[str, Any] = {"messageId": f"m-{uuid.uuid4()}", "role": "ROLE_USER", "parts": [{"text": step["send"]}]}
                if step.get("context_id"):
                    message["contextId"] = step["context_id"]
                _status, body = rpc_call(client, step["caller"], "SendMessage", {"message": message})
                task = body["result"]["task"]
                assert task["status"]["state"] == "TASK_STATE_COMPLETED"
                ids[step["as"]] = task["id"]
            elif "get" in step:
                _status, body = rpc_call(client, step["caller"], "GetTask", {"id": ids[step["get"]]})
                if "expect_error" in step:
                    assert body["error"]["code"] == step["expect_error"]["code"]
                    assert body["error"]["data"][0]["reason"] == step["expect_error"]["reason"]
                else:
                    task = body["result"]["task"]
                    if "expect_state" in step:
                        assert task["status"]["state"] == step["expect_state"]
                    if "expect_context_id" in step:
                        assert task["contextId"] == step["expect_context_id"]
            elif "cancel" in step:
                _status, body = rpc_call(client, step["caller"], "CancelTask", {"id": ids[step["cancel"]]})
                assert body["error"]["code"] == step["expect_error"]["code"]
                assert body["error"]["data"][0]["reason"] == step["expect_error"]["reason"]
            elif "list" in step:
                params = {"contextId": step["context_id"]} if step.get("context_id") else {}
                _status, body = rpc_call(client, step["caller"], "ListTasks", params)
                listed = [t["id"] for t in body["result"]["tasks"]]
                if "expect_contains" in step:
                    assert ids[step["expect_contains"]] in listed
                if "expect_not_contains" in step:
                    assert ids[step["expect_not_contains"]] not in listed
                if "expect_count" in step:
                    assert len(listed) == step["expect_count"]

    def test_streaming_send_streams_task_chunks_and_terminal_status(self, served):
        client = served.client
        streaming = VECTORS["streaming"]
        for body in (streaming["jsonrpc_body"], streaming["dotted_body"]):
            res = client.post(RPC, headers={"content-type": "application/json", "authorization": credential("caller_a")}, content=json.dumps(body))
            assert res.status_code == 200
            assert streaming["content_type"] in res.headers["content-type"]
            events = sse_events(res.text)
            assert len(events) >= 3
            for event in events:
                assert event["jsonrpc"] == "2.0"
                assert event["id"] == body["id"]
            results = [e["result"] for e in events]
            assert event_kind(results[0]) == streaming["expect"]["first_event"]
            assert results[0]["task"]["status"]["state"] in streaming["expect"]["first_state_in"]
            assert any("artifactUpdate" in r for r in results) is streaming["expect"]["has_artifact_update"]
            last = results[-1]
            assert event_kind(last) == streaming["expect"]["last_event"]
            assert last["statusUpdate"]["status"]["state"] == streaming["expect"]["last_state"]
            assert "".join(p["text"] for p in last["statusUpdate"]["status"]["message"]["parts"]) == streaming["expect"]["reply"]
            chunks = [r["artifactUpdate"]["artifact"]["parts"] for r in results if "artifactUpdate" in r]
            assert "".join(p["text"] for parts in chunks for p in parts) == streaming["expect"]["reply"]

    def test_subscribe_replays_a_finished_task_and_ends_at_its_terminal_status(self, served):
        client = served.client
        _status, body = rpc_call(client, "caller_a", "SendMessage", {"message": {"messageId": "sub-1", "role": "ROLE_USER", "parts": [{"text": "replay"}]}})
        task_id = body["result"]["task"]["id"]
        res = client.post(
            RPC,
            headers={"content-type": "application/json", "authorization": credential("caller_a")},
            content=json.dumps({"jsonrpc": "2.0", "id": "sub", "method": "SubscribeToTask", "params": {"id": task_id}}),
        )
        assert res.status_code == 200
        events = sse_events(res.text)
        assert event_kind(events[0]["result"]) == "task"
        assert events[-1]["result"]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
        # Another caller cannot subscribe to it.
        _status, other = rpc_call(client, "caller_b", "tasks/resubscribe", {"id": task_id}, headers={})
        assert other["error"]["code"] == -32001

    def test_a_running_task_is_working_under_return_immediately_and_can_be_canceled(self, served):
        client = served.client
        served.llm.delay_seconds = 0.3
        _status, body = rpc_call(
            client,
            "caller_a",
            "SendMessage",
            {"message": {"messageId": "slow-1", "role": "ROLE_USER", "parts": [{"text": "slow"}]}, "configuration": {"returnImmediately": True}},
        )
        task = body["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_WORKING"
        _status, got = rpc_call(client, "caller_a", "GetTask", {"id": task["id"]})
        assert got["result"]["task"]["status"]["state"] == "TASK_STATE_WORKING"
        _status, canceled = rpc_call(client, "caller_a", "CancelTask", {"id": task["id"]})
        assert canceled["result"]["task"]["status"]["state"] == "TASK_STATE_CANCELED"
        time.sleep(0.4)
        _status, after = rpc_call(client, "caller_a", "GetTask", {"id": task["id"]})
        assert after["result"]["task"]["status"]["state"] == "TASK_STATE_CANCELED"

    def test_a_blocking_send_answers_the_task_as_it_stands_after_the_timeout(self, tmp_path):
        slow = build_server(tmp_path, skill_config={"blocking_timeout_seconds": 0.05})
        slow.llm.delay_seconds = 1.0
        with TestClient(slow.server.app, raise_server_exceptions=False) as client:
            _status, body = rpc_call(client, "caller_a", "SendMessage", {"message": {"messageId": "b-1", "role": "ROLE_USER", "parts": [{"text": "later"}]}})
            task = body["result"]["task"]
            assert task["status"]["state"] == "TASK_STATE_WORKING"
            final = poll_task(client, "caller_a", task["id"])
            assert final["status"]["state"] == "TASK_STATE_COMPLETED"
            assert reply_of(final) == "echo: later"

    def test_an_identity_skills_refusal_is_the_gates_401_before_any_task_exists(self, served):
        client = served.client
        res = client.post(
            RPC,
            headers={"content-type": "application/json", "authorization": "Bearer refuse-me"},
            content=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {"message": {"messageId": "x", "role": "ROLE_USER", "parts": [{"text": "hi"}]}}}),
        )
        assert res.status_code == 401
        assert res.json()["error"]["code"] == "unauthorized"
        assert served.skill.task_count == 0

    def test_expired_tasks_are_gone_after_the_ttl(self, tmp_path):
        """On the store's own clock (`TaskStore(now=...)`), so a loaded machine
        cannot expire the task before the first read."""
        short = build_server(tmp_path, skill_config={"task_ttl_seconds": 60})
        clock = {"now": time.time()}
        short.skill._store._now = lambda: clock["now"]
        with TestClient(short.server.app, raise_server_exceptions=False) as client:
            _status, body = rpc_call(client, "caller_a", "SendMessage", {"message": {"messageId": "ttl-1", "role": "ROLE_USER", "parts": [{"text": "ttl"}]}})
            task_id = body["result"]["task"]["id"]
            clock["now"] += 59
            assert "result" in rpc_call(client, "caller_a", "GetTask", {"id": task_id})[1]
            clock["now"] += 2
            assert rpc_call(client, "caller_a", "GetTask", {"id": task_id})[1]["error"]["code"] == -32001
            assert short.skill.task_count == 0

    def test_a_failed_run_is_a_failed_task_carrying_the_status_and_a_safe_text(self, served):
        """An auth refusal inside the run keeps its status and its own text
        (it was written to be shown); any other error is the fixed text of
        S-228, never the exception's."""
        client = served.client
        served.llm.fail_with = AuthenticationError("the run refused this caller")
        _status, body = rpc_call(client, "caller_a", "SendMessage", {"message": {"messageId": "f-1", "role": "ROLE_USER", "parts": [{"text": "x"}]}})
        task = body["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert task["metadata"]["error"] == {"code": "unauthorized", "message": "the run refused this caller", "httpStatus": 401}
        assert reply_of(task) == "the run refused this caller"
        served.llm.fail_with = RuntimeError("secret provider detail")
        _status, body = rpc_call(client, "caller_a", "SendMessage", {"message": {"messageId": "f-2", "role": "ROLE_USER", "parts": [{"text": "x"}]}})
        task = body["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert task["metadata"]["error"]["code"] == "run_failed"
        assert "secret provider detail" not in json.dumps(task)


class TestTaskOwnershipWithoutAnIdentitySkill:
    def test_keys_tasks_on_the_presented_credential_so_another_bearer_sees_nothing(self, tmp_path):
        built = build_server(tmp_path, identity=False)
        with TestClient(built.server.app, raise_server_exceptions=False) as client:
            _status, body = rpc_call(client, "caller_a", "SendMessage", {"message": {"messageId": "c-1", "role": "ROLE_USER", "parts": [{"text": "cred"}]}})
            task_id = body["result"]["task"]["id"]
            assert "result" in rpc_call(client, "caller_a", "GetTask", {"id": task_id})[1]
            assert rpc_call(client, "caller_b", "GetTask", {"id": task_id})[1]["error"]["code"] == -32001


# ---------------------------------------------------------------------------
# HTTP+JSON vectors
# ---------------------------------------------------------------------------


def rest_request(client, caller: str, method: str, sub_path: str, headers: Dict[str, str], body: Any = None):
    return client.request(
        method,
        f"{BASE_PATH}{sub_path}",
        headers={**headers, "authorization": credential(caller)},
        content=json.dumps(body) if body is not None else None,
    )


class TestRestVectors:
    @pytest.mark.parametrize("case", VECTORS["rest"], ids=[c["name"] for c in VECTORS["rest"]])
    def test_vector(self, served, case):
        res = rest_request(served.client, case["caller"], case["method"], case["path"], case["headers"], case.get("body"))
        expect = case["expect"]
        assert res.status_code == expect["status"], res.text
        if "content_type" in expect:
            assert expect["content_type"] in res.headers["content-type"]
        body = res.json()
        if "error" in expect:
            assert body["error"]["code"] == expect["status"]
            assert body["error"]["status"] == expect["error"]["status"]
            assert body["error"]["details"][0]["reason"] == expect["error"]["reason"]
            return
        if "result_equals" in expect:
            assert body == expect["result_equals"]
            return
        task = body["task"]
        assert task["status"]["state"] in expect["state_in"]
        if "reply" in expect:
            assert reply_of(task) == expect["reply"]

    def test_flow_send_read_cancel_and_subscribe_over_rest(self, served):
        client = served.client
        ids: Dict[str, str] = {}
        for step in VECTORS["rest_flow"]["steps"]:
            if "send" in step:
                res = rest_request(client, step["caller"], "POST", "/a2a/message:send", {"content-type": "application/json"}, {"message": {"messageId": "rf", "role": "ROLE_USER", "parts": [{"text": step["send"]}]}})
                ids[step["as"]] = res.json()["task"]["id"]
            elif "get" in step:
                res = rest_request(client, step["caller"], "GET", f"/a2a/tasks/{ids[step['get']]}", {})
                assert res.status_code == step["expect_status"]
                if "expect_state" in step:
                    assert res.json()["status"]["state"] == step["expect_state"]
            elif "cancel" in step:
                res = rest_request(client, step["caller"], "POST", f"/a2a/tasks/{ids[step['cancel']]}:cancel", {})
                assert res.status_code == step["expect_status"]
                if "expect_reason" in step:
                    assert res.json()["error"]["details"][0]["reason"] == step["expect_reason"]
            elif "subscribe" in step:
                for method in ("POST", "GET"):
                    res = rest_request(client, step["caller"], method, f"/a2a/tasks/{ids[step['subscribe']]}:subscribe", {})
                    assert res.status_code == step["expect_status"], (method, res.text)
                    events = sse_events(res.text)
                    assert event_kind(events[0]) == step["expect_first_event"]

    def test_message_stream_streams_bare_stream_response_events(self, served):
        res = rest_request(served.client, "caller_a", "POST", "/a2a/message:stream", {"content-type": "application/json", "a2a-version": "1.0"}, VECTORS["streaming"]["rest_body"])
        assert res.status_code == 200
        events = sse_events(res.text)
        assert event_kind(events[0]) == "task"
        assert "jsonrpc" not in events[0]
        assert events[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"


# ---------------------------------------------------------------------------
# Dispatch through the servers
# ---------------------------------------------------------------------------


class TestRouteDispatch:
    def test_the_registry_holds_the_v1_routes_and_no_legacy_card(self):
        built = build_agent()
        routes = {(h["method"].upper(), h["subpath"]) for h in built.agent.get_all_http_handlers() if h["source"] != "builtin"}
        for expected in (
            ("POST", "/a2a"),
            ("POST", "/a2a/message:send"),
            ("POST", "/a2a/message:stream"),
            ("GET", "/a2a/tasks"),
            ("GET", "/a2a/tasks/{id}"),
            ("POST", "/a2a/tasks/{id}:cancel"),
            ("GET", "/a2a/tasks/{id}:subscribe"),
            ("POST", "/a2a/tasks/{id}:subscribe"),
            ("GET", "/.well-known/agent-card.json"),
        ):
            assert expected in routes, routes
        assert not any(sub in ("/.well-known/agent.json", "/tasks", "/tasks/{task_id}") for _m, sub in routes), routes

    def test_the_dynamic_door_serves_the_pattern_routes_and_the_card(self, tmp_path):
        """The catch-all route matches a skill's `{id}` and `:verb` subpaths
        with its own regex; the floor keeps anonymous callers off the task
        routes and the sends, and not off the card."""
        built = build_agent()

        def resolve(name, **kwargs):
            return built.agent

        server = WebAgentsServer(dynamic_agents=resolve, enable_monitoring=False, enable_prometheus=False, enable_rate_limiting=False, enable_request_logging=False, heartbeat=False, quiet=True)
        with TestClient(server.app, raise_server_exceptions=False) as client:
            send = client.post("/dyn/a2a/message:send", headers={"content-type": "application/json", "authorization": credential("caller_a")}, content=json.dumps({"message": {"messageId": "h-1", "role": "ROLE_USER", "parts": [{"text": "via the catch-all"}]}}))
            assert send.status_code == 200, send.text
            task = send.json()["task"]
            assert reply_of(task) == "echo: via the catch-all"
            get = client.get(f"/dyn/a2a/tasks/{task['id']}", headers={"authorization": credential("caller_a")})
            assert get.status_code == 200
            assert get.json()["id"] == task["id"]
            subscribe = client.get(f"/dyn/a2a/tasks/{task['id']}:subscribe", headers={"authorization": credential("caller_a")})
            assert subscribe.status_code == 200
            assert event_kind(sse_events(subscribe.text)[0]) == "task"
            assert client.get(f"/dyn/a2a/tasks/{task['id']}").status_code == 401
            assert client.post("/dyn/a2a/message:send", content="{}").status_code == 401
            card = client.get("/dyn/.well-known/agent-card.json")
            assert card.status_code == 200
            assert card.json()["supportedInterfaces"][0]["url"] == "http://testserver/dyn/a2a"

    def test_the_pre_v1_card_handler_is_gone_and_agent_json_is_the_registration_card(self, served):
        res = served.client.get(f"{BASE_PATH}/.well-known/agent.json")
        assert res.status_code == 200
        card = res.json()
        assert card["client_id"] == f"{ORIGIN}{BASE_PATH}/.well-known/agent.json"
        assert card["url"] == f"{ORIGIN}{BASE_PATH}"

    def test_a2a_is_nameable_in_the_agent_file_loader(self):
        from webagents.cli.agent_builder import SKILL_CLASSES, load_skills

        assert SKILL_CLASSES["a2a"].endswith("A2ATransportSkill")
        loaded = load_skills([{"a2a": {"peers": {"https://peer.example": {"token": "t"}}}}], "x")
        assert isinstance(loaded["a2a"], A2ATransportSkill)
        assert loaded["a2a"].settings["peers"] == {"https://peer.example": {"token": "t"}}


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


class TestAgentCard:
    def test_serves_the_fixture_card_beside_agent_json_signed_by_the_served_identity(self, served):
        client = served.client
        card_path = f"{BASE_PATH}{VECTORS['card']['well_known_path']}"
        res = client.get(card_path)
        assert res.status_code == 200
        assert VECTORS["card"]["content_type"] in res.headers["content-type"]
        assert res.headers["cache-control"] == VECTORS["card"]["cache_control"]
        assert res.headers["etag"]
        card = res.json()
        e = VECTORS["card"]["expect"]
        assert card["name"] == e["name"]
        assert card["description"] == e["description"]
        assert card["supportedInterfaces"] == e["supportedInterfaces"]
        assert card["version"] == e["version"]
        assert card["capabilities"] == e["capabilities"]
        assert card["securitySchemes"] == e["securitySchemes"]
        assert card["securityRequirements"] == e["securityRequirements"]
        assert card["defaultInputModes"] == e["defaultInputModes"]
        assert card["defaultOutputModes"] == e["defaultOutputModes"]
        ids = [s["id"] for s in card["skills"]]
        for skill_id in e["skill_ids_include"]:
            assert skill_id in ids
        for skill_id in e["skill_ids_exclude"]:
            assert skill_id not in ids
        for skill in card["skills"]:
            assert skill["tags"] == e["skill_tags"]
        assert client.get(card_path, headers={"if-none-match": res.headers["etag"]}).status_code == 304

        # Signed with the identity the server minted, verifiable against the key set it serves.
        assert len(card["signatures"]) == 1
        header = json.loads(b64url_decode(card["signatures"][0]["protected"]).decode("utf-8"))
        jwks = client.get(f"{BASE_PATH}/.well-known/jwks.json").json()
        kid = header["kid"]
        assert header == {"alg": "EdDSA", "jku": f"{ORIGIN}{BASE_PATH}/.well-known/jwks.json", "kid": kid, "typ": "JOSE"}
        assert any(k["kid"] == kid for k in jwks["keys"])
        verified = asyncio.run(verify_agent_card(card, keys=jwks["keys"]))
        assert (verified.ok, verified.kid, verified.alg, verified.checked) == (True, kid, "EdDSA", 1)

        # Through the jku, fetched only from the card's own origin.
        async def fetch_json(url: str):
            assert url == f"{ORIGIN}{BASE_PATH}/.well-known/jwks.json"
            return jwks

        via_jku = asyncio.run(verify_agent_card(card, card_url=f"{ORIGIN}{card_path}", fetch_json=fetch_json))
        assert via_jku.ok
        tampered = {**card, "description": "edited after signing"}
        assert asyncio.run(verify_agent_card(tampered, keys=jwks["keys"])).ok is False
        wrong = SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())
        assert asyncio.run(verify_agent_card(card, keys=[wrong.public_jwk()])).ok is False

        async def must_not_fetch(url: str):
            raise AssertionError("must not fetch")

        foreign = {**card, "signatures": [{**card["signatures"][0], "protected": b64url(json.dumps({"alg": "EdDSA", "kid": kid, "typ": "JOSE", "jku": "https://evil.example/jwks.json"}).encode())}]}
        refused = asyncio.run(verify_agent_card(foreign, card_url=f"{ORIGIN}/x", fetch_json=must_not_fetch))
        assert refused.ok is False
        assert "evil.example" in refused.reason

    def test_unsigned_without_an_identity_and_the_interface_url_falls_back_to_the_request_origin(self, monkeypatch):
        monkeypatch.delenv("WEBAGENTS_PUBLIC_URL", raising=False)
        built = build_agent(skill_config={"public_url": ORIGIN})
        request = SimpleNamespace(url=SimpleNamespace(scheme="http", netloc="127.0.0.1:4242", path=f"{BASE_PATH}/.well-known/agent-card.json"), headers={})
        card = asyncio.run(built.skill.build_card(built.agent, request))
        assert "signatures" not in card
        assert card["supportedInterfaces"][0]["url"] == f"{ORIGIN}{BASE_PATH}/a2a"
        bare = build_agent(name="mini")
        request = SimpleNamespace(url=SimpleNamespace(scheme="http", netloc="127.0.0.1:4242", path="/agents/mini/.well-known/agent-card.json"), headers={})
        assert bare.skill.principal_for(bare.agent, request) == "http://127.0.0.1:4242/agents/mini"


# ---------------------------------------------------------------------------
# Signature and canonicalisation vectors
# ---------------------------------------------------------------------------


class TestSignatureVectors:
    def test_reproduces_the_specs_hs256_byte_assembly(self):
        v = VECTORS["signatures"]["hs256_vector"]
        assert canonicalize(card_signing_payload(v["card"])) == v["canonical"]
        assert hashlib.sha256(card_signing_bytes(v["card"])).hexdigest() == v["sha256"]
        secret = v["key"].encode("utf-8")
        signed = sign_agent_card(v["card"], hmac_card_signer(v["protected"]["kid"], secret))
        assert signed["signatures"][0]["protected"] == v["protected_b64"]
        assert signed["signatures"][0]["signature"] == v["signature"]
        assert asyncio.run(verify_agent_card(signed, secret=secret)).ok
        assert asyncio.run(verify_agent_card(signed, secret=b"other")).ok is False
        # HS256 is never accepted without a secret.
        assert asyncio.run(verify_agent_card(signed)).ok is False

    def test_round_trips_eddsa_with_kid_and_jku(self):
        v = VECTORS["signatures"]["hs256_vector"]
        key = SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())
        signer = ed25519_card_signer(key.thumbprint, key.private_key, jku=VECTORS["signatures"]["eddsa_round_trip"]["jku"])
        signed = sign_agent_card(v["card"], signer)
        header = json.loads(b64url_decode(signed["signatures"][0]["protected"]).decode("utf-8"))
        assert header["jku"] == VECTORS["signatures"]["eddsa_round_trip"]["jku"]
        assert header["kid"] == key.thumbprint
        assert asyncio.run(verify_agent_card(signed, keys=[key.public_jwk()])).ok
        assert asyncio.run(verify_agent_card({**signed, "name": "changed"}, keys=[key.public_jwk()])).ok is False
        assert asyncio.run(verify_agent_card(signed, keys=[])).ok is False
        # Two signatures: one bad, one good, is a verified card.
        twice = {**signed, "signatures": [{"protected": signed["signatures"][0]["protected"], "signature": b64url(bytes(64))}, *signed["signatures"]]}
        result = asyncio.run(verify_agent_card(twice, keys=[key.public_jwk()]))
        assert (result.checked, result.ok) == (2, True)

    @pytest.mark.parametrize("case", VECTORS["signatures"]["signing_payload_cases"], ids=[c["name"] for c in VECTORS["signatures"]["signing_payload_cases"]])
    def test_signing_payload(self, case):
        assert canonicalize(card_signing_payload(case["card"])) == case["canonical"]


class TestJcsVectors:
    @pytest.mark.parametrize("case", VECTORS["jcs"]["cases"], ids=[c["name"] for c in VECTORS["jcs"]["cases"]])
    def test_vector(self, case):
        assert canonicalize(case["value"]) == case["canonical"]

    def test_refuses_what_json_cannot_carry(self):
        with pytest.raises(ValueError, match="nan"):
            canonicalize({"a": float("nan")})
        with pytest.raises(ValueError, match="inf"):
            canonicalize({"a": float("inf")})


# ---------------------------------------------------------------------------
# Configuration (shared)
# ---------------------------------------------------------------------------


class TestAgentFileEntry:
    @pytest.mark.parametrize("shape", CONFIG_SHAPES["shapes"], ids=[s["name"] for s in CONFIG_SHAPES["shapes"]])
    def test_shape(self, shape):
        assert A2ATransportSkill(shape["config"]).settings == shape["settings"]
        assert resolve_a2a_settings(shape["config"]) == shape["settings"]

    def test_peer_tokens_the_longest_configured_prefix_wins_nothing_else_matches(self):
        for case in CONFIG_SHAPES["peer_token"]["cases"]:
            assert peer_token_for(case["url"], CONFIG_SHAPES["peer_token"]["peers"]) == case["token"], case

    def test_camel_case_spellings_resolve_to_the_same_settings(self):
        settings = resolve_a2a_settings({"taskTtlSeconds": 5, "blockingTimeoutSeconds": 2, "documentationUrl": "https://d", "iconUrl": "https://i"})
        assert settings["task_ttl_seconds"] == 5
        assert settings["blocking_timeout_seconds"] == 2
        assert settings["documentation_url"] == "https://d"
        assert settings["icon_url"] == "https://i"


# ---------------------------------------------------------------------------
# The client (shared vectors)
# ---------------------------------------------------------------------------


class TestA2AClient:
    @pytest.mark.parametrize("case", VECTORS["client"]["interface_cases"], ids=[c["name"] for c in VECTORS["client"]["interface_cases"]])
    def test_interface(self, case):
        assert pick_interface(case["card"]) == case["expect"]

    @pytest.mark.parametrize("case", VECTORS["client"]["result_cases"], ids=[c["name"] for c in VECTORS["client"]["result_cases"]])
    def test_result(self, case):
        result = read_send_result(case["result"])
        assert case["expect"]["kind"] in result
        assert reply_of_result(result) == case["expect"]["reply"]

    def test_discovers_the_card_sends_with_the_version_header_and_the_peer_bearer_and_reads_the_task(self, tmp_path):
        peer = f"{ORIGIN}{BASE_PATH}"
        built = build_server(tmp_path, skill_config={"peers": {peer: {"token": "caller-a-token"}}})
        sent: List[httpx.Request] = []

        async def record(request: httpx.Request) -> None:
            sent.append(request)

        async def run():
            transport = httpx.ASGITransport(app=built.server.app)
            async with httpx.AsyncClient(transport=transport, event_hooks={"request": [record]}) as client:
                result = await built.skill.call_peer(peer, "hello peer", client=client)
                # The task is the caller's on the peer: readable with that bearer, not another.
                rpc_body = lambda token: {"jsonrpc": "2.0", "id": 1, "method": "GetTask", "params": {"id": result["task"]["id"]}}
                mine = await client.post(f"{peer}/a2a", headers={"authorization": "Bearer caller-a-token"}, json=rpc_body("a"))
                other = await client.post(f"{peer}/a2a", headers={"authorization": "Bearer caller-b-token"}, json=rpc_body("b"))
                return result, mine.json(), other.json()

        result, mine, other = asyncio.run(run())
        assert result["reply"] == "echo: hello peer"
        assert result["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        assert result["card_url"] == f"{peer}{VECTORS['client']['card_paths'][0]}"
        assert result["rpc_url"] == f"{peer}/a2a"
        assert (sent[0].method, str(sent[0].url)) == ("GET", f"{peer}{VECTORS['client']['card_paths'][0]}")
        send = sent[1]
        assert str(send.url) == f"{peer}/a2a"
        assert send.headers["a2a-version"] == VECTORS["client"]["version_header"]["A2A-Version"]
        assert send.headers["authorization"] == "Bearer caller-a-token"
        body = json.loads(send.content)
        assert body["method"] == "SendMessage"
        assert body["params"]["message"]["role"] == "ROLE_USER"
        assert body["params"]["message"]["parts"] == [{"text": "hello peer", "mediaType": "text/plain"}]
        assert mine["result"]["task"]["id"] == result["task"]["id"]
        assert other["error"]["code"] == -32001

    def test_retries_message_send_on_32601_falls_back_to_agent_json_refuses_a_v03_card_and_a_redirect(self):
        card = {
            "name": "stub",
            "description": "",
            "supportedInterfaces": [{"url": "https://stub.example/rpc", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
            "version": "1.0.0",
            "capabilities": {},
            "defaultInputModes": [],
            "defaultOutputModes": [],
            "skills": [],
        }
        methods: List[str] = []
        paths = VECTORS["client"]["card_paths"]

        def stub(request: httpx.Request) -> httpx.Response:
            if request.url.path == paths[0]:
                return httpx.Response(404, text="nope")
            if request.url.path == paths[1]:
                return httpx.Response(200, json=card)
            body = json.loads(request.content)
            methods.append(body["method"])
            if body["method"] == "SendMessage":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "error": {"code": VECTORS["client"]["retry_on"], "message": "Method not found"}})
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"id": "t-legacy", "contextId": "c", "status": {"state": "TASK_STATE_COMPLETED"}, "artifacts": [{"artifactId": "a", "parts": [{"text": "from a v0.3 server"}]}], "history": []},
                },
            )

        async def run(handler, base):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await call_agent(base, "hi", client=client)

        result = asyncio.run(run(stub, "https://stub.example"))
        assert result["card_url"] == "https://stub.example" + paths[1]
        assert methods == ["SendMessage", VECTORS["client"]["retry_method"]]
        assert result["reply"] == "from a v0.3 server"

        def legacy(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"url": "https://old.example/", "preferredTransport": "JSONRPC", "protocolVersion": "0.3"})

        with pytest.raises(A2AClientError, match="no A2A 1.0 JSON-RPC interface"):
            asyncio.run(run(legacy, "https://old.example"))

        def redirecting(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "https://elsewhere.example/"})

        with pytest.raises(A2AClientError, match="redirect"):
            asyncio.run(run(redirecting, "https://moved.example"))


class TestPartParsing:
    def test_reads_v1_and_v03_spellings_to_one_shape(self):
        assert read_part({"text": "a", "mediaType": "text/plain"}, 0) == {"text": "a", "mediaType": "text/plain"}
        assert read_part({"kind": "text", "text": "a"}, 0) == {"text": "a"}
        assert read_part({"kind": "file", "file": {"uri": "https://x/y.png", "mimeType": "image/png", "name": "y.png"}}, 0) == {"url": "https://x/y.png", "mediaType": "image/png", "filename": "y.png"}
        assert read_part({"kind": "file", "file": {"bytes": "AAAA", "mimeType": "application/pdf"}}, 0) == {"raw": "AAAA", "mediaType": "application/pdf"}
        assert read_part({"kind": "data", "data": {"k": 1}}, 0) == {"data": {"k": 1}}
        with pytest.raises(Exception, match="none of text, raw, url or data"):
            read_part({"kind": "text"}, 0)
