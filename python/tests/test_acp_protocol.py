"""
The ACP agent, in process (gap-closure plan item 1.6, 2026-09-26): what the
fixture `tests/fixtures/acp/acp_protocol.json` pins, the same in both SDKs
(TypeScript: tests/unit/transport/acp-protocol.test.ts): the constants,
the tool kinds, the transport entries an agent file may write, and the
JSON-RPC surface driven over an in-memory line pair. The S-269 pin lives
here too: the skill mounts no HTTP or WebSocket route, and the client's own
`fs/*` and `terminal/*` methods are not served over stdio either.
The spawned-CLI transcripts are `tests/test_acp_stdio.py`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.transport.acp import protocol as P
from webagents.agents.skills.core.transport.acp.skill import ACPTransportSkill, AcpSession, SessionStore
from webagents.cli.agent_builder import SKILL_CLASSES, load_skills

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROTOCOL = json.loads((FIXTURES / "acp" / "acp_protocol.json").read_text())


# -- the fixture's constants -----------------------------------------------------------------


def test_the_constants_are_the_fixture_s():
    assert P.PROTOCOL_VERSION == PROTOCOL["protocol_version"]
    assert P.AGENT_CAPABILITIES == PROTOCOL["agent_capabilities"]
    assert P.AUTH_METHODS == PROTOCOL["auth_methods"]
    assert P.AGENT_INFO_NAME == PROTOCOL["agent_info_name"]
    assert P.SESSION_ID_PREFIX == PROTOCOL["session_id_prefix"]
    errors = PROTOCOL["errors"]
    assert (P.PARSE_ERROR, P.INVALID_REQUEST, P.METHOD_NOT_FOUND, P.INVALID_PARAMS) == (
        errors["parse"], errors["invalid_request"], errors["method_not_found"], errors["invalid_params"])
    assert (P.INTERNAL_ERROR, P.AUTH_REQUIRED, P.RESOURCE_NOT_FOUND, P.REQUEST_CANCELLED) == (
        errors["internal"], errors["auth_required"], errors["resource_not_found"], errors["request_cancelled"])
    assert set(P.PERMISSION_KINDS) == set(PROTOCOL["permission"]["kinds"])
    assert P.PERMISSION_OPTIONS == PROTOCOL["permission"]["options"]
    assert P.REJECTED == PROTOCOL["permission"]["rejected"]
    assert P.CANCELLED == PROTOCOL["permission"]["cancelled"]
    assert (P.STOP_END_TURN, P.STOP_CANCELLED) == (PROTOCOL["stop_reasons"]["end_turn"], PROTOCOL["stop_reasons"]["cancelled"])


@pytest.mark.parametrize("case", PROTOCOL["tool_kinds"]["cases"], ids=lambda c: c["name"] or "empty")
def test_tool_kinds(case):
    assert P.tool_kind(case["name"]) == case["kind"]


def test_permission_follows_the_kind():
    for case in PROTOCOL["tool_kinds"]["cases"]:
        assert P.needs_permission(case["kind"]) == (case["kind"] in PROTOCOL["permission"]["kinds"])


def test_the_command_words_are_the_fixture_s():
    import click
    import typer

    from webagents.cli.help_format import _description
    from webagents.cli.main import COMMAND_ORDER, app

    words = PROTOCOL["cli"]
    root = typer.main.get_command(app)
    command = root.commands[words["command"]]
    assert _description(command) == words["description"]
    arguments = [[f"[{p.name}]" if not p.required else f"<{p.name}>", p.help or "", str(p.default)] for p in command.params if isinstance(p, click.Argument)]
    assert arguments == words["arguments"]
    assert [p for p in command.params if isinstance(p, click.Option) and not (set(p.opts) & {"-h", "--help"})] == []
    # Listed with the serving commands, after `mcp`; the exact order across
    # the two CLIs is `test_cli_parity.py`'s to hold.
    assert COMMAND_ORDER.index("acp") > COMMAND_ORDER.index("mcp")


# -- the transport entries an agent file may write --------------------------------------------


@pytest.mark.parametrize("name", ["acp", "completions", "realtime"])
def test_transport_config_shapes(name, tmp_path):
    shapes = PROTOCOL["config_shapes"][name]
    for shape in shapes["shapes"]:
        loaded = load_skills([{name: dict(shape["config"])}], agent_name="fixture", agent_path=tmp_path / "AGENT.md")
        assert name in loaded, shape["name"]
        assert loaded[name].settings == shape["settings"], shape["name"]
    bare = load_skills([name], agent_name="fixture", agent_path=tmp_path / "AGENT.md")
    assert bare[name].settings == shapes["defaults"]
    assert SKILL_CLASSES[name].endswith("TransportSkill")


# -- the skill mounts nothing (S-269) ---------------------------------------------------------


def test_the_skill_mounts_no_route_s269():
    agent = BaseAgent(name="t", instructions="", skills={"acp": ACPTransportSkill()})
    assert [h for h in agent.get_all_http_handlers() if h.get("source") in ("acp", "ACPTransportSkill")] == []
    assert [h for h in agent.get_all_websocket_handlers() if h.get("source") in ("acp", "ACPTransportSkill")] == []
    assert not hasattr(ACPTransportSkill, "acp_http") and not hasattr(ACPTransportSkill, "acp_websocket")
    assert not any(name.startswith("_handle_fs") or name.startswith("_handle_terminal") for name in dir(ACPTransportSkill))


# -- the JSON-RPC surface, driven in process --------------------------------------------------


async def drive(messages: List[Any], *, skill: ACPTransportSkill = None, agent: BaseAgent = None) -> List[Dict[str, Any]]:
    """Serve `agent` over an in-memory line pair: every `messages` entry in,
    every line out, parsed."""
    skill = skill or ACPTransportSkill()
    agent = agent or BaseAgent(name="t", instructions="", skills={"acp": skill})
    out: List[str] = []

    async def lines():
        for message in messages:
            yield (message if isinstance(message, str) else json.dumps(message)).encode() + b"\n"
            # Let the request's task answer before the next line, as a client would wait.
            for _ in range(20):
                await asyncio.sleep(0)

    await skill.serve(agent, lines(), out.append)
    return [json.loads(line) for line in out]


async def test_initialize_negotiates_and_advertises_the_fixture(capsys):
    replies = await drive([{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 99, "clientCapabilities": {"terminal": True}}}])
    assert replies == [{
        "jsonrpc": "2.0", "id": 0,
        "result": {
            "protocolVersion": 1,
            "agentCapabilities": PROTOCOL["agent_capabilities"],
            "agentInfo": {"name": "webagents", "title": "t", "version": replies[0]["result"]["agentInfo"]["version"]},
            "authMethods": PROTOCOL["auth_methods"],
        },
    }]
    assert f"[webagents] t: ACP over stdio" in capsys.readouterr().err


async def test_requests_and_notifications_are_told_apart_by_id():
    replies = await drive([
        {"jsonrpc": "2.0", "id": 0, "method": "no/such"},
        {"jsonrpc": "2.0", "method": "no/such"},
        {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": "sess_none"}},
        {"jsonrpc": "2.0", "id": 0, "result": {}},
        "{not json",
        [],
        {"jsonrpc": "2.0", "id": 7},
    ])
    assert replies == [
        {"jsonrpc": "2.0", "id": 0, "error": {"code": -32601, "message": "Method not found: no/such"}},
        {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
        {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}},
        {"jsonrpc": "2.0", "id": 7, "error": {"code": -32600, "message": "Invalid request"}},
    ]


@pytest.mark.parametrize("method", PROTOCOL["client_methods_not_served"])
async def test_client_methods_are_not_served_s269(method, tmp_path):
    marker = tmp_path / "pwned"
    params = {"sessionId": "x", "path": str(marker), "content": "x", "command": "touch", "args": [str(marker)], "cwd": str(tmp_path)}
    replies = await drive([{"jsonrpc": "2.0", "id": 1, "method": method, "params": params}])
    assert replies == [{"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": f"Method not found: {method}"}}]
    assert not marker.exists()


async def test_session_new_validates_and_persists(tmp_path):
    skill = ACPTransportSkill({"sessions_dir": str(tmp_path / "sessions")})
    replies = await drive([
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": "relative", "mcpServers": []}},
        {"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"cwd": str(tmp_path)}},
        {"jsonrpc": "2.0", "id": 3, "method": "session/new", "params": {"cwd": str(tmp_path), "mcpServers": []}},
        {"jsonrpc": "2.0", "id": 4, "method": "session/list", "params": {}},
        {"jsonrpc": "2.0", "id": 5, "method": "session/prompt", "params": {"sessionId": "sess_missing", "prompt": [{"type": "text", "text": "hi"}]}},
        {"jsonrpc": "2.0", "id": 6, "method": "session/load", "params": {"sessionId": "../../etc/passwd", "cwd": str(tmp_path), "mcpServers": []}},
        {"jsonrpc": "2.0", "id": 7, "method": "authenticate", "params": {"methodId": "login"}},
    ], skill=skill)
    assert replies[0]["error"]["code"] == -32602 and replies[1]["error"]["code"] == -32602
    session_id = replies[2]["result"]["sessionId"]
    assert session_id.startswith("sess_")
    assert (tmp_path / "sessions" / f"{session_id}.json").is_file()
    assert replies[3]["result"]["sessions"][0]["sessionId"] == session_id
    assert replies[3]["result"]["sessions"][0]["cwd"] == str(tmp_path)
    assert replies[4]["error"]["code"] == -32002
    assert replies[5]["error"]["code"] == -32602
    assert replies[6] == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_the_session_store_never_joins_an_arbitrary_id_into_a_path(tmp_path):
    store = SessionStore(tmp_path)
    assert store.path_of("../../etc/passwd") is None
    assert store.path_of("sess_ok.1-2") == tmp_path / "sess_ok.1-2.json"
    assert store.load("../x") is None
    session = AcpSession(session_id="sess_a", cwd="/p", agent="t", created_at="2026-09-26T00:00:00Z", updated_at="2026-09-26T00:00:01Z", title="hi",
                         messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
    store.save(session)
    loaded = store.load("sess_a")
    assert loaded is not None and loaded.to_dict() == session.to_dict()
    assert [s.session_id for s in store.list_all()] == ["sess_a"]


# -- the mappings ----------------------------------------------------------------------------


def test_prompt_text_and_the_blocks_it_refuses():
    assert P.prompt_text([{"type": "text", "text": "hi"}]) == "hi"
    assert P.prompt_text([
        {"type": "text", "text": "see"},
        {"type": "resource", "resource": {"uri": "file:///a.txt", "text": "A"}},
        {"type": "resource_link", "uri": "file:///b.txt"},
    ]) == "see\n\n[Resource: file:///a.txt]\nA\n\n[Resource link: file:///b.txt]"
    for bad in ([], "hi", [{"type": "image", "data": "", "mimeType": "image/png"}], [{"type": "nope"}], [1]):
        with pytest.raises(P.AcpError) as refused:
            P.prompt_text(bad)
        assert refused.value.code == -32602


def test_mcp_servers_config_takes_both_entry_shapes():
    servers = P.mcp_servers_config([
        {"name": "demo", "command": "python", "args": ["server.py"], "env": [{"name": "A", "value": "1"}]},
        {"type": "http", "name": "remote", "url": "https://x.example/mcp", "headers": [{"name": "Authorization", "value": "Bearer t"}]},
        {"type": "sse", "name": "old", "url": "https://x.example/sse"},
        {"name": "broken"},
        "not an object",
    ])
    assert servers == {
        "demo": {"command": "python", "args": ["server.py"], "env": {"A": "1"}},
        "remote": {"url": "https://x.example/mcp", "headers": {"Authorization": "Bearer t"}, "transport": "http"},
        "old": {"url": "https://x.example/sse", "headers": {}, "transport": "sse"},
    }


def test_plan_entries_from_a_todo_list():
    assert P.plan_entries([
        {"content": "a", "priority": "critical", "status": "in_progress"},
        {"content": "b", "priority": "low", "status": "cancelled"},
        {"content": "c"},
    ]) == [{"content": "a", "priority": "high", "status": "in_progress"}, {"content": "c", "priority": "medium", "status": "pending"}]
    assert P.plan_entries(None) is None


def test_parse_arguments_is_forgiving():
    assert P.parse_arguments('{"a": 1}') == {"a": 1}
    assert P.parse_arguments("not json") == {}
    assert P.parse_arguments("[1]") == {}
    assert P.parse_arguments({"b": 2}) == {"b": 2}
    assert P.parse_arguments(None) == {}
