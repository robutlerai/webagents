"""
S-298 (2026-09-26): a caller principal with a comma is not a namespace, and
namespace lists travel to the portal as repeated parameters, never joined.
The twin of the TypeScript ``memory-namespace-comma-w2fix.test.ts``; both read
``namespace_grammar`` of the shared fixture ``memory_tool/definition.json``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from webagents.agents.skills.local.memory.memory_namespace import NAMESPACE_RE, namespace_of, readable_namespaces
from webagents.agents.skills.local.memory.portal_memory_store import PortalMemoryStore

from fake_portal_w2mem import FakePortal

FIXTURE = json.loads((Path(__file__).parents[2] / "fixtures" / "memory_tool" / "definition.json").read_text())
G = FIXTURE["namespace_grammar"]


def run(coro):
    return asyncio.run(coro)


def test_the_grammar_accepts_the_valid_shapes_and_refuses_the_comma_probe():
    for ns in G["valid"]:
        assert NAMESPACE_RE.match(ns), ns
    for ns in G["invalid"]:
        assert not NAMESPACE_RE.match(ns), ns


def test_the_probe_principal_gets_no_namespace_and_reads_shared_alone():
    auth = SimpleNamespace(authenticated=True, provider="platform", scope="user", user_id=None, principals=[G["probe"]["principal"]])
    assert namespace_of(auth) == G["probe"]["namespace_of"]
    assert readable_namespaces(namespace_of(auth)) == ["shared"]
    clean = SimpleNamespace(authenticated=True, provider="platform", scope="user", user_id=None, principals=[G["probe"]["principal"].replace(",owner", "")])
    assert namespace_of(clean) == "caller:" + G["probe"]["principal"].replace(",owner", "")


def _store(portal: FakePortal, transport=None) -> PortalMemoryStore:
    return PortalMemoryStore("https://portal.test", lambda: "agent-key", agent_id=portal.agent_id, transport=transport or portal.transport)


def test_the_store_sends_repeated_parameters_never_a_joined_list():
    portal = FakePortal()
    s = _store(portal)
    run(s.list(["caller:user:alice", "shared"], exclude_sources=["compaction", "sync"]))
    run(s.search("short", ["caller:user:alice", "shared"], 5))
    listed, searched = portal.seen[0], portal.seen[1]
    assert listed.url.params.get_list(G["list_wire"]["parameter"]) == ["caller:user:alice", "shared"]
    assert listed.url.params.get_list(G["list_wire"]["exclude_parameter"]) == ["compaction", "sync"]
    assert searched.url.params.get_list(G["list_wire"]["parameter"]) == ["caller:user:alice", "shared"]
    for retired in G["list_wire"]["retired"]:
        assert retired not in listed.url.params and retired not in searched.url.params
    run(s.list(None))
    assert G["list_wire"]["parameter"] not in portal.seen[2].url.params


def test_the_store_post_filters_to_the_namespaces_it_asked_for():
    portal = FakePortal()
    portal.put("caller:user:alice", "preferences", "short", "tool")

    def leaky(request: httpx.Request) -> httpx.Response:
        response = portal._handle(request)
        body = json.loads(response.content)
        if "entries" in body:
            body["entries"].append({"id": "leak", "namespace": "owner", "key": "plan", "content": "October", "source": "owner", "created_at": "x", "updated_at": "x"})
        return httpx.Response(response.status_code, json=body)

    s = _store(portal, transport=httpx.MockTransport(leaky))
    listed = run(s.list(["caller:user:alice", "shared"]))
    assert [f"{e.namespace}/{e.key}" for e in listed] == ["caller:user:alice/preferences"]
    found = run(s.search("short", ["caller:user:alice", "shared"]))
    assert [e.namespace for e in found] == ["caller:user:alice"]
    assert any(e.namespace == "owner" for e in run(s.list(None)))
