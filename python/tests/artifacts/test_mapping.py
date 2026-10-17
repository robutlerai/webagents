"""A set in the protocols an agent speaks: A2A parts and MCP resource links."""

from __future__ import annotations

from webagents.artifacts import (
    ARTIFACT_SET_MEDIA_TYPE,
    artifact_set_from_a2a_parts,
    artifact_set_to_a2a_parts,
    artifact_set_to_mcp_resource_links,
    parse_artifact_set,
)

BASE = "https://agent.example/agents/flow"
H = lambda n: f"{n:02x}" * 32  # noqa: E731
SET = parse_artifact_set({
    "kind": "webagents.artifact-set/v1",
    "createdAt": "2026-10-17T10:00:00Z",
    "expiresAt": None,
    "producer": {"agent": BASE, "task": "t1"},
    "entries": [
        {"path": "gds/top.gds", "size": 10, "sha256": H(1), "mediaType": "application/octet-stream", "locations": [{"kind": "https", "url": f"{BASE}/artifacts/blobs/sha256:{H(1)}"}]},
        {"path": "report.txt", "size": 3, "sha256": H(2), "mediaType": "text/plain"},
    ],
})
SHA = H(9)


def test_a2a_parts():
    parts = artifact_set_to_a2a_parts(SET, SHA, base_url=BASE, manifest_url=f"{BASE}/artifacts/blobs/sha256:{SHA}")
    assert parts[0]["data"]["kind"] == "webagents.artifact-set/v1"
    assert parts[0]["mediaType"] == ARTIFACT_SET_MEDIA_TYPE
    assert parts[0]["metadata"] == {"sha256": SHA, "url": f"{BASE}/artifacts/blobs/sha256:{SHA}"}
    assert parts[1] == {"url": f"{BASE}/artifacts/blobs/sha256:{H(1)}", "mediaType": "application/octet-stream", "filename": "gds/top.gds", "metadata": {"sha256": H(1), "size": 10}}
    assert parts[2]["url"] == f"{BASE}/artifacts/blobs/sha256:{H(2)}"
    assert artifact_set_from_a2a_parts(parts)["sha256"] == SHA
    assert artifact_set_from_a2a_parts([{"text": "hi"}]) is None


def test_mcp_resource_links():
    links = artifact_set_to_mcp_resource_links(SET, SHA, base_url=BASE)
    assert links[0]["type"] == "resource_link" and links[0]["mimeType"] == ARTIFACT_SET_MEDIA_TYPE
    assert links[1] == {"type": "resource_link", "uri": f"{BASE}/artifacts/blobs/sha256:{H(1)}", "name": "gds/top.gds", "mimeType": "application/octet-stream", "size": 10}
    assert len(links) == 3
    bare = parse_artifact_set({**SET.to_json(), "entries": [SET.to_json()["entries"][1]]})
    assert artifact_set_to_mcp_resource_links(bare, SHA) == []
