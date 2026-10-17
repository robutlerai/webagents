"""The shared artifact-set vectors: canonical bytes, identities and refusal
codes, the same file the TypeScript SDK runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from webagents.artifacts import (
    LIMITS,
    ArtifactSetError,
    canonical_manifest,
    chunk_count,
    chunk_range,
    digest_name,
    manifest_hash,
    parse_artifact_set,
    parse_digest_name,
    read_artifact_set,
)

VECTORS = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "artifact_set" / "vectors.json").read_text("utf-8"))


def _generated(spec: dict):
    if spec["kind"] == "entries":
        return {
            "kind": "webagents.artifact-set/v1",
            "createdAt": "2026-10-17T10:00:00Z",
            "expiresAt": None,
            "producer": {"agent": "https://agent.example/agents/flow"},
            "entries": [{"path": f"f/{i}", "size": 1, "sha256": f"{i:08x}" * 8, "mediaType": "text/plain"} for i in range(spec["count"])],
        }
    if spec["kind"] == "bytes":
        return b" " * spec["count"]
    if spec["kind"] == "text":
        return spec["text"]
    raise AssertionError(spec)


@pytest.mark.parametrize("case", VECTORS["valid"], ids=[c["name"] for c in VECTORS["valid"]])
def test_valid_vectors(case):
    artifact_set = parse_artifact_set(case["manifest"])
    assert canonical_manifest(artifact_set) == case["canonical"]
    assert manifest_hash(artifact_set) == case["sha256"]
    again, sha = read_artifact_set(case["canonical"].encode("utf-8"))
    assert sha == case["sha256"]
    assert canonical_manifest(again) == case["canonical"]


@pytest.mark.parametrize("case", VECTORS["hostile"], ids=[c["name"] for c in VECTORS["hostile"]])
def test_hostile_vectors(case):
    data = _generated(case["generate"]) if "generate" in case else case["manifest"]
    with pytest.raises(ArtifactSetError) as caught:
        parse_artifact_set(data, depth=case.get("depth", 0))
    assert caught.value.code == case["code"]


def test_chunk_arithmetic_and_digest_names():
    assert chunk_count(0, 262144) == 0
    assert chunk_count(262144, 262144) == 1
    assert chunk_count(262145, 262144) == 2
    assert chunk_range(2, 600000, 262144) == (524288, 599999)
    sha = "ab" * 32
    assert digest_name(sha) == f"sha256:{sha}"
    assert parse_digest_name(f"sha256:{sha}") == sha
    assert parse_digest_name(f"sha256:{sha.upper()}") is None
    assert parse_digest_name(sha) is None
    assert LIMITS["max_depth"] == 8
