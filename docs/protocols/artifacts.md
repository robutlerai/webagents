---
title: Artifacts
description: Large files by reference between agents and with a platform - the artifact-set manifest, the producer endpoint, and the relay frames over an agent's outbound connection.
---

# Artifacts: large files by reference

Version: `webagents.artifact-set/v1`, relay frames version 1.

An agent that produces large files (a run's outputs, a folder of reports) does not send their bytes in a message. It publishes a **manifest** that names every file by the SHA-256 of its bytes and says where the bytes can be fetched. A reader fetches what it needs, by range, and verifies every byte against the manifest. The platform is one place bytes can come from, never a requirement: two agents exchange a manifest and its files with no platform at all.

The SDKs implement this in `webagents/artifacts` (TypeScript) and `webagents.artifacts` (Python): the manifest and its strict reader, streaming hashing, `files.publish(folder)`, the producer endpoint, the client, the A2A and MCP mapping, and the relay an agent serves over its outbound connection.

## 1. The manifest

A manifest is a JSON object:

```json
{
  "kind": "webagents.artifact-set/v1",
  "createdAt": "2026-10-17T10:00:00Z",
  "expiresAt": null,
  "producer": { "agent": "https://agent.example/agents/flow", "task": "task-42" },
  "entries": [
    {
      "path": "gds/top.gds",
      "size": 314572800,
      "sha256": "9f0e…",
      "mediaType": "application/octet-stream",
      "chunks": { "chunkSize": 8388608, "hashes": ["…", "…"] },
      "locations": [
        { "kind": "https", "url": "https://agent.example/agents/flow/artifacts/blobs/sha256:9f0e…" },
        { "kind": "relay", "agent": "flow" }
      ]
    },
    { "path": "reports/drc.txt", "size": 1200, "sha256": "ab12…", "mediaType": "text/plain" }
  ],
  "locations": [ { "kind": "robutler", "url": "https://portal.example/api/blob/cas/9f0e…" } ]
}
```

| Field | Meaning |
| --- | --- |
| `kind` | Always `webagents.artifact-set/v1`. |
| `createdAt`, `expiresAt` | ISO 8601 UTC with `Z`; `expiresAt` may be `null`. After `expiresAt` the producer may stop serving the set. |
| `producer.agent` | The producing agent's principal, its agent URL. `producer.task` (optional) names the task the set answers; `producer.name` is a display name. |
| `entries[].path` | Relative, `/`-separated, normalised: no `..`, `.` or empty segments, no absolute path, no backslash, no control character, at most 1024 bytes and 64 segments. Unique, and never both a file and a directory. |
| `entries[].size`, `sha256` | The byte count and the lower-case hex SHA-256 of the file's bytes. This is the file's identity, whatever the chunking. |
| `entries[].mediaType` | A media type. |
| `entries[].kind` | `file` (default) or `manifest`: the bytes are another manifest, whose paths are relative to this entry's directory. Manifests nest at most 8 deep. |
| `entries[].chunks` | Present when the file is longer than the chunk size: `chunkSize` (a multiple of 256 KiB, between 256 KiB and 64 MiB; 8 MiB unless the producer chooses otherwise) and one hex SHA-256 per chunk, in order. The last chunk is the remainder. |
| `entries[].locations`, `locations` | Ordered hints where the bytes can be fetched; an entry's list replaces the set's. |

### Identity

The manifest's identity is the SHA-256 of its **canonical form**: RFC 8785 (JSON Canonicalization Scheme), sorted keys, no insignificant whitespace, numbers and strings as `JSON.stringify` prints them. Both SDKs produce the same bytes from the same value; the vectors in `python/tests/fixtures/artifact_set/vectors.json` pin them.

### Locations

| `kind` | Fields | Meaning |
| --- | --- | --- |
| `https` | `url` | The producer's own endpoint (section 2). The URL is `https`; a loopback `http` URL is accepted for local development. |
| `oci` | `ref` | An OCI registry reference holding the blob as an artifact. |
| `s3` | `url` | A short-lived presigned URL on an S3-compatible store. |
| `robutler` | `url` | A platform content store serving the blob by hash. |
| `relay` | `agent` | The platform fetches the bytes over the producer's own outbound connection (section 4). |

A reader tries locations in order and verifies what it receives, so a location never has to be trusted. A kind a reader does not implement is refused by name (`location_kind_unsupported`) and the next location is tried. Version 1 of the SDK clients fetch `https` only.

### The strict reader

A manifest is input from another party. The reader refuses it, naming the rule, with `ArtifactSetError.code`: `wrong_kind`, `unknown_field`, `too_many_entries` (above 10,000), `path_absolute`, `path_parent_segment`, `path_invalid`, `path_too_long`, `path_too_many_segments`, `path_duplicate`, `path_conflict`, `size_invalid`, `sha256_invalid`, `media_type_invalid`, `kind_invalid`, `chunk_size_invalid`, `chunk_count_mismatch`, `chunk_hash_invalid`, `locations_invalid`, `too_many_locations` (above 16), `location_kind_invalid`, `location_url_invalid`, `created_at_invalid`, `expires_at_invalid`, `producer_invalid`, `nesting_too_deep`, `manifest_too_large` (above 16 MiB), `not_json`, `not_an_object`.

## 2. The producer endpoint

```
GET  <base>/artifacts/blobs/sha256:<hex>
HEAD <base>/artifacts/blobs/sha256:<hex>
```

`<base>` is the agent URL. `<hex>` names a published blob or a published manifest (its canonical bytes, `application/vnd.webagents.artifact-set+json`).

- **Ranges** (RFC 9110): `Accept-Ranges: bytes` on every answer; one `Range: bytes=a-b` is answered `206 Partial Content` with `Content-Range: bytes a-b/size`; a start past the end is `416` with `Content-Range: bytes */size`; several ranges or another unit are answered as the whole representation (`200`).
- **Digest** (RFC 9530): `Repr-Digest: sha-256=:<base64>:` of the whole representation on `200` and `206` alike, so a reader can check the name against the header before a byte arrives.
- **Streamed**: the body streams from disk; the producer holds one read buffer per transfer. Concurrent transfers are bounded; past the bound the answer is `503` with `{"error":{"code":"too_many_transfers","limit":N}}` and `Retry-After`.
- **Authorised by caller**: the request is signed (RFC 9421, `Signature-Agent`, as every agent-to-agent call). The agent identifies the signer (its access skill records `agent:<url>`) and the registry decides: a manifest is published to an audience (a list of principals, or every verified caller) and a blob is readable by whoever may read a manifest that lists it. A set published for a task also requires the header `Artifact-Task: <task>`. The refusals: an unverified caller `401 unauthorized`; a verified caller outside the audience, or an unknown hash, `404 not_found` (the two are not told apart); the right caller with no or the wrong task `403 task_required` / `403 task_mismatch`. A URL alone grants nothing.

## 3. The client

`fetchManifest(url)` fetches and parses a manifest (checking `Repr-Digest`, and the hash the caller expects). `fetchArtifactSet(set, { dest })` fetches every entry into a folder:

- one request per chunk (or per fixed range for a file without a chunk list), several in parallel, each written at its offset, each chunk hashed as it streams and refused on a mismatch (`chunk_hash_mismatch`);
- the whole-file hash is checked at the end for every file (`file_hash_mismatch` removes the file);
- progress lives beside the file in `<name>.part` and `<name>.part.json`, so a transfer cut at any point resumes with the chunks already verified;
- a `Repr-Digest` that names another hash than the manifest's is a `bad_response`, and the next location is tried;
- a server that ignores `Range` is accepted only for a whole-file request;
- an entry path can never resolve outside `dest` (`path_escape`).

## 4. The relay: frames over the agent's outbound connection

An agent behind a network it can only dial out of serves its artifacts through the platform it is connected to. The platform asks the agent for bytes over that connection and streams them on to its own HTTP clients; the agent never accepts a connection.

### Connections

The agent's **control connection** is its ordinary outbound UAMP WebSocket. For bulk frames the agent **may** open a **data connection**: a second WebSocket to the same URL, authenticated like the first (the same signed upgrade headers or the same token), with the header `UAMP-Connection: data` and the query `conn=data`, whose first frame is

```json
{ "type": "data.hello", "event_id": "…", "timestamp": 0, "uamp_version": "1.0", "agent": "<name>" }
```

`agent` is the name the platform knows the agent by, the same name `session.create` used on the control connection (with a per-agent key, the name the key carries). The platform answers `data.ready` or `data.error { error: { code, message } }`. The TypeScript SDK opens the data connection as soon as the control session exists when the agent has artifacts to serve, and reopens it when it drops. The platform sends a read on the data connection when it holds one, else on the control connection; **the agent answers on the connection the read arrived on**. A connection that closes ends every read on it.

### Frames

Platform to agent (JSON text frames):

| Frame | Fields | Meaning |
| --- | --- | --- |
| `blob.read` | `request_id` (1 to 64 bytes), `sha256`, `offset`, `length`, `credit?` | Send `length` bytes of the blob from `offset`. `credit` is the number of `blob.data` frames the agent may send before more credit arrives (default 4, at most 1024). One read asks for at most 64 MiB. |
| `blob.credit` | `request_id`, `frames` | More credit for a read in progress. |
| `blob.cancel` | `request_id` | Stop the read where it is. Nothing more is sent for it. |

Agent to platform:

| Frame | Fields | Meaning |
| --- | --- | --- |
| `blob.data` | binary, below | At most 256 KiB of the blob, at an offset. Never sent without credit. |
| `blob.end` | `request_id`, `offset`, `length`, `bytes` | Every byte of the read has been sent. |
| `blob.error` | `request_id`, `error: { code, message }` | The read cannot be served: `bad_request`, `unknown_blob`, `range_invalid`, `too_many_reads`, `duplicate_request`, `read_failed`. |
| `artifact.manifest` | `task_id`, `sha256`, `size`, `locations`, `manifest?`, `progress? { done, total }`, `session_id?` | A set is available (or being produced) for a task. `manifest` carries the parsed manifest when it is small enough to inline; `sha256` and `size` always name it. |

A platform that refuses an announcement answers `artifact.error { sha256, error: { code, message } }` on the same connection, echoing the announcement's `event_id`; a platform that accepts one sends nothing. An agent ignores frame types it does not know, so an older agent loses nothing.

### Announcing from an agent

The relay that serves the reads also announces. The connection that carries it (the portal bridge in TypeScript, `PortalConnectSkill` in both SDKs) attaches its control session to the relay when the platform confirms the session, and detaches it when the connection closes. `announceSet` / `announce_set` then sends `artifact.manifest` on that session:

- name the turn: pass the session id of the relayed turn that produced the set as `sessionId` (TypeScript `portalTurnOf(ctx).sessionId` in a tool, Python `portal_turn_of(context)["session_id"]`). The platform binds the set to the chat that turn came from, which is what lets that chat's participants read it. A set announced on the control session is bound to no chat, and a platform that serves sets only to a chat's participants serves it to nobody;
- the manifest is inlined by default (`inline: false` sends only its hash and size), so the platform can index every entry before the first read;
- `locations` defaults to this relay under the connected agent (`[{ "kind": "relay", "agent": "<name>" }]`);
- with no session attached yet, the call waits for one (30 seconds by default), then fails as `not_connected`;
- it returns once the frame is sent. With a confirm window (`confirmWithinMs` / `confirm_within`) it waits that long for an `artifact.error` naming this set and fails with its code; silence for the whole window is acceptance;
- every refusal, whichever call sent it, also reaches `onAnnounceError` / `on_announce_error`;
- nothing is re-announced on its own after a reconnect. The platform keeps what it was told.

TypeScript:

```ts
import { ArtifactRegistry, ArtifactRelay, files } from 'webagents/artifacts';
import { PortalConnectSkill, portalTurnOf } from 'webagents';

const registry = new ArtifactRegistry();
const relay = new ArtifactRelay(registry);
// The agent's skills include: new PortalConnectSkill({ artifacts: { relay } })

const published = await files.publish('./out', { producer: { agent: agentUrl, task: taskId }, registry, chunkSize: 8 * 1024 * 1024 });
relay.onAnnounceError((err) => console.error(`announcement refused: ${err.code} ${err.message}`));
// In the tool that produced the set (`ctx` is the context the tool receives):
await relay.announceSet({ taskId, published, sessionId: portalTurnOf(ctx)?.sessionId, confirmWithinMs: 2000 });
```

Python:

```python
from webagents.agents.skills.robutler.portal_connect import portal_turn_of
from webagents.artifacts import ArtifactProducer, ArtifactRegistry, ArtifactRelay, files

registry = ArtifactRegistry()
relay = ArtifactRelay(registry)
# PortalConnectSkill({"agents": [...], "artifact_relay": relay})

published = files.publish("./out", producer=ArtifactProducer(agent=agent_url, task=task_id), registry=registry)
# In the tool that produced the set (`context` is the run's context):
turn = portal_turn_of(context)
await relay.announce_set(published, task_id=task_id, session_id=turn["session_id"] if turn else None, confirm_within=2.0)
```

A Python connection may carry several agents; `announce_set(..., agent="<name>")` picks the session, and may be left out when only one is attached.

### `blob.data` binary framing

| Bytes | Content |
| --- | --- |
| 0 | version, `1` |
| 1 | kind, `1` (data) |
| 2 to 3 | request id length `n`, big-endian |
| 4 to 4+n-1 | request id, UTF-8 |
| 4+n to 11+n | offset of the first payload byte in the blob, unsigned 64-bit big-endian |
| 12+n onwards | payload, at most 262,144 bytes |

### Credit

The reader grants credit; the agent never sends ahead of it. A `blob.read` carries the initial credit; every `blob.credit` adds to it (the total held is capped at 1024 frames). A platform grants credit as it writes bytes on to its own client, so a slow reader slows the agent instead of filling memory. Frames of concurrent reads interleave by `request_id`, one frame per turn, so a large transfer does not starve a small one.

### Reconnects

A dropped connection is not a failed transfer. The platform notes how far each read got and, when the agent is back, asks again for the remaining range (a new `request_id`), which the chunk hashes make safe. The agent keeps no state across connections.

## 5. In A2A and MCP

- **A2A**: a result carries the manifest as a data part whose `data.kind` is `webagents.artifact-set/v1` (media type `application/vnd.webagents.artifact-set+json`, `metadata.sha256` its identity), plus one file part per entry with `url` (the preferred `https` location), `mediaType` and `filename` (the entry path). A plain A2A client follows the URLs; a webagents client reads the data part and verifies against it.
- **MCP**: resource links (`type: "resource_link"`) play the same role, one for the manifest and one per entry, with `uri`, `name`, `mimeType` and `size`.
- **Inputs**: a caller passes a manifest, never raw bytes for anything large; the agent fetches through the listed locations.

## 6. Limits, in one place

| Limit | Value |
| --- | --- |
| manifest bytes | 16 MiB |
| entries per manifest | 10,000 |
| path | 1024 bytes, 64 segments, 255 bytes per segment |
| nesting depth | 8 |
| locations per list | 16 |
| chunk size | 256 KiB to 64 MiB, a multiple of 256 KiB; default 8 MiB |
| `blob.data` payload | 256 KiB |
| credit | default 4 frames per read, at most 1024 held |
| one `blob.read` | 64 MiB |
| concurrent reads served (relay) | 16 |
| concurrent transfers served (endpoint) | 16 |
