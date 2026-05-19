# UAMP Extensions

UAMP's core vocabulary — `session.*`, `input.*`, `response.*`, `tool.*`,
`payment.*`, `audio.*`, `transcript.*`, `presence.*`, `capabilities.*`,
`ping`/`pong`, `rate_limit`, etc. — is shaped around **agent
conversations** (LLM-mediated turns, content items, tool calls, payments).

Some applications need to ride the same WebSocket for sub-protocols that
are NOT part of the agent loop — workspace widgets, peer-machine control
channels, file browsers, clipboard sync, screencast, and so on.

UAMP supports this via **one** generic frame: `extension.message`.

## The envelope

```ts
export interface ExtensionMessageEvent extends BaseEvent {
  type: 'extension.message';
  /** Dotted, owned identifier (e.g. `workspace.terminal`). */
  namespace: string;
  /** Optional sub-protocol version (default 1). */
  extension_version?: number;
  /** Sub-protocol payload, opaque to UAMP. */
  payload: unknown;
}
```

`extension.message` is the **only** UAMP frame that flows in both
directions and is the **only** member of both `ClientEvent` and
`ServerEvent`.

## Rules

### 1. Namespace ownership

`namespace` is a dotted, owned identifier. Names that overlap UAMP's
core (anything starting with `session.`, `input.`, `response.`, `tool.`,
`payment.`, `audio.`, `transcript.`, `presence.`, `capabilities.`,
`conversation.`, `client.`, `rate_limit`, `ping`, `pong`) are reserved
for the spec.

The application that defines a namespace owns its semantics, payload
schema, error vocabulary, and version policy. UAMP does not validate any
of these.

### 2. Opaque payload

`payload` is opaque to UAMP. UAMP does not parse, validate, or transform
it. Sub-protocols can use any JSON-serialisable shape.

The envelope **does not carry a `session_id` field**. UAMP-session
correlation, if a sub-protocol needs it, lives inside `payload`. We
deliberately avoid a top-level `session_id` to prevent confusion with
sub-protocol-defined session ids (e.g. `workspace.terminal` carries its
own per-PTY `session_id` inside the payload).

### 3. Sub-protocol versioning

`extension_version` is optional, default `1`. Sub-protocols use it to
negotiate forward/backward compatibility WITHOUT bumping the UAMP
version.

The receiving sub-protocol router (NOT UAMP) is the validator. v1
routers reject unknown versions with a sub-protocol-typed error inside
the payload, e.g.

```json
{
  "type": "extension.message",
  "namespace": "workspace.terminal",
  "extension_version": 99,
  "payload": { "type": "open", "session_id": "...", "cols": 80, "rows": 24 }
}
```

→

```json
{
  "type": "extension.message",
  "namespace": "workspace.terminal",
  "extension_version": 1,
  "payload": {
    "type": "err",
    "session_id": "...",
    "code": "unsupported_version",
    "message": "workspace.terminal supports version 1, got 99"
  }
}
```

### 4. Bidirectional, intentionally non-classified

Because the envelope flows both ways, the legacy classification guards
intentionally treat it as neither a client nor a server event:

```ts
isClientEvent(envelope)     // false
isServerEvent(envelope)     // false
isExtensionMessage(envelope) // true   ← use this
```

Adding `extension.message` to `isClientEvent` would mis-classify
envelopes coming FROM a server, and vice versa. Use the dedicated
predicate instead.

### 5. Unsupported namespaces

A daemon that doesn't host a namespace responds with an
`extension.message` whose payload signals "unsupported" in the
sub-protocol's own vocabulary.

UAMP does NOT add a generic `extension.unsupported` frame — each
namespace owns its error model. For `workspace.terminal` the contract is
`{ type: 'err', code: 'not_supported', message: '...' }` returned to the
first `open` payload received.

This keeps the spec narrow and lets sub-protocols evolve independently.

## Helpers

```ts
import {
  createExtensionMessage,
  isExtensionMessage,
  parseEvent,
  serializeEvent,
} from 'webagents/uamp';

const env = createExtensionMessage('workspace.terminal', {
  type: 'open',
  session_id: 'term_abc',
  peer_id: 'usr_42',
  cols: 80,
  rows: 24,
}, { version: 1 });

const wire = serializeEvent(env);
const parsed = parseEvent(wire);

if (isExtensionMessage(parsed)) {
  console.log(parsed.namespace, parsed.payload);
}
```

`createExtensionMessage(namespace, payload, opts?)`:

- `namespace` (string, required) — dotted owned identifier.
- `payload` (unknown, required) — opaque sub-protocol payload.
- `opts.version` (number, optional) — emitted as `extension_version` only
  when set. Omit to default to "no field" (which receivers treat as
  version 1).

## Worked example: `workspace.terminal`

Wire layout (browser → portal binary frames are converted by the portal
gateway into `extension.message` envelopes on the daemon WS):

```
browser ──[binary]──► portal ──[ws.extension.message]──► daemon ──► PTY
                                ◄──[ws.extension.message]──
```

Inner payload shapes (all carry `session_id` for the per-PTY id):

| Direction | `payload.type` | Fields |
| --- | --- | --- |
| client → daemon | `open` | `session_id`, `peer_id`, `cols`, `rows` |
| client → daemon | `in` | `session_id`, `data` (base64) |
| client → daemon | `resize` | `session_id`, `cols`, `rows` |
| client → daemon | `pause` / `resume` | `session_id` |
| client → daemon | `close` | `session_id`, `reason?` |
| daemon → client | `ready` | `session_id` |
| daemon → client | `out` | `session_id`, `data` (base64) |
| daemon → client | `exit` | `session_id`, `code`, `signal?` |
| daemon → client | `err` | `session_id`, `code`, `message` |

Defined error codes for `workspace.terminal` v1:

- `not_supported` — daemon has no PTY surface.
- `bad_payload` — malformed sub-protocol payload.
- `pty_open_failed` / `pty_write_failed` / `pty_resize_failed` /
  `pty_close_failed` — Tauri / OS error.
- `unsupported_version` — `extension_version` higher than the router
  supports.
- `concurrency_limit` — per-router 8-session cap reached.
- `duplicate_session` — `open` for an already-open `session_id`.

See [`webagents/typescript/src/transport/terminal/`](../src/transport/terminal/)
for the reference router implementation.

## Adding a new namespace

1. Pick a dotted name you own (e.g. `workspace.vnc`, `acme.audit`).
2. Define typed `IncomingPayload` / `OutgoingPayload` unions in your
   own module.
3. Implement a router that:
   - Detects malformed payloads -> `{ type: 'err', code: 'bad_payload' }`
     (or your sub-protocol's equivalent).
   - Validates `extension_version` against your supported set -> typed
     `unsupported_version` error if mismatched.
   - Handles missing capability cleanly -> typed `not_supported` error.
4. Wire the router into the transport that exposes the namespace
   (e.g. `PortalTransportSkill`'s `terminal` config slot).
5. Document the namespace in YOUR repo, NOT the UAMP spec — UAMP only
   owns the envelope.

UAMP versions never need to bump for new namespaces.
