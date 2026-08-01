---
title: REST API — Functions
description: Owner-scoped REST endpoints for declaring, validating, invoking, and inspecting agent-owned functions.
---

All routes are scoped to a single agent (`/api/agents/<idOrUsername>/...`). Auth is owner session OR `portal_token` (RS256 JWT) unless noted otherwise.

## List functions

```
GET /api/agents/:id/functions
```

Returns the agent's declared functions enriched with `usedBy` (which skills consume each function).

```json
{
  "functions": [
    {
      "name": "stripeHandler",
      "declaration": { "contentId": "ctn_abc", "runtime": "js-v1", "permissions": { ... } },
      "usedBy": [{ "skill": "custom_http", "entryId": "stripe_webhook", "description": "POST /webhooks/stripe" }]
    }
  ]
}
```

## Declare or update

```
POST /api/agents/:id/functions
```

Body: `{ name, manifest, source? }`. Validates the manifest, stores the function under `agent_configs.functions[name]`, and writes an audit row to `function_invocations` (source_skill = `authoring`).

Returns `{ ok: true, name, requiresUserAction?: [...] }` — `requiresUserAction` is non-empty when the manifest declares secret bindings the owner hasn't set yet.

## Remove

```
DELETE /api/agents/:id/functions/:name
```

Removes the entry and detaches all consumer references. Audit row recorded.

## Validate

```
POST /api/agents/:id/functions/:name/validate
```

Body: `{ manifest, source? }`. Returns `{ ok, errors[], warnings[] }`. Counts against the validation quota bucket; runtime-side validation is forwarded to the executor `/validate` endpoint when `WEBAGENTS_EXECUTOR_URL` is set.

## Manual invoke

```
POST /api/agents/:id/functions/:name/invoke
```

Headers: `Idempotency-Key` (24h Redis dedupe). Body shape depends on the consumer:

| Consumer        | Body                                                  |
| ---             | ---                                                   |
| `custom_tools`  | `{ args: <parameter-schema-validated payload> }`      |
| `custom_http`   | `{ method, path?, query?, headers?, body? }`          |
| `cron` (replay) | `{}`                                                  |

Counts against quotas / billing same as any other invocation.

## Invocation history

```
GET /api/agents/:id/functions/:name/invocations?limit=50&cursor=<iso>
```

Paginated by `started_at` desc; rows from `function_invocations`.

## Set secret

```
POST /api/agents/:id/functions/:name/secret
```

Body: `{ binding, value }`. Owner-session-authenticated only. Stores the value as JWE in `memory(serverEncrypted=true, namespace='fn-secret:<name>')`. The function reads it via `ctx.secrets.get('<binding>')`.

```
DELETE /api/agents/:id/functions/:name/secret?binding=<name>
```

Removes the stored secret value.

## Auto-generated OpenAPI

```
GET /api/agents/:id/functions/openapi.json
```

OpenAPI 3.1 spec derived from `agent_configs.functions[*].parameters` plus the active `custom_tools` / `custom_http` skill consumers, plus the manual-invoke endpoints.

## Auth headers

| Surface             | Header(s)                                                    |
| ---                 | ---                                                          |
| Owner session       | Cookie-based session, no extra headers                       |
| Portal token        | `Authorization: Bearer <RS256 JWT>`                          |
| Factory / host edit | `Function-Authoring-Surface: factory \| host \| ui \| cli`   |

The portal validates the surface header against the calling agent id (host-edit can't edit other agents).

## Runtime: js-v1 host API surface

User-authored functions run inside a per-tenant V8 isolate (`isolated-vm`) and reach the platform exclusively through the `ctx` argument passed to `handler(ctx)`. Egress (`ctx.fetch`) executes in the executor's worker thread; every other stateful API round-trips to the portal's `/api/internal/fn-host` endpoint, authed with a 60-second RS256 JWT (`typ: 'fn-invocation'`) whose claims pin agent / function / invocation / permissions / folder bindings (see [ADR-0009](../../docs/adr/0009-function-host-bridge.md)).

Permission shapes live under `manifest.permissions`:

```jsonc
{
  "permissions": {
    "fetch":   ["https://api.example.com", "*.openai.com", "*"],   // allowlist
    "secrets": ["STRIPE_KEY", "write"],                            // names + "write" sentinel
    "kv":      "ro" | "rw" | "none",                               // mode (NOT array)
    "content": { "read": true, "write": false },
    "folders": [{ "alias": "uploads", "contentId": "ctn_…", "permissions": "rw" }],
    "portal":  ["payment.lock", "payment.settle", "callTool"],     // allowlist
    "rawBody": false,
    "selfRecursion": false
  }
}
```

| `ctx` API                  | Permission required                                        | Notes                                                                                                          |
| ---                        | ---                                                        | ---                                                                                                            |
| `ctx.fetch(url, init?)`    | `permissions.fetch` allowlist (`*`, exact URL, `*.host`)   | Worker-direct (no host bridge). Tracks `ingressBytes`/`egressBytes`. Non-http(s) protocols rejected.            |
| `ctx.secrets.get(name)`    | `name` ∈ `permissions.secrets`                             | Returns plaintext (decryption stays in the portal).                                                            |
| `ctx.secrets.put(name, v)` | `permissions.secrets` includes `"write"` AND `name`        | Encrypted at rest (`memory.serverEncrypted=true`, namespace `fn-secret:<fn>`).                                 |
| `ctx.secrets.list()`       | `permissions.secrets` non-empty                            | Returns names ∈ allowlist that exist in storage.                                                                |
| `ctx.kv.get(key)`          | `permissions.kv` ∈ `{ ro, rw }`                            | Namespace `fn:<fn>` per agent.                                                                                 |
| `ctx.kv.put(k, v, opts?)`  | `permissions.kv` = `rw`                                    | Counts against per-invocation `kvPutBytes` quota (default 10 MB).                                              |
| `ctx.kv.delete(k)`         | `permissions.kv` = `rw`                                    |                                                                                                                |
| `ctx.kv.putIfAbsent(k, v)` | `permissions.kv` = `rw`                                    | Atomic claim: inserts ONLY if absent, returns `{ won: boolean }`. Same quotas as `put`; a lost race still charges. |
| `ctx.kv.list(prefix?, …)`  | `permissions.kv` ∈ `{ ro, rw }`                            | Cursor-paginated.                                                                                              |
| `ctx.content.get(id)`      | `permissions.content.read = true`                          | Returns `{ id, mimeType, displayName, size, arrayBuffer() }`. Body capped by per-invocation ingress quota.     |
| `ctx.content.put(item)`    | `permissions.content.write = true`                         | Counts against per-invocation `contentWrites` quota (default 5).                                               |
| `ctx.folders[alias].list()` | binding ∈ `permissions.folders`                           | Token-frozen at envelope build — renamed bindings can't escalate scope mid-flight.                             |
| `ctx.folders[alias].read(name)` | binding ∈ `permissions.folders`                       |                                                                                                                |
| `ctx.folders[alias].write(name, body)` | binding `permissions: "rw"`                    | Counts against `contentWrites` quota.                                                                          |
| `ctx.fn.invoke(name, args)` | sibling fn declared on same agent                         | Chain depth ≤ 4 (matches `FunctionRuntimeSkillConfig.maxChainDepth`); cycles detected.                         |
| `ctx.fn.list()`            | none                                                       | Returns sibling fn names.                                                                                      |
| `ctx.portal.<method>(...)` | `method` ∈ `permissions.portal`                            | Methods: `verifyToken`, `verifyHmac`, `lookupAgent`, `callTool`, `getOwner`, `notifyOwner`, `signContentUrl`, `payment.{lock,settle,release}`. |
| `ctx.log.{debug,info,warn,error}(...)` | none                                           | Buffered host-side, capped at 64 KB per invocation; returned in `ExecutorResponse.logs`.                       |
| `ctx.emit(event, payload?)` | none                                                      | Surfaced to Server-Sent Events stream when the calling skill subscribes.                                       |
| `ctx.request.rawBody`      | `permissions.rawBody = true`                               | `Uint8Array` view; only populated when the caller actually carries a body.                                     |

### Runtime sandbox details

- Sandbox: bare V8 isolate (`isolated-vm`) on Node 20 LTS. **Blocked**: `process`, `Buffer`, `require`, `fs`, `eval` (throws `EVAL_DENIED`), `Function` constructor (throws `FUNCTION_DENIED`).
- **Available globals** (web platform subset): `URL`, `URLSearchParams`, `atob`, `btoa`, `JSON`, `Math`, `Date`, `Promise`, `Map`/`Set`/`WeakMap`/`WeakSet`, `RegExp`, `Symbol`, `Proxy`, `Reflect`, `Intl`, `console`, `TextEncoder`/`TextDecoder`, `structuredClone`, `crypto.{randomUUID, getRandomValues, subtle}` (digest/sign/verify/encrypt/decrypt/importKey).
- `fetch` is also installed as a top-level alias of `ctx.fetch` (same allowlist enforcement).
- npm packages, Node-only modules, and curated host libs are **not** available in v1; bundling is tracked for v2.
- Entrypoint: must export an async `handler(ctx)` — `export default async function handler(ctx)` (preferred), `export async function handler(ctx)`, or `module.exports = async (ctx) => …`. Source ≤ 16 KB UTF-8 (inline) or 64 KB base64 (`inlineB64`); larger source must move to a content row.
- Limits: `wallMs` default 10 s (max 30 s), `memoryMb` default 256 (max 512), enforced by an `isolated-vm` memory cap and a host-side wall-clock watchdog that disposes the isolate on overshoot.
- Error codes: `WALL_TIMEOUT`, `MEMORY_LIMIT_EXCEEDED`, `EVAL_DENIED`, `FUNCTION_DENIED`, `FETCH_FORBIDDEN`, `HOST_QUOTA_EXCEEDED`, `PERMISSION_DENIED`, `HOST_BRIDGE_ERROR`, `JS_NO_HANDLER`, `JS_RESULT_NOT_SERIALIZABLE`, `JS_RUNTIME_ERROR`.
- Host-bridge quotas (per invocation, Redis-backed): `callsTotal ≤ 100`, `kvPutBytes ≤ 10 MB`, `contentWrites ≤ 5`. Breach surfaces as HTTP 409 `HOST_QUOTA_EXCEEDED` to the executor and `HOST_QUOTA_EXCEEDED` to the function.

`python-pyodide-v1` is deferred ([ADR-0008](../../docs/adr/0008-pyodide-deferred.md)); manifests pinning it fail validation with `RUNTIME_DISABLED`. `wasm-v1` is reserved for v2.
