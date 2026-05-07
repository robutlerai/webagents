# REST API — Functions

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
