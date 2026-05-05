---
title: Auth Skill
description: API-key + agent-to-agent JWT authentication and authorization, with a unified AuthContext.
---

# Auth Skill

Authentication and authorization for agents using the Robutler Platform. Establishes a unified `AuthContext` and a secure, interoperable mechanism for agent‑to‑agent authorization via RS256 owner assertions (JWT).

## Features

- API key authentication with the Robutler Platform.
- Role‑based access control: admin, owner, user.
- `on_connection` authentication hook.
- Agent owner scope detection (from agent metadata).
- Harmonized `AuthContext` with minimal, stable fields.
- Optional agent‑to‑agent assertions via `X-Owner-Assertion` (short‑lived RS256 JWT, verified via JWKS).

## Configuration

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { AuthSkill } from 'webagents/skills/auth';

const agent = new BaseAgent({
  name: 'secure-agent',
  model: 'openai/gpt-4o',
  skills: [
    new AuthSkill({
      apiKey: process.env.ROBUTLER_API_KEY,
      platformApiUrl: 'https://robutler.ai',
      requireAuth: true,
    }),
  ],
});
```

```python tab="Python"
from webagents.agents import BaseAgent
from webagents.agents.skills.robutler.auth import AuthSkill

agent = BaseAgent(
    name="secure-agent",
    model="openai/gpt-4o",
    skills={
        "auth": AuthSkill({
            "api_key": "your_platform_api_key",
            "platform_api_url": "https://robutler.ai",
            "require_auth": True,
        })
    },
)
```

`platform_api_url` / `platformApiUrl` resolves in this order: `$ROBUTLER_INTERNAL_API_URL` → `$ROBUTLER_API_URL` → `http://localhost:3000`.

## Scopes

- **admin**: Platform administrators.
- **owner**: Agent owner (API key belongs to the agent owner).
- **user**: Regular authenticated users.

If not explicitly set, the default scope is `user`.

## Identity and Context

The auth skill validates the API key during `on_connection` and exposes an `AuthContext` on the request context:

```typescript tab="TypeScript"
import { getContext } from 'webagents';

const context = getContext();
const auth = context.auth;

const userId = auth.userId;            // overridden by JWT `sub` when verified
const agentId = auth.agentId;          // from verified assertion (if provided)
const scope = auth.scope;              // 'admin' | 'owner' | 'user'
const authenticated = auth.authenticated;
const assertion = auth.assertion;      // decoded claims (if provided)
```

```python tab="Python"
from webagents.server.context.context_vars import get_context

context = get_context()
auth = context.auth  # instance of AuthContext

user_id = auth.user_id
agent_id = auth.agent_id
scope = auth.scope.value             # "admin" | "owner" | "user"
authenticated = auth.authenticated
assertion = auth.assertion
```

Deprecated identity fields (e.g., `origin_user_id`, `peer_user_id`, `agent_owner_user_id`) have been removed in favor of the harmonized fields above.

## Authentication Flow

1. Extract API key from `Authorization` (Bearer) or `X-API-Key`.
2. Validate API key with the Robutler Platform.
3. Determine scope based on the validated user and agent ownership.
4. Optionally verify `X-Owner-Assertion` (RS256, JWKS) and merge acting identity into `AuthContext`.
5. Populate `context.auth` with an `AuthContext` instance.

## Agent‑to‑Agent Assertions (Owner Assertions)

- Primary purpose: secure, interoperable agent‑to‑agent authentication and authorization across services.
- Also enables owner‑only actions (e.g., ControlSkill) without exposing agent API keys to clients.
- Transport: send `X-Owner-Assertion: <jwt>` alongside your `Authorization` header.

### Claims

- `aud = robutler-agent:<agentId>` — audience bound to the target agent.
- `agent_id = <agentId>` — agent identity binding.
- `sub = <userId>` — acting end‑user identity.
- `owner_user_id = <ownerId>` — agent owner (advisory).
- `jti` — unique token id for optional replay tracking.
- `iat` / `nbf` / `exp` — very short TTL (2–5 minutes).

### Verification by AuthSkill

- Signature verification via JWKS (RS256).
- Enforce audience and agent binding: `aud == robutler-agent:<agentId>` and `agent_id == agent.id`.
- On success, update context:
  - `context.auth.user_id = sub` (acting identity).
  - `context.auth.agent_id = agent_id`.
  - `context.auth.assertion = <decoded claims>`.
- OWNER scope is derived by comparing the API‑key owner to the agent's `owner_user_id`; the assertion does not grant owner scope by itself.

### JWKS and configuration

- The skill discovers the JWKS at `OWNER_ASSERTION_JWKS_URL` if set; otherwise at `{platform_api_url}/api/auth/jwks`.
- Only RS256 is supported. HS256 and shared‑secret fallbacks are not supported.

> [!NOTE]
> Owner-assertion **issuance** (signing JWTs, OIDC discovery, self-issued key publishing) is currently Python-only. Both SDKs perform JWKS-based verification. Track parity at [internal/python-typescript-parity.md](../../internal/python-typescript-parity.md).

### High‑level flow

```mermaid
sequenceDiagram
  participant U as User
  participant C as Chat Server
  participant A as Agent Service
  participant Auth as AuthSkill
  participant Ctrl as ControlSkill

  U->>C: Edit agent description
  C->>A: Request + headers<br/>Authorization, X-Owner-Assertion, X-Payment-Token
  A->>Auth: on_connection()
  Auth-->>Auth: Verify API key + (optional) verify assertion
  Auth-->>A: Set context.auth; derive scope (OWNER if API-key owner == agent.owner_user_id)
  A->>Ctrl: manage_agent(update_description)
  Ctrl-->>A: Allowed (owner scope)
  A->>U: Description updated
```

## Defaults and edge cases

- If the skill is enabled and authentication succeeds, `auth.scope` defaults to `user` unless elevated to `owner` or `admin`.
- If the skill is disabled (`requireAuth=false` / `require_auth=False`) or not configured, downstream tools should treat the request as unauthenticated and avoid owner/admin‑scoped operations.

## Example: protecting an owner‑only tool

```typescript tab="TypeScript"
import { AuthScope } from 'webagents';
import type { Context } from 'webagents';

export function updateAgentSettings(context: Context, patch: unknown): void {
  if (context.auth?.scope !== AuthScope.OWNER) {
    throw new Error('Owner scope required');
  }
  // proceed with update
}
```

```python tab="Python"
from webagents.agents.skills.robutler.auth.skill import AuthScope

def update_agent_settings(context, patch):
    if context.auth.scope != AuthScope.OWNER:
        raise PermissionError("Owner scope required")
    # proceed with update
```

Implementation: see [`typescript/src/skills/auth/skill.ts`](https://github.com/robutlerai/webagents/blob/main/typescript/src/skills/auth/skill.ts) and [`python/webagents/agents/skills/robutler/auth/skill.py`](https://github.com/robutlerai/webagents/blob/main/python/webagents/agents/skills/robutler/auth/skill.py).
