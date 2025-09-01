# Auth Skill

Authentication and authorization for agents using the Robutler Platform. Establishes a unified `AuthContext` and a secure, interoperable mechanism for agent‑to‑agent authorization via RS256 owner assertions (JWT).

## Features
- API key authentication with the Robutler Platform
- Role‑based access control: admin, owner, user
- `on_connection` authentication hook
- Agent owner scope detection (from agent metadata)
- Harmonized `AuthContext` with minimal, stable fields
- Optional agent‑to‑agent assertions via `X-Owner-Assertion` (short‑lived RS256 JWT, verified via JWKS)

## Configuration
```python
from webagents.agents import BaseAgent
from webagents.agents.skills.robutler.auth import AuthSkill

agent = BaseAgent(
    name="secure-agent",
    model="openai/gpt-4o",
    skills={
        "auth": AuthSkill({
            "api_key": "your_platform_api_key",         # Optional: defaults to agent.api_key
            "platform_api_url": "https://robutler.ai",  # Optional: $ROBUTLER_INTERNAL_API_URL or $ROBUTLER_API_URL or http://localhost:3000
            "require_auth": True                          # Optional: defaults to True
        })
    }
)
```

Note: `platform_api_url` resolves in this order: `$ROBUTLER_INTERNAL_API_URL` → `$ROBUTLER_API_URL` → `http://localhost:3000`.

## Scopes
- **admin**: Platform administrators
- **owner**: Agent owner (API key belongs to the agent owner)
- **user**: Regular authenticated users

If not explicitly set, the default scope is `user`.

## Identity and Context
The auth skill validates the API key during `on_connection` and exposes an `AuthContext` on the request context:

```python
from webagents.server.context.context_vars import get_context

context = get_context()
auth = context.auth  # instance of AuthContext

# Harmonized fields
user_id = auth.user_id               # caller identity; overridden by JWT `sub` when verified
agent_id = auth.agent_id             # agent id from verified assertion (if provided)
scope = auth.scope.value             # "admin" | "owner" | "user"
authenticated = auth.authenticated   # bool
assertion = auth.assertion           # dict of decoded claims (if provided)
```

Deprecated identity fields (e.g., `origin_user_id`, `peer_user_id`, `agent_owner_user_id`) have been removed in favor of the harmonized fields above.

## Authentication Flow
1. Extract API key from `Authorization` (Bearer) or `X-API-Key`
2. Validate API key with the Robutler Platform
3. Determine scope based on the validated user and agent ownership
4. Optionally verify `X-Owner-Assertion` (RS256, JWKS) and merge acting identity into `AuthContext`
5. Populate `context.auth` with an `AuthContext` instance

## Agent‑to‑Agent Assertions (Owner Assertions)
- Primary purpose: secure, interoperable agent‑to‑agent authentication and authorization across services.
- Also enables owner‑only actions (e.g., ControlSkill) without exposing agent API keys to clients.
- Transport: send `X-Owner-Assertion: <jwt>` alongside your `Authorization` header.

### Claims
- `aud = robutler-agent:<agentId>` — audience bound to the target agent
- `agent_id = <agentId>` — agent identity binding
- `sub = <userId>` — acting end‑user identity
- `owner_user_id = <ownerId>` — agent owner (advisory)
- `jti` — unique token id for optional replay tracking
- `iat` / `nbf` / `exp` — very short TTL (2–5 minutes)

### Verification by AuthSkill
- Signature verification via JWKS (RS256)
- Enforce audience and agent binding: `aud == robutler-agent:<agentId>` and `agent_id == agent.id`
- On success, update context:
  - `context.auth.user_id = sub` (acting identity)
  - `context.auth.agent_id = agent_id`
  - `context.auth.assertion = <decoded claims>`
- OWNER scope is derived by comparing the API‑key owner to the agent’s `owner_user_id`; the assertion does not grant owner scope by itself.

### JWKS and configuration
- The skill discovers the JWKS at `OWNER_ASSERTION_JWKS_URL` if set; otherwise at `{platform_api_url}/api/auth/jwks`.
- Only RS256 is supported. HS256 and shared‑secret fallbacks are not supported.

### High‑level flow
```mermaid
sequenceDiagram
  participant U as User
  participant C as Chat Server
  participant A as Agent Service
  participant Auth as AuthSkill
  participant Ctrl as ControlSkill

  U->>C: Edit agent description
  C->>A: Request + headers\nAuthorization, X-Owner-Assertion, X-Payment-Token
  A->>Auth: on_connection()
  Auth-->>Auth: Verify API key + (optional) verify assertion
  Auth-->>A: Set context.auth; derive scope (OWNER if API‑key owner == agent.owner_user_id)
  A->>Ctrl: manage_agent(update_description)
  Ctrl-->>A: Allowed (owner scope)
  A->>U: ✅ Description updated
```

## Defaults and edge cases
- If the skill is enabled and authentication succeeds, `auth.scope` defaults to `user` unless elevated to `owner` or `admin`.
- If the skill is disabled (`require_auth=False`) or not configured, downstream tools should treat the request as unauthenticated and avoid owner/admin‑scoped operations.

## Example: protecting an owner‑only tool
```python
from webagents.agents.skills.robutler.auth.skill import AuthScope

def update_agent_settings(context, patch):
    if context.auth.scope != AuthScope.OWNER:
        raise PermissionError("Owner scope required")
    # proceed with update
```

Implementation: see `robutler/agents/skills/robutler/auth/skill.py`.