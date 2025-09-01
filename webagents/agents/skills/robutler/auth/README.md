# AuthSkill (Robutler V2)

Owner-aware authentication and authorization for agents. Integrates with the Robutler Portal and supports secure owner assertions for owner-only tools (e.g., ControlSkill).

## Features
- API key validation against the Portal
- Owner scope detection (admin/user/owner)
- Owner assertions via `X-Owner-Assertion`:
  - RS256 with JWKS (recommended for external agents)
  - HS256 fallback for in-infra setups
- Identity propagation: `origin_user_id`, `peer_user_id`, `agent_owner_user_id`

## Headers
- `Authorization: Bearer <key>` one of:
  - Service token (srv_...) for backend-to-backend
  - Agent API key (rok_...)
  - Session/JWT for owner (when calling Portal endpoints)
- `X-Owner-Assertion: <jwt>` short‑lived JWT binding the caller to an agent (see below)
- `X-Origin-User-ID`, `X-Peer-User-ID`, `X-Agent-Owner-User-ID` optional hints

## Owner Assertion JWT
- Claims:
  - `iss`: robutler-portal
  - `aud`: robutler-agent:<agentId>
  - `sub`: origin_user_id
  - `agent_id`: <agentId>
  - `owner_user_id`: <ownerId>
  - `jti`: unique id
  - `iat/nbf/exp`: short TTL (2–5 minutes)
- Signing:
  - RS256 with `OWNER_ASSERTION_PRIVATE_KEY` and JWKS served at `/api/auth/jwks`
  - HS256 fallback with `OWNER_ASSERTION_SECRET` (or `AUTH_SECRET`)

## Env (Agent Service)
- RS256: `OWNER_ASSERTION_JWKS_URL` → Portal JWKS URL
- HS256: `OWNER_ASSERTION_SECRET` (shared with Portal) or rely on `AUTH_SECRET`

## Env (Portal)
- RS256: `OWNER_ASSERTION_PRIVATE_KEY` (PEM), `OWNER_ASSERTION_KID`
- HS256: `OWNER_ASSERTION_SECRET` or `AUTH_SECRET`
- `SERVICE_TOKEN` for backend issuance

## Issuance API (Portal)
- `POST /api/auth/owner-assertion`
  - Service token: mint for any owner userId
  - Owner session/API key: owner-only; assertion is always for the caller
  - Body: `{ agentId, originUserId?, ttlSeconds? }`
  - Response: `{ assertion, expiresAt }`
- `GET /api/auth/jwks` → public keys for RS256 verification

## Verification (AuthSkill)
- Validates `Authorization` with Portal
- Verifies `X-Owner-Assertion` when present:
  - RS256 via JWKS if `OWNER_ASSERTION_JWKS_URL` is set
  - HS256 via `OWNER_ASSERTION_SECRET`/`AUTH_SECRET` fallback
  - Enforces `aud = robutler-agent:<agentId>`
  - Enforces `claims.agent_id == agent.id`
  - Upgrades scope to OWNER when the verified user is the agent owner

## Best Practices
- Always send over TLS; never log assertion values
- Short TTL; consider `jti` replay cache for high-security deployments
- Don’t expose agent API keys to browsers; inject headers server-side only
