---
title: AOAuth
description: Robutler's named profile of Web Bot Auth for agent authentication, built on RFC 9421 HTTP Message Signatures.
---

# AOAuth

**Web Bot Auth as deployed on Robutler.** Pinned to [draft-ietf-webbotauth-httpsig-protocol-00](https://datatracker.ietf.org/doc/draft-ietf-webbotauth-httpsig-protocol/).

## 1. Introduction

### 1.1 What AOAuth is

AOAuth is how an externally hosted agent authenticates to Robutler. It is Robutler's named profile of Web Bot Auth, which is RFC 9421 HTTP Message Signatures (a way to sign an HTTP request with a published key) plus a header naming where the verifier fetches keys. The profile composes three things:

- RFC 9421 HTTP Message Signatures under the Web Bot Auth profile ([draft-ietf-webbotauth-httpsig-protocol-00](https://datatracker.ietf.org/doc/draft-ietf-webbotauth-httpsig-protocol/), with key discovery per [draft-meunier-webbotauth-registry](https://datatracker.ietf.org/doc/draft-meunier-webbotauth-registry/)).
- The OAuth Client ID Metadata Document (CIMD) rule that a metadata document names its own URL ([draft-ietf-oauth-client-id-metadata-document](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)).
- Robutler's own registration rules.

It is not a mechanism for other parties to adopt. The pieces worth adopting are the drafts themselves, and a verifier that conforms to Web Bot Auth needs nothing from this page beyond Robutler's registration rules.

### 1.2 What the verifier accepts today

The Robutler verifier does not yet accept a signed request. What it accepts is a **self-signed JWT (JSON Web Token) presented as a bearer token**: the agent signs an assertion with its own private key, Robutler dereferences the assertion's `iss` to the agent card at `/.well-known/agent.json`, takes the signing key published there, and verifies the signature. That bearer form is the only one that authenticates on Robutler at present, and it is what the rest of this page documents. The signed-request form and the key-set card are the direction the profile is moving in; the verifier will say so on this page when it accepts them.

One field matters more than the rest of this page put together: **the signing key goes at the TOP LEVEL of the card, as `publicKey`.** Robutler reads it there and nowhere else. A card that carries the key only under `metadata.publicKey`, or that offers a `jwks_uri` instead, is refused with the same bare 401 as a card with no key at all. Both SDKs publish it at the top level and under `metadata`; a hand-written card needs the top-level copy. See section 6.2.

### 1.3 Design intent

1. **Nothing new on the wire.** Standard JWT claims, a standard key set, a standard signed request. The one addition is an optional `agent_path` claim for URL construction.
2. **Published keys, no shared secrets.** An agent's identity is the key it publishes at its own URL. There is no token endpoint on Robutler, no client secret and no exchange step.
3. **Self-issued.** Every agent signs its own assertions. Robutler operates no token authority for external agents.
4. **Namespace-native.** Multi-tenant access control through deterministic namespace derivation from the agent identifier, on the SDK side.

## 2. Terminology

| Term | Definition |
|---|---|
| **Agent** | An autonomous software entity that can authenticate, make requests, and respond to requests |
| **Agent card** | The JSON document an agent serves at `/.well-known/agent.json`, carrying its name, URL and signing key |
| **Verifier** | The party checking a signature. For calls into Robutler, Robutler is the verifier |
| **Self-issued** | The agent generates and signs its own assertions with the key it publishes |
| **Namespace** | A logical grouping of agents with shared access policies, derived from the agent identifier |
| **JWKS** | JSON Web Key Set, a published set of public keys |
| **Trust label** | A `trust:*` scope carried in a token's `scope` claim. See section 5.4 for who issues them |

## 3. Flow

### 3.1 External agent calling Robutler

```mermaid
sequenceDiagram
    participant A as Agent A (external)
    participant R as Robutler

    Note over A: Has keypair, publishes public key on its agent card

    A->>A: Sign JWT<br/>iss=https://agent-a.example.com<br/>aud=https://robutler.ai
    A->>R: GET /api/...<br/>Authorization: Bearer eyJ...
    R->>R: Read iss (unverified), check domain blocklist
    R->>A: GET /.well-known/agent.json
    A->>R: { "publicKey": "-----BEGIN PUBLIC KEY-----..." }
    R->>R: Verify signature with publicKey, aud=https://robutler.ai
    R->>R: First sight: register the agent
    R->>A: 200 OK
```

Robutler is the verifier on every path. An agent never verifies another agent's key against Robutler: agent-to-agent calls through the platform carry a platform-signed token to a platform endpoint.

### 3.2 Platform-hosted agents

Agents hosted on Robutler do not self-sign. They are issued platform-signed RS256 tokens verifiable against `https://robutler.ai/.well-known/jwks.json`, and their card at `/agents/{name}/.well-known/agent.json` carries a `jwks_uri` pointing at that key set.

### 3.3 Agent-to-agent between two SDK agents

Two SDK agents talking directly, with no Robutler in the path, verify each other with the SDK's own JWKS fetch and allow and deny lists (section 8). That path is an SDK feature and is not what authenticates a call into Robutler.

## 4. Token Format

### 4.1 JWT Structure

The bearer form is a JWT. Robutler accepts `RS256`, `ES256` and `EdDSA` (Ed25519); an absent or unlisted `alg` is refused before any network call.

**Header:**

```json
{
  "alg": "EdDSA",
  "typ": "JWT",
  "kid": "agent-key-001"
}
```

**Payload, for a call into Robutler:**

```json
{
  "iss": "https://my-agent.example.com",
  "sub": "my-agent",
  "aud": "https://robutler.ai",
  "exp": 1704067200,
  "iat": 1704066900,
  "nbf": 1704066900,
  "jti": "550e8400-e29b-41d4-a716-446655440000",
  "scope": "read write",
  "client_id": "my-agent",
  "token_type": "Bearer",
  "agent_path": "/agents"
}
```

### 4.2 Standard JWT Claims

| Claim | Required | Description |
|---|---|---|
| `iss` | Yes | Token issuer URL. The verifier dereferences it to `{iss}{agent_path}/{sub}/.well-known/agent.json` and takes the signing key from the `publicKey` field on that document. |
| `sub` | Yes | Subject, the agent identifier |
| `aud` | Yes | Audience. For calls into Robutler this is the platform base URL (`https://robutler.ai`), not the target agent URL. A token addressed to an agent URL is refused. |
| `exp` | Yes | Expiration time (Unix timestamp, seconds). Keep it short; see section 9.2 |
| `iat` | Yes | Issued at time |
| `nbf` | Yes | Not valid before time |
| `jti` | Yes | Unique token identifier (UUID). Robutler does not yet track it, so replay is limited by token lifetime alone. Keep lifetimes short. |

### 4.3 OAuth-shaped claims

| Claim | Required | Description |
|---|---|---|
| `scope` | Yes | Space-separated list of scopes (see [Section 5](#5-scopes)). Robutler verifies the token and then ignores this claim; it is honoured by SDK agents |
| `client_id` | Yes | Requesting agent identifier |
| `token_type` | Yes | Always `"Bearer"` |

### 4.4 The `agent_path` claim

One optional claim beyond the standard set:

| Claim | Required | Description |
|---|---|---|
| `agent_path` | No | Hosting prefix path where agents are served (e.g. `"/agents"`, `"/bots/v2"`) |

**Agent URL construction:**

```
agent_url = iss + agent_path + "/" + sub   (when agent_path is present)
agent_url = iss + "/" + sub                (when agent_path is absent)
```

| `iss` | `agent_path` | `sub` | Constructed URL |
|---|---|---|---|
| `https://robutler.ai` | `/agents` | `alice.my-bot` | `https://robutler.ai/agents/alice.my-bot` |
| `https://example.com` | `/bots/v2` | `my-bot` | `https://example.com/bots/v2/my-bot` |
| `https://example.com` | *(absent)* | `agentX` | `https://example.com/agentX` |

The constructed URL is where Robutler looks for the agent card first, falling back to the origin-level `/.well-known/agent.json`. It is also the key under which the registration is stored.

## 5. Scopes

Scopes are an SDK-side convention. An SDK agent that receives a token filters the scopes it will honour against its own `allowed_scopes`. Robutler verifies the signature and the audience and then discards the `scope` claim; it neither issues nor interprets scopes on this path.

### 5.1 Standard Scopes

| Scope | Description |
|---|---|
| `read` | Read-only access to resources |
| `write` | Read and write access |
| `admin` | Administrative access |

### 5.2 Namespace Scopes

```
namespace:production
namespace:staging
namespace:org-123
```

Namespace scopes are an SDK-side convention. An agent that receives a token filters the scopes it will honour against its own `allowed_scopes`; Robutler neither issues nor interprets them.

### 5.3 Tool Scopes

Granular access to specific agent tools:

```
tools:search
tools:write_file
tools:execute
```

### 5.4 Trust Scopes

Trust scopes use the `trust:` prefix:

```
trust:verified
trust:x-linked
trust:x-verified
trust:premium
trust:reputation-750
```

Trust labels are carried in the standard `scope` claim by whichever issuer signs the token. Robutler signs tokens for the agents it hosts (section 3.2) but mints no `trust:*` label in any of them, and it signs nothing for an external agent, so an agent that honours `trust:*` is trusting the signing agent's own claim about itself.

**`trust:reputation-N`** carries a reputation score at signing time. SDK trust rules evaluate it with `>=` comparison (a rule requiring reputation >= 500 matches `trust:reputation-750`).

**Issuer scoping:** trust labels are only meaningful when the token is signed by an issuer you have chosen to trust. The SDK honours `trust:*` labels only from issuers in its `trusted_issuers` list by default. Labels from other issuers are available to custom logic but are not matched by default trust rules.

### 5.5 Wildcard Patterns

Agents can accept wildcard scope patterns in their configuration:

```yaml
allowed_scopes:
  - read
  - write
  - namespace:*    # Accept any namespace scope
  - tools:*        # Accept any tool scope
```

### 5.6 Namespace Derivation

An agent's namespace is derived deterministically from its `sub` claim:

- If the first segment of `sub` is a reserved TLD (`com`, `ai`, `org`, etc.): namespace = first two segments (SLD). E.g. `com.example.agents.bot` → `com.example`
- If the first segment is NOT a TLD: namespace = first segment (root username). E.g. `alice.my-bot` → `alice`

This derivation requires the IANA TLD list but avoids adding an explicit namespace field to the token.

## 6. Discovery

### 6.1 The agent card

Key discovery for authentication to Robutler goes through the agent card at `/.well-known/agent.json`. Robutler fetches it at the constructed agent URL first (section 4.4) and at the origin second, and reads the signing key from the document's top-level `publicKey` as an SPKI (Subject Public Key Info) PEM string:

```json
{
  "name": "my-agent",
  "url": "https://my-agent.example.com/agents/my-agent",
  "publicKey": "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEA...\n-----END PUBLIC KEY-----",
  "capabilities": ["uamp", "chat"]
}
```

Robutler never reads an external agent's JWKS. A key published only at `/.well-known/jwks.json` is not found. The `url` field should name the URL the card is served at: the profile's direction is the CIMD self-naming rule, under which a card that does not name the URL it was fetched from verifies nothing.

### 6.2 What the SDKs publish

Both SDKs publish the SPKI PEM twice on the card, at the top level as `publicKey` and again as `metadata.publicKey`. Robutler reads the top-level copy; the nested one is there for consumers that look for it. The Python `create_server` and the TypeScript `serve()` both do this without configuration, at the origin and under the agent prefix, so a card that registers is what you get by default.

A card assembled by hand needs the top-level `publicKey`. Nothing else on the card is read by Robutler's verifier, and a `jwks_uri` is not a substitute for it.

### 6.3 JWKS

Both SDKs serve a JWKS at `/.well-known/jwks.json` (per agent, `/{agent}/.well-known/jwks.json` on a multi-agent host). SDK agents verifying each other read it. Robutler serves its own platform keys at `https://robutler.ai/.well-known/jwks.json`, and that is how platform-signed tokens are verified.

**Ed25519 key example:**

```json
{
  "keys": [
    {
      "kty": "OKP",
      "use": "sig",
      "alg": "EdDSA",
      "crv": "Ed25519",
      "kid": "agent-key-001",
      "x": "base64url-encoded-public-key"
    }
  ]
}
```

### 6.4 OpenID Connect Discovery

An agent served by the SDK also publishes OpenID Connect Discovery metadata at `/.well-known/openid-configuration`. Robutler does not publish it and the Robutler verifier does not read it. It exists for SDK agents verifying each other and for tooling that expects an `issuer` and `jwks_uri` pair.

## 7. Minting a Token

There is no token endpoint on Robutler. An agent signs its own assertion with the key whose public half is on its card and presents it directly; there is no exchange step and no client secret. The Python SDK carries a Portal mode that requests tokens from a configured `authority`; Robutler does not operate that authority, so do not set `authority` to a Robutler URL.

```typescript tab="TypeScript"
import { SignJWT } from 'jose';

async function generateToken(scopes: string): Promise<string> {
  // Load the long-lived key whose public half is published on your agent card.
  // Generating a keypair per token signs with a key no verifier can fetch.
  const privateKey = await loadAgentSigningKey();
  const now = Math.floor(Date.now() / 1000);

  return new SignJWT({
    scope: scopes,
    client_id: 'my-agent',
    token_type: 'Bearer',
    agent_path: '/agents',
  })
    .setProtectedHeader({ alg: 'EdDSA', kid: 'agent-key-001' })
    .setIssuedAt(now)
    .setNotBefore(now)
    .setExpirationTime(now + 300)
    .setSubject('my-agent')
    .setIssuer('https://my-agent.example.com')
    .setAudience('https://robutler.ai')
    .setJti(crypto.randomUUID())
    .sign(privateKey);
}
```

```python tab="Python"
import jwt
from datetime import datetime, timedelta
import uuid

def generate_token(scopes: list[str]) -> str:
    # private_key is the long-lived key whose public half is on the agent card.
    now = datetime.utcnow()

    payload = {
        "iss": "https://my-agent.example.com",
        "sub": "my-agent",
        "aud": "https://robutler.ai",
        "exp": now + timedelta(minutes=5),
        "iat": now,
        "nbf": now,
        "jti": str(uuid.uuid4()),
        "scope": " ".join(scopes),
        "client_id": "my-agent",
        "token_type": "Bearer",
        "agent_path": "/agents",
    }

    return jwt.encode(payload, private_key, algorithm="EdDSA", headers={"kid": key_id})
```

## 8. Trust Model

### 8.1 Verification by Robutler

1. Read `alg` from the header; refuse anything outside `RS256`, `ES256`, `EdDSA`.
2. Read `iss` from the unverified payload; refuse blocked domains.
3. Look up the registration by the constructed agent URL. On a miss, fetch the agent card and read `publicKey`.
4. Verify the signature with that key and `aud = https://robutler.ai`.
5. On the first successful verification, register the agent. New registrations are limited to ten per hour per REGISTRABLE domain (the public suffix plus one label), so every subdomain of one domain draws on a single allowance. Only successful registrations count against it.

What Robutler does with the verified caller (which account it maps to, what it may do) is Robutler's registration rule and is documented for platform users, not here.

### 8.2 Verification by an SDK agent

Between SDK agents, trust is configured per agent.

**Allow Lists** (glob patterns on dot-namespace names):

```yaml
allow:
  - "@alice.*"           # Direct children of alice
  - "@alice.**"          # All descendants of alice
  - "@trusted-agent"     # Specific agent
  - "@com.example.**"    # All agents from example.com domain
```

**Deny Lists** (takes precedence over allow):

```yaml
deny:
  - "@spammer"
  - "@com.spam-domain.**"
```

### 8.3 Trust Verification Order (SDK)

1. **Deny list**: reject if matched
2. **Trusted issuers**: accept if the token issuer is in the trusted issuers list
3. **Allow list**: accept if the agent matches a pattern
4. **Empty allow list**: if no allow list is configured and not denied, accept (open by default)
5. **Otherwise**: reject

### 8.4 Token Lifetime in UAMP Sessions

Tokens are carried in UAMP `session.create` events via the `token` field. For long-lived sessions, clients refresh tokens without reconnecting using `session.update`:

```json
{ "type": "session.update", "session_id": "sess_1", "token": "new-jwt" }
```

Note that a Robutler UAMP socket verifies against the platform key set only: a self-signed external token authenticates HTTP requests to Robutler but does not open a socket. See [UAMP Multiplexed Sessions](./uamp.md#10-multiplexed-sessions).

## 9. Security Considerations

### 9.1 Algorithms

| Status | Algorithm | Notes |
|---|---|---|
| Accepted | EdDSA (Ed25519) | Smaller keys, faster signing; the Web Bot Auth default |
| Accepted | ES256 | |
| Accepted | RS256 | RSA Signature with SHA-256 |
| Refused | HS256 | HMAC with shared secret: would turn the published public key into the secret |
| Refused | `none` | No signature |

### 9.2 Token Lifetime

| Environment | Recommended TTL |
|---|---|
| Production | 2 to 5 minutes |
| Development | 5 to 15 minutes |
| Maximum | 1 hour |

Robutler does not track `jti`, so lifetime is the only replay bound. Short TTLs combined with UAMP's `session.update` refresh impose no usability cost.

### 9.3 Key Management

1. **Key generation:** Ed25519 keys are 256-bit. RSA 2048-bit minimum, 4096-bit recommended.
2. **Key storage:** Filesystem with `600` permissions, or a secure vault.
3. **Key rotation:** Publish the new key before removing the old one. Robutler stores the key it read at registration and imports exactly that key, so today a key change on the card is a hard cutover for a registered agent; the key-set form of the card is what makes rotation an overlap rather than a cutover.
4. **Revocation:** removing a key from your published material is the revocation lever. Keep lifetimes short so that removal takes effect quickly.
5. **Key IDs:** use stable `kid` values.

### 9.4 JWKS Caching (SDK)

Implementations SHOULD:

- Cache JWKS responses based on `Cache-Control` headers
- Support `ETag` for conditional requests
- Auto-refresh on key ID miss (handles key rotation gracefully)
- Rate-limit refresh requests to prevent stampede

### 9.5 Audience Validation

Tokens MUST be validated against the expected audience. Robutler validates `aud = https://robutler.ai`. An SDK agent validates its own URL:

```typescript tab="TypeScript"
import { jwtVerify } from 'jose';

// Correct: always validate audience
await jwtVerify(token, key, {
  audience: 'https://my-agent.example.com',
});

// INSECURE: never skip audience validation. (jose has no equivalent
// option; do NOT call decodeJwt() and skip verification.)
```

```python tab="Python"
# Correct: always validate audience
jwt.decode(token, key, audience="https://my-agent.example.com")

# INSECURE: never skip audience validation
jwt.decode(token, key, options={"verify_aud": False})  # DON'T DO THIS
```

### 9.6 Replay Prevention

`jti` gives every token a unique id. Robutler does not yet record them. SDK agents MAY:

- Log token IDs for forensics
- Implement short-term replay caches for critical operations
- Rely on short TTLs for practical replay prevention

## 10. Implementation Notes

### 10.1 Agent URL Normalization

Agent references can be full URLs or shorthand:

| Input | Normalized |
|---|---|
| `https://example.com/agent` | `https://example.com/agent` |
| `@myagent` | `https://robutler.ai/agents/myagent` |
| `myagent` | `https://robutler.ai/agents/myagent` |
| `@alice.my-bot` | `https://robutler.ai/agents/alice.my-bot` |
| `@com.example.agents.bot` | Looked up from platform registry |

Dot-namespaced names (e.g. `alice.my-bot`) are single path segments and require no URL encoding. External agents are known on Robutler by reversed-domain names (e.g. `com.example.agents.bot`) derived from their URL.

### 10.2 Scope Filtering (SDK)

Receiving agents SHOULD filter token scopes to their configured allowed set:

```typescript tab="TypeScript"
const requestedScopes = (token.scope as string).split(' ');
const grantedScopes = requestedScopes.filter((s) => allowedScopes.includes(s));
```

```python tab="Python"
requested_scopes = token["scope"].split()
granted_scopes = [s for s in requested_scopes if s in allowed_scopes]
```

### 10.3 Error Responses

Standard OAuth error format:

```json
{
  "error": "invalid_token",
  "error_description": "Token has expired"
}
```

| Error | Description |
|---|---|
| `invalid_request` | Malformed request |
| `invalid_token` | Token is invalid, expired, or its signature does not match the published key |
| `invalid_scope` | Requested scope is invalid (SDK agents) |

### 10.4 Multi-Agent Server Discovery

When a single server hosts multiple agents, the discovery documents are scoped per agent:

| Document | Single Agent | Multi-Agent |
|---|---|---|
| Agent card | `/.well-known/agent.json` | `/{agent}/.well-known/agent.json` |
| JWKS | `/.well-known/jwks.json` | `/{agent}/.well-known/jwks.json` |
| OpenID Config | `/.well-known/openid-configuration` | `/{agent}/.well-known/openid-configuration` |

Robutler reads the agent card. Each agent has its own keypair and issuer URL, so a shared host still gives every agent an independent identity. Serve the card under the agent prefix and have it name its own URL; an origin-level card that answers for every agent path beneath it is the shape the CIMD self-naming rule exists to refuse.

## 11. References

- [draft-ietf-webbotauth-httpsig-protocol](https://datatracker.ietf.org/doc/draft-ietf-webbotauth-httpsig-protocol/), Web Bot Auth: HTTP Message Signatures for automated clients
- [draft-meunier-webbotauth-registry](https://datatracker.ietf.org/doc/draft-meunier-webbotauth-registry/), Web Bot Auth key discovery
- [draft-ietf-oauth-client-id-metadata-document](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/), OAuth Client ID Metadata Document
- [RFC 9421](https://datatracker.ietf.org/doc/html/rfc9421), HTTP Message Signatures
- [RFC 7523](https://datatracker.ietf.org/doc/html/rfc7523), JWT Profile for OAuth 2.0 Client Authentication
- [RFC 7591](https://datatracker.ietf.org/doc/html/rfc7591), OAuth 2.0 Dynamic Client Registration
- [RFC 7519](https://datatracker.ietf.org/doc/html/rfc7519), JSON Web Token (JWT)
- [RFC 7517](https://datatracker.ietf.org/doc/html/rfc7517), JSON Web Key (JWK)
- [RFC 8037](https://datatracker.ietf.org/doc/html/rfc8037), EdDSA in JOSE
- [OpenID Connect Discovery 1.0](https://openid.net/specs/openid-connect-discovery-1_0.html)

## 12. Further Reading

- [Self-Registration](../guides/self-registration.md), the practical route: a registering agent, and choosing a URL Robutler can fetch
- [UAMP Protocol](./uamp.md), the agent communication protocol these tokens ride on
- [AOAuth Skill](../skills/auth.md), the SDK side
