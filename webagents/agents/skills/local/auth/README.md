# AOAuth: Agent OAuth Protocol

OAuth 2.0 extension for agent-to-agent authentication in the WebAgents framework.

## Overview

AOAuth (Agent OAuth) extends OAuth 2.0 to support:

1. **Portal Mode**: Centralized token issuance via Robutler Portal with namespace scopes
2. **Self-Issued Mode**: Decentralized authentication where each agent is its own IdP

## Quick Start

### Portal Mode (Production)

```yaml
skills:
  auth:
    authority: "https://robutler.ai"
    agent_id: "my-agent"
    allowed_scopes:
      - read
      - write
      - namespace:*
```

### Self-Issued Mode (Development)

```yaml
skills:
  auth:
    base_url: "@my-local-agent"
    allowed_scopes:
      - read
      - write
    allow:
      - "@myteam/*"
      - "@trusted-agent"
```

## Features

### Token Generation

```python
# Generate token for another agent
token = auth_skill.generate_token("@target-agent", ["read", "write"])
```

### Token Validation

```python
# Validate incoming token
auth_context = await auth_skill.validate_token(token)
if auth_context and auth_context.authenticated:
    print(f"Authenticated: {auth_context.agent_id}")
    print(f"Scopes: {auth_context.scopes}")
    print(f"Namespaces: {auth_context.namespaces}")
```

### Automatic Token Injection

The `on_request_outgoing` hook automatically injects Bearer tokens into outgoing agent requests.

### Automatic Token Validation

The `on_connection` hook automatically validates incoming Bearer tokens and attaches `AuthContext` to the request.

## OAuth Endpoints

When initialized with an HTTP server, AuthSkill exposes:

| Endpoint | Description |
|----------|-------------|
| `/.well-known/openid-configuration` | OpenID Connect Discovery |
| `/.well-known/jwks.json` | JSON Web Key Set |
| `/auth/token` | Token endpoint |

## Commands

| Command | Description |
|---------|-------------|
| `/auth` | Show AOAuth status and configuration |
| `/auth/token <target>` | Generate token for target agent |
| `/auth/validate <token>` | Validate a JWT token |
| `/auth/jwks` | Show JWKS cache statistics |

## JWT Structure

Tokens include standard OAuth claims plus AOAuth extensions:

```json
{
  "iss": "https://robutler.ai",
  "sub": "agent-a",
  "aud": "https://robutler.ai/agents/agent-b",
  "exp": 1234567890,
  "iat": 1234567890,
  "jti": "unique-token-id",
  "scope": "read write namespace:production",
  "client_id": "agent-a",
  "token_type": "Bearer",
  "aoauth": {
    "mode": "portal",
    "agent_url": "https://robutler.ai/agents/agent-a"
  }
}
```

## Scope Format

Scopes follow OAuth standards (space-separated):

- `read`, `write`, `admin` - Basic permissions
- `namespace:production` - Portal-assigned namespace membership
- `tools:search` - Tool-specific access

## Trust Model

### Portal Mode

- Portal signs all tokens
- Portal JWKS validates signatures
- Portal assigns namespace scopes
- Centralized trust management

### Self-Issued Mode

- Each agent signs own tokens
- Agents publish JWKS at `/.well-known/jwks.json`
- Trust via allow/deny lists with glob patterns
- Decentralized but configurable trust

## Configuration Reference

```yaml
skills:
  auth:
    # Operating Mode
    authority: "https://robutler.ai"  # Set for Portal mode, omit for self-issued
    
    # Agent Identity
    agent_id: "my-agent"              # Unique agent identifier
    base_url: "@my-agent"             # Agent URL (or @name for normalization)
    
    # Token Settings
    token_ttl: 300                    # Token lifetime in seconds
    
    # Scope Control
    allowed_scopes:                   # Scopes this agent accepts
      - read
      - write
      - namespace:*                   # Wildcard for all namespace scopes
      - tools:*                       # Wildcard for all tool scopes
    
    # Trust Configuration
    trusted_issuers:                  # Explicit trusted issuers
      - issuer: "https://partner.ai"
        jwks_uri: "https://partner.ai/.well-known/jwks.json"
        type: "agent"
    
    allow:                            # Allow list (glob patterns)
      - "@myteam/*"
      - "@trusted-agent"
    
    deny:                             # Deny list (takes precedence)
      - "@banned-*"
    
    # OAuth Providers
    google:
      client_id: "${GOOGLE_CLIENT_ID}"
      client_secret: "${GOOGLE_CLIENT_SECRET}"
      hosted_domain: "company.com"    # Optional G Suite restriction
    
    robutler:
      client_id: "my-agent"
      client_secret: "${ROBUTLER_SECRET}"
    
    # Key Management
    keys_dir: "~/.webagents/keys"     # RSA key storage
    jwks_cache_ttl: 3600              # JWKS cache lifetime
```

## JWKS Caching

The JWKSManager provides smart caching:

- **TTL-based expiration**: Respects Cache-Control headers
- **ETag support**: Conditional requests for efficiency
- **Auto-refresh on miss**: Handles key rotation gracefully
- **Rate limiting**: Prevents cache stampede

## Dependencies

```
PyJWT>=2.8
cryptography>=41.0
httpx>=0.25
```

## Security Considerations

1. **Key Storage**: RSA keys are stored in `~/.webagents/keys/` by default. Ensure proper file permissions.

2. **Token TTL**: Default 5 minutes. Adjust based on your security requirements.

3. **Allow/Deny Lists**: Use specific patterns. Empty allow list means "allow all non-denied".

4. **Portal Mode**: Recommended for production. Centralizes trust and provides namespace-based access control.

5. **Self-Issued Mode**: Suitable for development and trusted environments.

## Protocol Comparison

| Component | OAuth 2.0 | AOAuth Extension |
|-----------|-----------|------------------|
| Token issuer | Authorization Server | Portal OR self-issued |
| Identity proof | AS vouches | JWKS proves identity |
| Token endpoint | Required | Optional (self-issued fallback) |
| Trust model | Trust AS | Trust Portal OR allow list |
| Scopes | Granted by AS | `namespace:` from Portal OR declared |
