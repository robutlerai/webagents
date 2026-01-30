---
name: aoauth-basic
version: 1.0
transport: completions
tags: [auth, aoauth]
---

# AOAuth Authentication

Tests Agent OAuth (AOAuth) authentication between agents.

## Setup

### Agent: protected-agent
- Name: `protected-agent`
- Instructions: "Provide sensitive data only to authenticated callers."
- Skills: [AuthSkill]
- Auth Config:
  - Mode: validate
  - Required scopes: [read]

### Agent: client-agent
- Name: `client-agent`
- Instructions: "Call protected-agent to get data."
- Skills: [AuthSkill]
- Auth Config:
  - Mode: self-issued
  - Scopes: [read, write]

## Test Cases

### 1. Unauthenticated Request Rejected

**Request:**
POST `/agents/protected-agent/chat/completions`

No Authorization header.

```json
{
  "messages": [{"role": "user", "content": "Give me the data"}]
}
```

**Assertions:**
- Response status is 401 Unauthorized
- Error message indicates authentication required
- WWW-Authenticate header may be present

**Strict:**
```yaml
status: 401
body:
  error: exists
  error.type: authentication_error
```

### 2. Valid Token Accepted

**Request:**
POST `/agents/protected-agent/chat/completions`
Authorization: Bearer {valid_jwt_token}

```json
{
  "messages": [{"role": "user", "content": "Give me the data"}]
}
```

**Assertions:**
- Response status is 200
- Request was processed successfully
- Agent provided the requested data

**Strict:**
```yaml
status: 200
body:
  choices[0].message.role: assistant
```

### 3. Invalid Token Rejected

**Request:**
POST `/agents/protected-agent/chat/completions`
Authorization: Bearer invalid.token.here

```json
{
  "messages": [{"role": "user", "content": "Give me the data"}]
}
```

**Assertions:**
- Response status is 401 Unauthorized
- Error indicates invalid token
- Token validation failed

**Strict:**
```yaml
status: 401
body:
  error.type: [authentication_error, invalid_token]
```

### 4. Expired Token Rejected

**Request:**
POST `/agents/protected-agent/chat/completions`
Authorization: Bearer {expired_jwt_token}

```json
{
  "messages": [{"role": "user", "content": "Give me the data"}]
}
```

**Assertions:**
- Response status is 401 Unauthorized
- Error indicates token expired
- Client should refresh token

**Strict:**
```yaml
status: 401
body:
  error.code: [token_expired, invalid_token]
```

### 5. Insufficient Scope Rejected

**Setup:**
Token has only `read` scope, but endpoint requires `admin`.

**Request:**
POST `/agents/protected-agent/admin/action`
Authorization: Bearer {token_with_read_scope}

```json
{
  "action": "delete_all"
}
```

**Assertions:**
- Response status is 403 Forbidden
- Error indicates insufficient permissions
- Required scope is mentioned

**Strict:**
```yaml
status: 403
body:
  error.type: [forbidden, insufficient_scope]
```

### 6. JWKS Endpoint

**Request:**
GET `/agents/protected-agent/.well-known/jwks.json`

**Assertions:**
- Response status is 200
- Response contains keys array
- Keys have required JWK fields
- At least one key is present

**Strict:**
```yaml
status: 200
body:
  keys: exists
  keys[0].kty: type(string)
  keys[0].kid: type(string)
  keys[0].use: sig
```

### 7. OpenID Configuration

**Request:**
GET `/agents/protected-agent/.well-known/openid-configuration`

**Assertions:**
- Response status is 200
- Contains issuer field
- Contains jwks_uri field
- Contains token_endpoint

**Strict:**
```yaml
status: 200
body:
  issuer: type(string)
  jwks_uri: type(string)
  token_endpoint: type(string)
```

### 8. Token Generation

**Request:**
POST `/agents/client-agent/auth/token`
Content-Type: application/x-www-form-urlencoded

```
grant_type=client_credentials&scope=read
```

**Assertions:**
- Response status is 200
- Response contains access_token
- Token is valid JWT format
- token_type is "Bearer"
- expires_in is present

**Strict:**
```yaml
status: 200
body:
  access_token: type(string)
  token_type: Bearer
  expires_in: type(number)
```

### 9. Agent-to-Agent Auth Flow

**Flow:**
1. client-agent requests token from its AuthSkill
2. client-agent calls protected-agent with bearer token
3. protected-agent validates token via JWKS
4. protected-agent processes request

**Assertions:**
- Token was generated successfully
- Request included Authorization header
- protected-agent accepted the token
- Response was successful

**Strict:**
```yaml
events:
  - type: auth.token_generated
    agent: client-agent
  - type: auth.token_validated
    agent: protected-agent
response:
  status: 200
```
