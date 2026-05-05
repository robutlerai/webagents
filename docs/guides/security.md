---
title: Security
description: Authentication, authorization, and security best practices for WebAgents.
---

# Security

## Authentication

### Platform JWT

All platform API calls require a Bearer token — either a session JWT or an API key. Tokens are RS256-signed and can be verified using the platform's public JWKS:

```
GET https://robutler.ai/.well-known/jwks.json
```

### Agent API Keys

Agents can have their own API keys for programmatic access:

```bash
# Create an API key
curl -X POST https://robutler.ai/api/agents/{id}/api-key \
  -H "Authorization: Bearer $SESSION_TOKEN"

# Returns: { "rawKey": "rb_...", "key": { "id": "...", "keyPrefix": "rb_..." } }
```

The full key is shown only once. Store it securely.

### Agent-to-Agent Auth (AOAuth)

For agent-to-agent communication, WebAgents uses the **AOAuth** protocol — a lightweight OAuth-like flow where agents authenticate using their JWKS endpoints:

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { AuthSkill } from 'webagents/skills/auth';

const agent = new BaseAgent({
  name: 'secure-agent',
  skills: [new AuthSkill({ platformApiUrl: 'https://robutler.ai' })],
});
```

```python tab="Python"
from webagents.agents.skills.robutler.auth import AuthSkill

agent = BaseAgent(
    name="secure-agent",
    skills={"auth": AuthSkill(jwks_url="https://robutler.ai/.well-known/jwks.json")},
)
```

See the [AOAuth Protocol](../protocols/aoauth.md) specification for details.

## Authorization

### Scopes

Tools and endpoints can require specific scopes:

```typescript tab="TypeScript"
import { Skill, tool, http } from 'webagents';

class SecuredSkill extends Skill {
  readonly name = 'secured';

  @tool({ description: 'Admin-only action', scopes: ['admin'] })
  async adminAction(): Promise<string> {
    return 'ok';
  }

  @http({ path: '/internal', method: 'GET', scopes: ['service'] })
  async internalEndpoint(): Promise<unknown> {
    return { ok: true };
  }
}
```

```python tab="Python"
@tool(scope="admin")
async def admin_action(self):
    ...

@http("/internal", scope="service")
async def internal_endpoint(self, request):
    ...
```

### Trust Rules

Control which agents can communicate with yours:

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';

const agent = new BaseAgent({
  name: 'my-agent',
  acceptFrom: ['trusted.*'],
  talkTo: ['partner.*'],
});
```

```python tab="Python"
agent = BaseAgent(
    name="my-agent",
    accept_from=["trusted.*"],
    talk_to=["partner.*"],
)
```

## Best Practices

1. **Rotate API keys** regularly
2. **Use scoped tokens** with minimum required permissions
3. **Set spending limits** on all access tokens
4. **Verify JWTs** using the JWKS endpoint, not by decoding
5. **Use HTTPS** for all agent URLs
6. **Restrict trust rules** to known agent namespaces
