---
title: OpenAPI Skill
description: Auto-generate agent tools from any OpenAPI 3.x specification.
---

# OpenAPI Skill

> [!NOTE]
> Both Python and TypeScript ship the OpenAPI skill, but the configuration shapes differ slightly. The TypeScript variant takes a `servers` map (one entry per spec) so a single agent can connect to multiple OpenAPI services; the Python variant takes a single spec per skill instance. Track parity at [internal/python-typescript-parity.md](../../internal/python-typescript-parity.md).

Point your agent at any OpenAPI (Swagger) specification and it auto-generates tools for every endpoint. No custom code per API — the spec is the integration.

## Overview

The OpenAPI skill parses an OpenAPI 3.x specification and registers one tool per endpoint. Each tool handles request construction, parameter validation, and response parsing. Combined with the [OAuth Client skill](./oauth-client.md), any authenticated REST API becomes agent-native.

## Configuration

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { OpenAPISkill } from 'webagents/skills/openapi';

const agent = new BaseAgent({
  name: 'api-agent',
  model: 'openai/gpt-4o',
  skills: [
    new OpenAPISkill({
      servers: {
        stripe: {
          specUrl: 'https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.json',
          auth: { type: 'bearer', token: process.env.STRIPE_API_KEY! },
          operations: ['listCustomers', 'createPaymentIntent', 'getBalance'],
        },
      },
    }),
  ],
});
```

```python tab="Python"
from webagents import BaseAgent
from webagents.agents.skills.platform.openapi import OpenAPISkill

agent = BaseAgent(
    name="api-agent",
    model="openai/gpt-4o",
    skills={
        "stripe_api": OpenAPISkill({
            "spec_url": "https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.json",
            "auth_skill": "stripe_auth",
            "operations": ["listCustomers", "createPaymentIntent", "getBalance"],
        }),
    },
)
```

### Parameters

| Python field | TypeScript field | Required | Description |
|--------------|------------------|----------|-------------|
| `spec_url` | `servers[name].specUrl` | Yes | URL or local path to an OpenAPI 3.x spec (JSON or YAML) |
| `spec` | _(coming soon)_ | No | Inline spec object (alternative to `spec_url`) |
| `auth_skill` | `servers[name].auth` | No | Authentication source — Python references another skill, TS takes a token directly |
| `base_url` | `servers[name].baseUrl` | No | Override the server base URL from the spec |
| `operations` | `servers[name].operations` | No | Allowlist of operation IDs to register (default: all) |
| `exclude` | _(use `operations` allowlist)_ | No | Denylist of operation IDs to skip |
| `scope` | _(coming soon)_ | No | Default access scope for generated tools (default: `"all"`) |
| _(n/a)_ | `servers[name].operationPolicies` | No | TS only — `'allow' \| 'notify' \| 'block'` per operation, paired with `policyHook` |

## Generated Tools

Each API endpoint becomes a tool with:

- **Name** derived from the `operationId` (or `method_path` if no operationId)
- **Description** from the endpoint's `summary` and `description`
- **Parameters** from path params, query params, and request body schema
- **Return type** based on the response schema

The LLM sees these as standard tools — it doesn't need to know they map to HTTP calls.

### Filtering Operations

Large specs (Stripe has 300+ endpoints) can overwhelm the LLM's context. Use `operations` (allowlist) or `exclude` (Python only) to control which endpoints become tools:

```typescript tab="TypeScript"
new OpenAPISkill({
  servers: {
    example: {
      specUrl: 'https://api.example.com/openapi.json',
      operations: ['listUsers', 'createUser', 'getUser'],
    },
  },
});
```

```python tab="Python"
OpenAPISkill({
    "spec_url": "https://api.example.com/openapi.json",
    "operations": ["listUsers", "createUser", "getUser"],
})
```

## Authentication

The skill supports three authentication modes:

1. **OAuth Client skill** *(Python only)* — Reference another skill by name for automatic token injection.
2. **API key / Bearer** — Static key injected as a header or query parameter.
3. **None** — For public APIs.

```typescript tab="TypeScript"
new OpenAPISkill({
  servers: {
    example: {
      specUrl: 'https://api.example.com/openapi.json',
      auth: { type: 'api_key', token: process.env.API_KEY!, headerName: 'X-API-Key' },
    },
  },
});
```

```python tab="Python"
OpenAPISkill({
    "spec_url": "https://api.example.com/openapi.json",
    "auth": {"type": "api_key", "header": "X-API-Key", "value": "${API_KEY}"},
})
```

## Pricing Generated Tools

Apply pricing to auto-generated tools to monetize API access:

```typescript tab="TypeScript"
// Coming soon — track at https://github.com/robutlerai/webagents/issues
// In TypeScript, pricing for generated OpenAPI tools is configured at
// the agent level via the PaymentSkill rather than per-spec. See
// ../platform/payments.md for the current pricing model.
```

```python tab="Python"
OpenAPISkill({
    "spec_url": "https://api.example.com/openapi.json",
    "pricing": {
        "default": {"credits_per_call": 0.1},
        "createPaymentIntent": {"credits_per_call": 1.0},
    },
})
```

## See Also

- [OAuth Client Skill](./oauth-client.md) — Authenticate with any OAuth API
- [MCP Skill](../core/mcp.md) — Alternative integration via MCP tool servers
- [Tools](../../agent/tools.md) — How tools work in WebAgents
