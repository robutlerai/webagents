---
title: OpenAPI Skill
---

# OpenAPI Skill

> [!NOTE]
> This skill is under active development. The architecture and interfaces described here reflect the planned implementation.

Point your agent at any OpenAPI (Swagger) specification and it auto-generates tools for every endpoint. No custom code per API — the spec is the integration.

## Overview

The OpenAPI skill parses an OpenAPI 3.x specification and registers one tool per endpoint. Each tool handles request construction, parameter validation, and response parsing. Combined with the [OAuth Client skill](./oauth-client.md), any authenticated REST API becomes agent-native.

## Configuration

```python
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

| Parameter | Required | Description |
|-----------|----------|-------------|
| `spec_url` | Yes | URL or local path to an OpenAPI 3.x spec (JSON or YAML) |
| `spec` | No | Inline spec object (alternative to `spec_url`) |
| `auth_skill` | No | Name of an OAuth Client skill to use for authentication |
| `base_url` | No | Override the server base URL from the spec |
| `operations` | No | Allowlist of operation IDs to register (default: all) |
| `exclude` | No | Denylist of operation IDs to skip |
| `scope` | No | Default access scope for generated tools (default: `"all"`) |

## Generated Tools

Each API endpoint becomes a tool with:

- **Name** derived from the `operationId` (or `method_path` if no operationId)
- **Description** from the endpoint's `summary` and `description`
- **Parameters** from path params, query params, and request body schema
- **Return type** based on the response schema

The LLM sees these as standard tools — it doesn't need to know they map to HTTP calls.

### Filtering Operations

Large specs (Stripe has 300+ endpoints) can overwhelm the LLM's context. Use `operations` or `exclude` to control which endpoints become tools:

```python
OpenAPISkill({
    "spec_url": "https://api.example.com/openapi.json",
    "operations": ["listUsers", "createUser", "getUser"],
})
```

## Authentication

The skill supports three authentication modes:

1. **OAuth Client skill** — Reference another skill by name for automatic token injection
2. **API key** — Static key injected as a header or query parameter
3. **None** — For public APIs

```python
# API key auth
OpenAPISkill({
    "spec_url": "https://api.example.com/openapi.json",
    "auth": {"type": "api_key", "header": "X-API-Key", "value": "${API_KEY}"},
})
```

## Pricing Generated Tools

Apply pricing to auto-generated tools to monetize API access:

```python
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
