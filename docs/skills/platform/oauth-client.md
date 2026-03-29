---
title: OAuth Client Skill
---

# OAuth Client Skill

> [!NOTE]
> This skill is under active development. The architecture and interfaces described here reflect the planned implementation.

Connect your agent to any OAuth2-protected API. The OAuth Client skill handles the full authorization flow — token acquisition, refresh, and secure storage — so your agent can work with external services programmatically.

## Overview

Most web APIs require OAuth2 authentication. Instead of writing custom auth code for each service, the OAuth Client skill provides a generic OAuth2 client that works with any compliant provider. Combined with the [OpenAPI skill](./openapi.md), your agent can connect to and operate any REST API.

### How It Works

1. Configure the OAuth provider (authorization URL, token URL, client credentials, scopes)
2. The skill handles the authorization flow (redirect-based or client credentials)
3. Tokens are stored securely in the agent's [memory](./memory.md) (encrypted, not visible to LLM)
4. When your agent calls an API, the skill injects the Bearer token automatically
5. Token refresh happens transparently

## Configuration

```python
from webagents import BaseAgent
from webagents.agents.skills.platform.oauth_client import OAuthClientSkill

github_oauth = OAuthClientSkill({
    "provider": "github",
    "authorization_url": "https://github.com/login/oauth/authorize",
    "token_url": "https://github.com/login/oauth/access_token",
    "client_id": "${GITHUB_CLIENT_ID}",
    "client_secret": "${GITHUB_CLIENT_SECRET}",
    "scopes": ["repo", "read:user"],
})

agent = BaseAgent(
    name="dev-assistant",
    model="openai/gpt-4o",
    skills={
        "github_auth": github_oauth,
    },
)
```

### Portal Mode

When running on the Robutler platform, the OAuth Client skill uses the portal's provider registry — 50+ pre-configured providers (GitHub, Slack, Google, Stripe, Salesforce, and more). The agent owner authorizes via the portal UI and the skill receives tokens through the platform's secure token relay.

### Self-Hosted Mode

For self-hosted agents, provide the full OAuth configuration. The skill manages the authorization redirect, callback handling (via an `@http` endpoint on the agent), and secure token storage in local memory.

## Tools

The skill registers tools for managing OAuth connections:

| Tool | Scope | Description |
|------|-------|-------------|
| `oauth_connect` | `owner` | Initiate authorization flow for a provider |
| `oauth_status` | `owner` | Check connection status and token validity |
| `oauth_disconnect` | `owner` | Revoke tokens and remove connection |

## Combining with OpenAPI

The OAuth Client skill + [OpenAPI skill](./openapi.md) is a powerful combination. Point your agent at an API spec, configure OAuth credentials, and your agent can operate the entire API:

```python
agent = BaseAgent(
    name="github-agent",
    model="openai/gpt-4o",
    skills={
        "github_auth": OAuthClientSkill({...}),
        "github_api": OpenAPISkill({
            "spec_url": "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json",
            "auth_skill": "github_auth",
        }),
    },
)
```

The agent now has tools for every GitHub API endpoint, authenticated automatically.

## See Also

- [OpenAPI Skill](./openapi.md) — Auto-generate tools from API specs
- [AOAuth](../auth.md) — Agent-to-agent authentication
- [Memory](./memory.md) — Secure token storage
- [Connected Accounts](/docs/guides/connected-accounts) — Portal OAuth provider management
