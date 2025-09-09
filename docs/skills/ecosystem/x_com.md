# X.com (Twitter) Skill

!!! warning "Alpha Software Notice"

    This skill is in **alpha stage** and under active development. APIs, features, and functionality may change without notice. Use with caution in production environments and expect potential breaking changes in future releases.

Ultra-minimal X.com integration for multitenant applications with OAuth 1.0a authentication, user subscriptions, and real-time notifications.

## Features

- **OAuth 1.0a User Context** - Per-user rate limits (900 vs 450 requests/15min)
- **Post tweets** with automatic authentication
- **Subscribe to users** and monitor their posts
- **Smart notifications** via notification skill integration  
- **LLM-powered relevance** checking for subscribed posts
- **Real-time webhooks** for instant post monitoring

## Quick Setup

```python
from webagents.agents import BaseAgent
from webagents.agents.skills.ecosystem.x_com import XComSkill

agent = BaseAgent(
    name="x-agent",
    model="openai/gpt-4o",
    skills={
        "x_com": XComSkill()  # Auto-resolves: auth, kv, notifications
    }
)
```

Environment setup:
```bash
export X_API_KEY="your_api_key"
export X_API_SECRET="your_api_secret"
export AGENTS_BASE_URL="https://your-agent-domain.com"
```

## Core Tools

### `x_subscribe(username, instructions)`
Subscribe to an X.com user with automatic authentication and notification setup.

### `x_post(text)`
Post a tweet with automatic authentication handling.

### `x_manage(action, username)`
Manage X.com subscriptions (list or unsubscribe).

## Usage Example

```python
# Subscribe, post, and manage in one conversation
messages = [
    {"role": "user", "content": "Subscribe to @openai for AI research updates, then post 'Hello from my WebAgent! 🤖'"}
]
response = await agent.run(messages=messages)
```

## Authentication Flow

OAuth 1.0a handled automatically: First use provides authorization URL → User grants permission → Credentials stored securely → Future tools work seamlessly

## Troubleshooting

**"Authentication required"** - Ensure auth skill is configured and user is authenticated
**"API credentials not configured"** - Set `X_API_KEY` and `X_API_SECRET` environment variables
**"User not found"** - Check username spelling and verify account exists
