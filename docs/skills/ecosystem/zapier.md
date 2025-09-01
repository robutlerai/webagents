# Zapier Skill

Minimalistic Zapier integration for workflow automation. Trigger Zaps, monitor executions, and automate tasks across 7,000+ supported applications.

## Features

- **Secure API key storage** via auth and KV skills
- **Trigger Zaps** with custom input data
- **List Zaps** from your Zapier account
- **Monitor task status** in real-time
- **7,000+ app integrations** via Zapier ecosystem

## Quick Setup

```python
from webagents.agents import BaseAgent
from webagents.agents.skills.ecosystem.zapier import ZapierSkill

agent = BaseAgent(
    name="zapier-agent",
    model="openai/gpt-4o",
    skills={
        "zapier": ZapierSkill()  # Auto-resolves: auth, kv
    }
)
```

## Core Tools

### `zapier_setup(api_key)`
Set up Zapier API credentials with automatic validation.

### `zapier_trigger(zap_id, data)`
Trigger a Zapier Zap with optional input data.

### `zapier_list_zaps()`
List all available Zaps in your Zapier account.

### `zapier_status(task_id)`
Check the status of a Zap execution.

## Usage Example

```python
# Setup, list Zaps, trigger, and check status
messages = [
    {"role": "user", "content": "Set up Zapier with API key AK_your_key, list my Zaps, then trigger lead processing Zap for John Smith"}
]
response = await agent.run(messages=messages)
```

## Getting Your Zapier API Key

1. Log into [zapier.com](https://zapier.com)
2. Go to Account Settings > Developer > Manage API Keys
3. Create API Key and copy it

## Troubleshooting

**Authentication Issues** - Verify API key is correct and has required permissions
**Zap Execution Problems** - Ensure Zap is enabled and properly configured
**Rate Limiting** - Respect 1 request/second limit and monitor task usage