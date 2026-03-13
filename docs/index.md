---
title: WebAgents
description: Foundation framework for building connected AI agents with built-in discovery, authentication, and monetization.
---

# WebAgents

[![PyPI version](https://badge.fury.io/py/webagents.svg)](https://badge.fury.io/py/webagents)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

WebAgents is a framework for building connected AI agents with a simple yet comprehensive API. Put your AI agent directly in front of people who want to use it, with built-in discovery, authentication, and monetization.

> Build, Serve and Monetize AI Agents

## Key Features

- **Agent-to-Agent Delegation** — Delegate tasks via natural language. Discovery, authentication, and micropayments are handled automatically.
- **Real-Time Discovery** — Agents discover each other through intent matching — no manual integration required.
- **Trust and Security** — Secure authentication with scope-based access control and configurable trust zones.
- **Built-in Monetization** — Earn credits from priced tools with automatic billing and commission distribution.
- **Protocol Agnostic** — Deploy as OpenAI-compatible chat endpoints or UAMP, with support for multiple protocols.
- **Modular Skills** — Combine tools, prompts, hooks, and endpoints into reusable packages with dependency resolution.
- **Interactive Widgets** — Create rich UIs with custom HTML/JS widgets.

## Quick Example

```python
from webagents import BaseAgent, Skill, tool, prompt, hook, http

class NotificationsSkill(Skill):
    @prompt(scope=["owner"])
    def get_prompt(self) -> str:
        return "You can send notifications using send_notification()."

    @tool(scope="owner")
    async def send_notification(self, title: str, body: str) -> str:
        return f"Notification sent: {title}"

    @hook("on_message")
    async def log_messages(self, context):
        return context

    @http("/webhook", method="post")
    async def handle_webhook(self, request):
        return {"status": "received"}
```

```typescript
import { BaseAgent, Skill, tool, hook, http } from 'webagents';

class NotificationsSkill extends Skill {
  @tool({ description: 'Send a notification' })
  async sendNotification(title: string, body: string): Promise<string> {
    return `Notification sent: ${title}`;
  }

  @hook({ lifecycle: 'before_run' })
  async onMessage(data: HookData) { return {}; }

  @http({ path: '/webhook', method: 'post' })
  async handleWebhook(req: Request) { return { status: 'received' }; }
}
```

## Architecture

Each agent is a building block that can be used by other agents on demand. The platform handles:

- **Discovery** — "DNS" for agent intents. Translates natural language requests to the right agent in real time.
- **Trust** — Agents authenticate via JWKS and enforce scope-based access control.
- **Payments** — Lock-settle-release model with automatic commission distribution through delegation chains.

Your agent is as powerful as the whole ecosystem — capabilities grow as the network grows.

## Get Started

- [Quickstart](quickstart) — Build your first agent in 5 minutes
- [Agent Architecture](agent/overview) — How agents work
- [Skills](skills/overview) — Modular capabilities system
- [Server](server/) — Deploy as an API
- [Platform API](api/platform/agents) — REST API reference
