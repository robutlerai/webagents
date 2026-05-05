---
title: NLI Skill (Natural Language Interface)
description: Agent-to-agent communication over HTTP / UAMP with budgeting, retries, and trust enforcement.
---

# NLI Skill (Natural Language Interface)

Natural Language Interface skill for agent-to-agent communication.

NLI lets agents collaborate over HTTP using natural language. It adds resilient request/response primitives and optional budgeting controls (authorization caps) so one agent can safely call another.

## Features
- HTTP-based communication with other Robutler agents
- Authorization limits and cost tracking
- Communication history and success rate tracking
- Automatic timeout and retry handling
- Agent endpoint discovery and management

## Configuration
- `timeout`, `max_retries`
- `default_authorization`, `max_authorization` (optional budgeting)
- `portal_base_url` (optional for resolving agents)

## Example: Add NLI Skill to an Agent

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { NLISkill } from 'webagents/skills/nli';

const agent = new BaseAgent({
  name: 'nli-agent',
  model: 'openai/gpt-4o',
  skills: [
    new NLISkill({
      timeout: 20_000,
      maxRetries: 3,
      transport: 'auto',
    }),
  ],
});
```

```python tab="Python"
from webagents.agents import BaseAgent
from webagents.agents.skills.robutler.nli import NLISkill

agent = BaseAgent(
    name="nli-agent",
    model="openai/gpt-4o",
    skills={
        "nli": NLISkill({
            "timeout": 20.0,
            "max_retries": 3,
        })
    },
)
```

## Example: Use NLI Tool in a Skill

```typescript tab="TypeScript"
import { Skill, tool } from 'webagents';
import type { NLISkill } from 'webagents/skills/nli';

class CollaborateSkill extends Skill {
  readonly name = 'collaborate';

  @tool({ description: 'Send a message to another agent and get the response' })
  async askAgent(params: { agentUrl: string; message: string }): Promise<string> {
    const nli = this.agent!.skills.find((s) => s.name === 'nli') as NLISkill;
    return nli.callAgent(params.agentUrl, params.message);
  }
}
```

```python tab="Python"
from webagents.agents.skills import Skill, tool

class CollaborateSkill(Skill):
    def __init__(self):
        super().__init__()
        self.nli = self.agent.skills["nli"]

    @tool
    async def ask_agent(self, agent_url: str, message: str) -> str:
        """Send a message to another agent and get the response"""
        return await self.nli.nli_tool(agent_url=agent_url, message=message)
```

## Agent Identifiers

The `agent` parameter accepts multiple formats:

| Format | Example | Description |
|:-------|:--------|:------------|
| `@username` | `@alice` | Platform agent by username |
| `@owner.agent` | `@alice.my-bot` | Namespaced agent (dot-namespace) |
| `@owner.agent.sub` | `@alice.my-bot.helper` | Sub-agent |
| `username` | `alice.my-bot` | Same as above, without `@` |
| URL | `https://example.com/agents/bot` | Direct URL to an external agent |

Dot-namespace names (`@alice.my-bot.helper`) are single URL path segments and route correctly through all transports.

## Trust Enforcement

Before making an outbound NLI call, the skill checks the calling agent's `talkTo` trust rules. If the target agent is not in scope, the call is refused with an error message:

> "Cannot communicate with @target — not in your trust scope."

Trust rules support presets (`everyone`, `family`, `platform`, `nobody`), glob patterns (`@alice.*`, `@com.example.**`), and trust labels (`#verified`, `#reputation:100`). See [Namespaces & Trust](../../guides/namespaces.md) for details.

Implementation: [`typescript/src/skills/nli/skill.ts`](https://github.com/robutlerai/webagents/blob/main/typescript/src/skills/nli/skill.ts) and [`python/webagents/agents/skills/robutler/nli/skill.py`](https://github.com/robutlerai/webagents/blob/main/python/webagents/agents/skills/robutler/nli/skill.py).