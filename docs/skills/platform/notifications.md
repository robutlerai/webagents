---
title: Notifications Skill
description: Send push notifications to agent owners through the Robutler platform.
---

# Notifications Skill

Send push notifications to agent owners through the Robutler platform.

## Usage

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { NotificationsSkill } from 'webagents/skills/social';

const agent = new BaseAgent({
  name: 'notification-agent',
  model: 'openai/gpt-4o-mini',
  skills: [new NotificationsSkill()],
});
```

```python tab="Python"
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.notifications.skill import NotificationsSkill

agent = BaseAgent(
    name="notification-agent",
    model="openai/gpt-4o-mini",
    skills={
        "notifications": NotificationsSkill(),
    },
)
```

The skill provides a single tool, scoped to `owner` only.

## Tool Reference

### `send_notification`

Send a push notification to the agent owner.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `title` | str | Yes | — | Notification title |
| `body` | str | Yes | — | Notification body text |
| `tag` | str | No | — | Grouping tag |
| `type` | str | No | `agent_update` | `chat_message`, `agent_update`, `system_announcement`, `marketing` |
| `priority` | str | No | `normal` | `low`, `normal`, `high`, `urgent` |
| `requireInteraction` | bool | No | `false` | Whether notification requires user interaction |
| `silent` | bool | No | `false` | Silent notification |
| `ttl` | int | No | `86400` | Time-to-live in seconds |

## Cross-Skill Usage

Other skills can send notifications by looking up the skill instance:

```typescript tab="TypeScript"
import { Skill, tool } from 'webagents';
import type { NotificationsSkill } from 'webagents/skills/social';

class TaskSkill extends Skill {
  readonly name = 'tasks';

  @tool({ description: 'Mark a task complete and notify the owner' })
  async completeTask(params: { taskName: string }): Promise<string> {
    const result = `Completed: ${params.taskName}`;
    const notify = this.agent!.skills.find(
      (s) => s.name === 'notifications',
    ) as NotificationsSkill;
    await notify.sendNotification({
      title: `Task Complete: ${params.taskName}`,
      body: `Your task '${params.taskName}' has been completed.`,
    });
    return result;
  }
}
```

```python tab="Python"
class TaskSkill(Skill):
    @tool
    async def complete_task(self, task_name: str) -> str:
        result = f"Completed: {task_name}"
        await self.discover_and_call(
            "notifications",
            f"Task Complete: {task_name}",
            f"Your task '{task_name}' has been completed."
        )
        return result
```
