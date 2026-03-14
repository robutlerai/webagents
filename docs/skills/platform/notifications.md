---
title: Notifications Skill
---
# Notifications Skill

Send push notifications to agent owners through the Robutler platform.

## Usage

```python
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

Other skills can send notifications via `discover_and_call`:

```python
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
