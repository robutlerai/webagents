# Notifications Skill

Send push notifications to agent owners through the Robutler platform.

## Overview

The `NotificationsSkill` provides owner-scoped push notification capabilities, allowing agents to send notifications directly to their owners through the Robutler portal notification system.

## Features

- **Owner-Only Access**: Notifications are restricted to the agent owner using `scope="owner"`
- **Push Notification Delivery**: Integrates with the Robutler portal notification API
- **Customizable Notifications**: Support for different notification types, priorities, and settings
- **Automatic Authentication**: Uses agent API keys for secure notification delivery

## Usage

### Basic Setup

```python
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.notifications.skill import NotificationsSkill

agent = BaseAgent(
    name="notification-agent",
    model="openai/gpt-4o-mini",
    skills={
        "notifications": NotificationsSkill()
    }
)
```

### Sending Notifications

The skill provides a single tool for sending notifications:

```python
# The agent can use this tool to send notifications
response = await agent.run(messages=[
    {"role": "user", "content": "Send me a notification that the task is complete"}
])
```

## Tool Reference

### `send_notification`

Send a push notification to the agent owner.

**Parameters:**

- `title` (str, required): Notification title
- `body` (str, required): Notification body text
- `tag` (str, optional): Notification tag for grouping
- `type` (str, optional): Notification type (`chat_message`, `agent_update`, `system_announcement`, `marketing`). Default: `agent_update`
- `priority` (str, optional): Priority level (`low`, `normal`, `high`, `urgent`). Default: `normal`
- `requireInteraction` (bool, optional): Whether notification requires user interaction. Default: `false`
- `silent` (bool, optional): Whether notification should be silent. Default: `false`
- `ttl` (int, optional): Time-to-live in seconds. Default: `86400` (24 hours)

**Returns:**

- Success: `"✅ Notification queued: {message}"`
- Error: `"❌ Failed to send notification: {error}"`

**Scope:** `owner` - Only the agent owner can trigger notifications

## Configuration

The skill requires no additional configuration beyond adding it to your agent. It automatically:

- Resolves the agent owner's user ID
- Uses the agent's API key for authentication
- Connects to the appropriate Robutler portal API endpoint

## Security

- **Owner-Only Access**: All notification tools are scoped to `owner` only
- **API Key Authentication**: Uses secure agent API keys for portal communication
- **User ID Resolution**: Automatically identifies the correct recipient based on agent ownership

## Example Integration

```python
from webagents import Skill, tool

class TaskSkill(Skill):
    @tool
    async def complete_task(self, task_name: str) -> str:
        # Perform task logic here
        task_result = f"Completed: {task_name}"
        
        # Send notification when task completes
        await self.discover_and_call(
            "notifications", 
            f"Task Complete: {task_name}", 
            f"Your task '{task_name}' has been completed successfully."
        )
        
        return task_result
```

## Error Handling

The skill handles common error scenarios:

- **Missing Owner ID**: Returns error message if agent owner cannot be resolved
- **API Authentication Failures**: Handles missing or invalid API keys
- **Network Issues**: Provides meaningful error messages for connection problems
- **Portal API Errors**: Surfaces API error responses for debugging

## Dependencies

- **Agent API Key**: Requires valid agent API key for portal authentication
- **Owner Context**: Requires agent to have identifiable owner for targeting notifications
- **Portal Connectivity**: Requires network access to Robutler portal API endpoints
