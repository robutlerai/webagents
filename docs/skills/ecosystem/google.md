# Google Skill

Integrate with Google APIs (Drive, Sheets, etc.).

## Features
- File access
- Spreadsheet operations
- OAuth authentication

## Example: Add Google Skill to an Agent
```python
from webagents.agents import BaseAgent
from webagents.agents.skills.ecosystem.google import GoogleSkill

agent = BaseAgent(
    name="google-agent",
    model="openai/gpt-4o",
    skills={
        "google": GoogleSkill({})
    }
)
```

## Example: Use Google Tool in a Skill
```python
from webagents.agents.skills import Skill, tool

class GoogleOpsSkill(Skill):
    def __init__(self):
        super().__init__()
        self.google = self.agent.skills["google"]

    @tool
    async def list_drive_files(self) -> str:
        """List files in Google Drive"""
        return await self.google.list_files()
```

**Implementation:** See `robutler/agents/skills/ecosystem/google/skill.py`. 