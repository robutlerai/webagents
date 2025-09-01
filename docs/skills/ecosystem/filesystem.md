# Filesystem Skill

Status: Under construction – implementation is in progress. The API below is illustrative and may change.

## Features
- Read/write files
- List directories
- File metadata access

## Planned usage: Add Filesystem Skill to an Agent
```python
from webagents.agents import BaseAgent
# from webagents.agents.skills.ecosystem.filesystem import FilesystemSkill  # coming soon

agent = BaseAgent(
    name="fs-agent",
    model="openai/gpt-4o",
    skills={
        # "filesystem": FilesystemSkill({})  # coming soon
    }
)
```

## Planned usage: Use Filesystem Tool in a Skill
```python
from webagents.agents.skills import Skill, tool  # example pattern

class FileOpsSkill(Skill):  # illustrative
    @tool
    async def read_file(self, path: str) -> str:
        """Read a file from the filesystem"""
        # return await self.agent.skills["filesystem"].read_file(path)
        raise NotImplementedError("filesystem skill integration coming soon")
```

**Implementation:** Under construction in `webagents/agents/skills/ecosystem/filesystem/`.