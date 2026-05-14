"""
CronSkill (Python) — reads `agent_configs.skills.cron.schedules[]` and invokes
either a function (when `use` is set) or the host agent main loop.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from ...base import Skill


@dataclass
class CronScheduleEntry:
    id: str
    cron: str
    use: Optional[str] = None
    enabled: bool = True
    description: Optional[str] = None


class CronSkill(Skill):
    name = "cron"

    def __init__(self, schedules: Optional[List[CronScheduleEntry]] = None) -> None:
        super().__init__(dependencies=["function-runtime"])
        self.schedules = schedules or []

    @property
    def system_prompt(self) -> str:
        if not self.schedules:
            return ""
        lines = [f"- {s.cron}: {s.use or '(host agent)'}" for s in self.schedules if s.enabled]
        return "## Cron\n\nScheduled tasks:\n" + "\n".join(lines)
