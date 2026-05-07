"""
CustomToolsSkill (Python) — exposes user functions as LLM tools.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from ..base import Skill


@dataclass
class CustomToolEntry:
    id: str
    use: str
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class CustomToolsSkill(Skill):
    name = "custom_tools"

    def __init__(self, tools: Optional[List[CustomToolEntry]] = None) -> None:
        super().__init__(dependencies=["function-runtime"])
        self.tools = tools or []

    @property
    def system_prompt(self) -> str:
        if not self.tools:
            return ""
        lines = [f"- {t.name} → {t.use}: {t.description or ''}" for t in self.tools if t.enabled]
        return "## Custom tools (function-backed)\n\n" + "\n".join(lines)
