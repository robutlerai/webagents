"""
CustomHttpSkill (Python) — exposes user functions as HTTP endpoints.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional
from ..base import Skill


@dataclass
class CustomHttpEndpointEntry:
    id: str
    use: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str
    auth: Literal["public", "signature", "session", "portal_token"] = "public"
    description: Optional[str] = None
    enabled: bool = True


class CustomHttpSkill(Skill):
    name = "custom_http"

    def __init__(self, endpoints: Optional[List[CustomHttpEndpointEntry]] = None) -> None:
        super().__init__(dependencies=["function-runtime"])
        self.endpoints = endpoints or []

    @property
    def system_prompt(self) -> str:
        if not self.endpoints:
            return ""
        lines = [f"- {e.method} {e.path} → {e.use} ({e.auth})" for e in self.endpoints if e.enabled]
        return "## Custom HTTP endpoints\n\n" + "\n".join(lines)
