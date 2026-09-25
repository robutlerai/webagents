"""REST calls, signed with Web Bot Auth when the agent can sign (ADR-0045)."""

from .skill import RestSkill, TOOL_DEFINITION

__all__ = ["RestSkill", "TOOL_DEFINITION"]
