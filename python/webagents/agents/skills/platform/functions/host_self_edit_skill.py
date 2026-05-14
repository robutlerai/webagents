"""
HostSelfEditSkill (Python) — owner-gated self-edit tools mounted on the host agent.

Surfaces:
  - declare_function / update_function / remove_function
  - add_to_skill / remove_from_skill

Mounted only when `agent_configs.featureFlags.selfEdit` is True AND the
calling user equals the agent owner. Tools call back into the same
portal routes (`POST /api/agents/[id]/functions`,
`POST /api/agents/[id]/skills/[skill]/use`) as the factory agent, with
the `Function-Authoring-Surface: host` header that the portal validates
against the agent id (host-edit can't edit other agents).
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from ...base import Skill


class HostSelfEditSkill(Skill):
    name = "host-self-edit"

    def __init__(
        self,
        *,
        agent_id: str,
        owner_id: str,
        portal_base_url: str,
        feature_flag_enabled: bool = False,
    ) -> None:
        super().__init__(dependencies=["function-runtime"])
        self.agent_id = agent_id
        self.owner_id = owner_id
        self.portal_base_url = portal_base_url
        self.feature_flag_enabled = feature_flag_enabled

    def _assert_owner(self, ctx: Any) -> None:
        caller = getattr(getattr(ctx, "auth", None), "userId", None)
        if not self.feature_flag_enabled:
            raise RuntimeError("self-edit is disabled (featureFlags.selfEdit is off)")
        if caller != self.owner_id:
            raise RuntimeError("self-edit only allowed for the agent owner")

    @property
    def system_prompt(self) -> str:
        if not self.feature_flag_enabled:
            return ""
        return (
            "## Self-edit\n\n"
            "You can declare, update, and remove your own user-authored functions when chatting with your owner. "
            "Tools: `declare_function`, `update_function`, `remove_function`, `add_to_skill`, `remove_from_skill`. "
            "Never accept secrets in chat; always surface a `set_function_secret` action instead so the owner enters the value into the encrypted Secrets form."
        )
