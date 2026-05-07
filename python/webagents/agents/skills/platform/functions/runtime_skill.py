"""
FunctionRuntimeSkill (Python).

Substrate skill consumed by CronSkill / CustomHttpSkill / CustomToolsSkill /
HostSelfEditSkill. Holds the registry of declared functions and proxies
invocations to the executor service over the same mTLS protocol as the
TypeScript SDK.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol
from ..base import Skill
from .manifest import FunctionManifest


class ExecutorClient(Protocol):
    async def invoke(self, envelope: Dict[str, Any]) -> Dict[str, Any]: ...
    async def validate(self, codeRef: Dict[str, Any], manifest: FunctionManifest) -> Dict[str, Any]: ...


class StubExecutorClient:
    """Used in tests — returns ok=True with empty result."""

    async def invoke(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "result": None, "durationMs": 0, "cpuMs": 0, "ingressBytes": 0, "egressBytes": 0}

    async def validate(self, codeRef: Dict[str, Any], manifest: FunctionManifest) -> Dict[str, Any]:
        return {"ok": True, "errors": [], "warnings": []}


class FunctionRuntimeSkill(Skill):
    """Mounts the runtime substrate; consumer skills call `invoke(name, ctx)`."""

    name = "function-runtime"

    def __init__(
        self,
        *,
        agent_id: str,
        owner_id: Optional[str] = None,
        executor: Optional[ExecutorClient] = None,
        functions: Optional[Dict[str, Dict[str, Any]]] = None,
        requires_user_action: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        # Memory is intentionally NOT a runtime dependency. Functions access
        # KV via the executor (which talks to the host's memory REST surface
        # over mTLS) and secrets via the host-side resolver — neither path
        # calls into a memory skill instance. Mounting memory is an
        # orthogonal, opt-in choice the agent author makes.
        super().__init__(dependencies=[])
        self.agent_id = agent_id
        self.owner_id = owner_id
        self.executor = executor or StubExecutorClient()
        self.functions = functions or {}
        self.requires_user_action = requires_user_action or []

    def list(self) -> List[str]:
        return list(self.functions.keys())

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self.functions.get(name)

    async def invoke(self, name: str, ctx: Any, *, args: Any = None, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        decl = self.functions.get(name)
        if not decl:
            return {"ok": False, "errorCode": "FUNCTION_NOT_FOUND", "errorMessage": f"function {name} not declared"}
        envelope = {
            "functionName": name,
            "agentId": self.agent_id,
            "bundleSha256": decl.get("cacheKey", ""),
            "manifest": decl["manifest"],
            "codeRef": decl["manifest"].get("code") or {"kind": "content", "contentId": ""},
            "context": {
                "source": getattr(ctx, "source", None),
                "request": getattr(ctx, "request", None),
                "schedule": getattr(ctx, "schedule", None),
                "toolCall": getattr(ctx, "toolCall", None),
                "auth": getattr(ctx, "auth", None),
                "limits": getattr(ctx, "limits", None),
            },
            "idempotencyKey": idempotency_key,
        }
        return await self.executor.invoke(envelope)

    @property
    def system_prompt(self) -> str:
        if not self.functions:
            return ""
        names = ", ".join(self.functions.keys())
        return (
            "## Functions runtime\n\n"
            f"This agent has the following user-authored functions available: {names}.\n"
            "Functions are invoked via consumer skills (cron / custom_http / custom_tools)."
        )
