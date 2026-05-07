"""
WebAgents — Functions skill suite (Python).

Mirrors the TypeScript SDK shape:

  - FunctionRuntimeSkill   — substrate; consumed by cron/custom_http/custom_tools.
  - CronSkill              — cron schedules; entries with optional `use` field.
  - CustomHttpSkill        — HTTP endpoints registered via `@http`.
  - CustomToolsSkill       — LLM tools registered via `@tool`.
  - HostSelfEditSkill      — owner-gated self-edit tools (declare/update/remove).

Function manifest, codeRef variants, and FunctionContext are defined in
the sibling `manifest.py` and `context.py` modules.
"""

from .manifest import (
    FunctionManifest,
    FunctionPermissions,
    FunctionLimits,
    CodeRef,
    FunctionRuntimeId,
)
from .context import (
    FunctionContext,
    FunctionSource,
    FunctionAuth,
    FunctionRequest,
    FunctionScheduleInfo,
    FunctionToolCall,
    FunctionFnApi,
    FunctionLimitsResolved,
    InvocationChain,
    DEFAULT_MAX_FN_CHAIN_DEPTH,
    FN_BUDGET_BUFFER_MS,
    FN_ERR_CHAIN_TOO_DEEP,
    FN_ERR_CYCLE_DETECTED,
    FN_ERR_QUOTA_EXHAUSTED,
)
from .runtime_skill import FunctionRuntimeSkill, ExecutorClient, StubExecutorClient
from .cron_skill import CronSkill, CronScheduleEntry
from .custom_http_skill import CustomHttpSkill, CustomHttpEndpointEntry
from .custom_tools_skill import CustomToolsSkill, CustomToolEntry
from .host_self_edit_skill import HostSelfEditSkill

__all__ = [
    "FunctionManifest",
    "FunctionPermissions",
    "FunctionLimits",
    "CodeRef",
    "FunctionRuntimeId",
    "FunctionContext",
    "FunctionSource",
    "FunctionAuth",
    "FunctionRequest",
    "FunctionScheduleInfo",
    "FunctionToolCall",
    "FunctionFnApi",
    "FunctionLimitsResolved",
    "InvocationChain",
    "DEFAULT_MAX_FN_CHAIN_DEPTH",
    "FN_BUDGET_BUFFER_MS",
    "FN_ERR_CHAIN_TOO_DEEP",
    "FN_ERR_CYCLE_DETECTED",
    "FN_ERR_QUOTA_EXHAUSTED",
    "FunctionRuntimeSkill",
    "ExecutorClient",
    "StubExecutorClient",
    "CronSkill",
    "CronScheduleEntry",
    "CustomHttpSkill",
    "CustomHttpEndpointEntry",
    "CustomToolsSkill",
    "CustomToolEntry",
    "HostSelfEditSkill",
]
