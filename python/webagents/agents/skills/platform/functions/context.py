"""
FunctionContext + invocation chain (Python).

Mirror of the TS shape. Host APIs are bridged by the executor coordinator
over mTLS; user code only sees the `ctx.*` surface.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Protocol


FunctionSourceSkill = Literal["cron", "custom_http", "custom_tools", "manual", "function"]


@dataclass
class FunctionSource:
    skill: FunctionSourceSkill
    consumerId: str
    invocationId: str


@dataclass
class FunctionAuth:
    userId: Optional[str] = None
    agentId: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    payment: Optional[Dict[str, Any]] = None
    claims: Optional[Dict[str, Any]] = None


@dataclass
class FunctionRequest:
    method: str
    path: str
    params: Dict[str, str]
    query: Dict[str, str]
    headers: Dict[str, str]
    body: Any = None
    rawBody: Optional[bytes] = None


@dataclass
class FunctionScheduleInfo:
    plannedAt: str
    firedAt: str


@dataclass
class FunctionToolCall:
    name: str
    params: Any
    callId: str


@dataclass
class FunctionLimitsResolved:
    wallMs: int
    cpuMs: int
    memoryMb: int
    ingressBytes: int
    egressBytes: int


class FunctionFnApi(Protocol):
    async def invoke(self, name: str, args: Any, *, timeoutMs: Optional[int] = None, idempotencyKey: Optional[str] = None) -> Any: ...
    def list(self) -> List[str]: ...


class FunctionSecrets(Protocol):
    async def get(self, name: str) -> Optional[str]: ...
    async def put(self, name: str, value: str) -> None: ...
    async def list(self) -> List[str]: ...


class FunctionKv(Protocol):
    async def get(self, key: str) -> Any: ...
    async def put(self, key: str, value: Any, *, ttlMs: Optional[int] = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def list(self, prefix: Optional[str] = None, *, limit: Optional[int] = None, cursor: Optional[str] = None) -> Dict[str, Any]: ...


class FunctionLog(Protocol):
    def debug(self, *args: Any) -> None: ...
    def info(self, *args: Any) -> None: ...
    def warn(self, *args: Any) -> None: ...
    def error(self, *args: Any) -> None: ...


class PortalHelpers(Protocol):
    async def verifyToken(self, token: str, *, expectAudience: Optional[str] = None, expectBalance: bool = False) -> Dict[str, Any]: ...
    async def verifyHmac(self, *, algo: str, secretBinding: str, payload: bytes, expected: str) -> bool: ...
    async def lookupAgent(self, idOrUsername: str) -> Optional[Dict[str, Any]]: ...
    async def callTool(self, agentRef: str, toolName: str, params: Any, *, timeoutMs: Optional[int] = None, paymentToken: Optional[str] = None) -> Any: ...
    async def getOwner(self) -> Dict[str, Any]: ...
    async def notifyOwner(self, *, title: str, body: str, severity: str = "info", deepLink: Optional[str] = None) -> None: ...
    async def signContentUrl(self, contentId: str, *, expiresInSeconds: int = 600) -> str: ...


@dataclass
class FunctionContext:
    source: FunctionSource
    auth: FunctionAuth
    fetch: Callable[[str, Optional[Dict[str, Any]]], Awaitable[Any]]
    secrets: FunctionSecrets
    kv: FunctionKv
    fn: FunctionFnApi
    log: FunctionLog
    portal: PortalHelpers
    limits: FunctionLimitsResolved
    request: Optional[FunctionRequest] = None
    schedule: Optional[FunctionScheduleInfo] = None
    toolCall: Optional[FunctionToolCall] = None
    folders: Dict[str, Any] = field(default_factory=dict)
    content: Optional[Any] = None
    emit: Optional[Callable[[Dict[str, int]], None]] = None


@dataclass
class InvocationChain:
    rootInvocationId: str
    depth: int
    path: List[str]
    budgetRemaining: FunctionLimitsResolved
    traceparent: Optional[str] = None


DEFAULT_MAX_FN_CHAIN_DEPTH = 5
FN_BUDGET_BUFFER_MS = 50
FN_ERR_CHAIN_TOO_DEEP = "FN_CHAIN_TOO_DEEP"
FN_ERR_CYCLE_DETECTED = "FN_CYCLE_DETECTED"
FN_ERR_QUOTA_EXHAUSTED = "FN_QUOTA_EXHAUSTED"
