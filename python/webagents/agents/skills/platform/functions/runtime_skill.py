"""
FunctionRuntimeSkill (Python).

Substrate skill consumed by CronSkill / CustomHttpSkill / CustomToolsSkill /
HostSelfEditSkill. Holds the registry of declared functions and proxies
invocations to the executor service over the same mTLS protocol as the
TypeScript SDK.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol
from ...base import Skill
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
        # Mirrors the TS `functionsRuntimeStatic` @prompt block so Python-runtime
        # agents get the same runtime guidance (sandbox, host APIs, error
        # playbook, templates, iteration loop) when the FunctionRuntimeSkill is
        # mounted. Keep in lock-step with
        # `webagents/typescript/src/skills/functions/skill.ts` →
        # `functionsRuntimeStatic`.
        sections = []
        if self.functions:
            names = ", ".join(self.functions.keys())
            sections.append(
                "## Functions runtime\n\n"
                f"This agent has the following user-authored functions available: {names}.\n"
                "Functions are invoked via consumer skills (cron / custom_http / custom_tools)."
            )
        sections.append(_RUNTIME_STATIC_BLOCK)
        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Static runtime guidance (sandbox / host APIs / error playbook / templates).
# Single source of truth on the Python side; mirrors TS
# `functionsRuntimeStatic`. Kept as a module-level constant so it composes
# into `system_prompt` regardless of whether functions are declared.
# ---------------------------------------------------------------------------

_RUNTIME_STATIC_BLOCK = """## Functions runtime (js-v1)

js-v1 is the only enabled runtime. Setting `runtime` to anything else fails validation with `RUNTIME_DISABLED` (ADR-0008).

### Sandbox
- Bare V8 isolate (isolated-vm). NO Node globals: `process`, `Buffer`, `require`, `fs`, `eval`, `Function` are all blocked.
- NO npm packages or Node-only modules — bundling is deferred to v2. If you need a library, inline the small subset you actually use.
- Entrypoint: an async handler. Prefer `export default async function handler(ctx) { ... }`. `export async function handler(ctx)` and `module.exports = async (ctx) => ...` also work.
- Globals available: `URL`, `URLSearchParams`, `atob`, `btoa`, `JSON`, `Math`, `Date`, `Promise`, `Map`, `Set`, `RegExp`, `Symbol`, `Proxy`, `Reflect`, `Intl`, `console`, `TextEncoder`/`TextDecoder`, `structuredClone`, `crypto.{randomUUID,getRandomValues,subtle}`.
- Host APIs (only via `ctx`, all permission-gated by `manifest.permissions`): `ctx.fetch` (URL allowlist), `ctx.secrets`, `ctx.kv` (`none`/`ro`/`rw`), `ctx.content`, `ctx.folders`, `ctx.fn`, `ctx.portal` (payment.*, agent.*), `ctx.log`, `ctx.emit`.
- Inline source cap: 16 KB UTF-8 (`inline`) or 64 KB base64 (`inlineB64`); bigger source must move to a content row.
- Defaults: `wallMs=30s` (fallback `10s` for some entry paths), `cpuMs=5s`, `memoryMb=128` (max 512), `ingressBytes=egressBytes=1MB`. Long-running work MUST finish within `wallMs` — there is a host-side watchdog.

### Error playbook (errorCode → what to do)
When `invocation.ok === false`, map the `errorCode` to the user-facing fix:
- `FN_NOT_FOUND` — function name is wrong or not declared. Confirm the canonical name; do NOT auto-create a same-named function.
- `FN_CHAIN_TOO_DEEP` — `ctx.fn.invoke` recursed past the per-plan depth cap. Restructure to a flat fan-out instead of a chain; if recursion is intentional, set `manifest.permissions.selfRecursion: true`.
- `FN_CYCLE_DETECTED` — same function appeared twice in the chain path. Same fix as above; cycles are blocked even when `selfRecursion` is on.
- `FN_QUOTA_EXHAUSTED` / `QUOTA_EXCEEDED` — cumulative wall/cpu/network budget for the chain ran out. Move heavy work to a single top-level invocation, or cache via `ctx.kv` and short-circuit on a hit.
- `TIMEOUT` — single invocation exceeded `wallMs`. Profile the slow step (most often a synchronous JSON.parse on a large body or a slow `ctx.fetch`). Split the work, lower the body size, or raise `wallMs` in the manifest (capped at 30s).
- `RUNTIME_DISABLED` — manifest pinned `python-pyodide-v1`/`wasm-v1`. Switch to `js-v1`; the others are reserved.
- `HOST_BRIDGE_MINT_FAILED` — host token couldn't be minted (auth/quota issue on the portal). Surface to the user; do NOT retry with the same envelope, the failure is on the host side.
- `EXECUTOR_THREW` / `FUNCTION_ERROR` — function code threw. The `errorMessage` carries the user-thrown message — surface it (sanitised) to the user; if you authored the function, fix the throw.
- `HOST_QUOTA_EXCEEDED` — agent owner ran out of plan budget for a host API (KV writes, fetch egress, content storage). Tell the user the owner needs to upgrade their plan; do NOT retry.

### Iteration loop for declaring NEW functions
Always: `declare_function` → manual invoke as a smoke test → read the invocation result (success or `errorCode`) → iterate. NEVER attach a freshly-declared function to a skill (`add_to_skill cron|custom_http|custom_tools`) before the smoke test passes — a broken function attached to cron will fail silently every minute, and a broken `custom_http` endpoint surfaces 500s to whoever calls it.

### Common templates (start from these instead of guessing)

1. **Fetch + parse JSON** (with `manifest.permissions.fetch: ["https://api.example.com"]`):
   ```
   export default async function handler(ctx) {
     const r = await ctx.fetch("https://api.example.com/v1/items");
     if (!r.ok) throw new Error(`upstream ${r.status}`);
     return await r.json();
   }
   ```

2. **KV read/write** (with `manifest.permissions.kv: "rw"`):
   ```
   export default async function handler(ctx) {
     const seen = await ctx.kv.get(`seen:${ctx.toolCall.params.id}`);
     if (seen) return { cached: true, value: seen };
     const fresh = computeSomething(ctx.toolCall.params);
     await ctx.kv.set(`seen:${ctx.toolCall.params.id}`, fresh, { ttlSeconds: 3600 });
     return { cached: false, value: fresh };
   }
   ```

3. **Chain to another function** (with declared `targetFn` in `agent_configs.functions`):
   ```
   export default async function handler(ctx) {
     const inner = await ctx.fn.invoke("targetFn", { foo: 1 });
     return { wrapped: inner };
   }
   ```

4. **Throw a structured error** (so the caller sees `errorMessage`):
   ```
   export default async function handler(ctx) {
     if (!ctx.toolCall.params.email) throw new Error("missing email");
     return { ok: true };
   }
   ```

5. **Emit a non-result side-channel event** (e.g. progress for a long task):
   ```
   export default async function handler(ctx) {
     ctx.emit({ type: "progress", percent: 42 });
     // ... continue work
     return { done: true };
   }
   ```

### HTML / browser-renderable returns
Returning HTML/JS bytes from a function is a separate, security-sensitive feature spec'd in the **HTML Rendering Security Model** plan. Do NOT improvise an inline iframe rendering surface from a function return — wait until that plan ships its dedicated content type and renderer. For now: if a user asks for an HTML artifact, write it via `text_editor create path="/file.html"` (sandboxed iframe rendering already covers that path), not via a function return."""
