# Payment Skill

Payment processing and billing skill for the Robutler platform. This skill enforces billing policies via per-call lock/settle/release cycles for both LLM calls and tool calls.

!!! note "x402 Protocol Support"
    For full x402 protocol support (HTTP endpoint payments, blockchain payments, automatic exchange), see [PaymentSkillX402](../robutler/payments-x402.md). PaymentSkill focuses on per-call charging and basic token validation, while PaymentSkillX402 extends it with multi-scheme payments and automatic payment handling for HTTP APIs.

## Architecture

The PaymentSkill uses a **per-call lock/settle/release** model:

1. **`on_connection`** — Verify payment token and balance (no lock created)
2. **`before_llm_call`** — Lock funds based on model pricing and estimated input tokens
3. **`after_llm_call`** — Settle actual LLM usage against the lock, then release
4. **`before_toolcall`** — Lock funds for priced tools; free tools are skipped
5. **`after_toolcall`** — Settle actual tool cost against the lock, then release
6. **`finalize_connection`** — Safety net: release any orphaned locks

```
Client                          PaymentSkill                    Portal
  │                                 │                              │
  ├─ connect ──────────────────────►│                              │
  │                                 ├── verify token ─────────────►│
  │                                 │◄── {valid, balance} ─────────┤
  │                                 │                              │
  ├─ message ──────────────────────►│                              │
  │                                 ├── lock(model, input_tokens) ►│
  │                                 │◄── {lockId} ────────────────┤
  │        (LLM processes)          │                              │
  │                                 ├── settle(lockId, usage) ────►│
  │                                 │◄── {charged} ───────────────┤
  │                                 │                              │
  │        (tool: image_gen)        │                              │
  │                                 ├── lock(amount=0.05) ────────►│
  │                                 │◄── {lockId} ────────────────┤
  │        (tool executes)          │                              │
  │                                 ├── settle(lockId, amount) ───►│
  │                                 │◄── {charged} ───────────────┤
  │                                 │                              │
  │◄─ response ─────────────────────┤                              │
  │                                 ├── finalize (release orphans) │
```

## Key Features

- Payment token verification during `on_connection` (returns 402 if required and missing)
- **Dynamic LLM lock**: estimated input tokens + max output tokens, priced via Portal's `MODEL_PRICING` catalog
- **Per-tool lock/settle**: each priced tool gets its own lock; free tools skip payment entirely
- BYOK-aware: LLM costs settle as `byok_llm` when user provides their own API key
- Transport-agnostic token extraction (HTTP headers, UAMP events, query params)
- Depends on `AuthSkill` for user identity propagation

## Configuration

```python
PaymentSkill({
    "enable_billing": True,        # Enable/disable billing
    "minimum_balance": 1.0,        # USD required to proceed (0 = free trial)
    "webagents_api_url": "...",    # Portal API base URL
    "robutler_api_key": "...",     # Service-to-portal API key
})
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_billing` | `True` | Toggle billing on/off |
| `minimum_balance` | `0` | Minimum balance in USD; 0 allows free trials without token |
| `webagents_api_url` | — | Portal API URL |
| `robutler_api_key` | — | API key for portal calls |

!!! warning "Removed Parameters"
    `per_message_lock` and `default_tool_lock` have been removed. Locks are now computed dynamically per-call.

## Example: Add Payment Skill to an Agent

```python
from webagents.agents import BaseAgent
from webagents.agents.skills.robutler.auth.skill import AuthSkill
from webagents.agents.skills.robutler.payments import PaymentSkill

agent = BaseAgent(
    name="paid-agent",
    model="openai/gpt-4o",
    skills={
        "auth": AuthSkill(),
        "payments": PaymentSkill({
            "enable_billing": True,
            "minimum_balance": 1.0
        })
    }
)
```

## LLM Cost Handling

LLM costs are handled per-call by two hooks:

### before_llm_call

1. Estimates input tokens from `context.messages`
2. Reads the active LLM skill's `model` name and `max_tokens` config
3. Sends `POST /payments/lock` with `{model, input_tokens, max_output_tokens}` to the Portal
4. The Portal computes the lock amount using `MODEL_PRICING` catalog:
   - `lock = input_tokens * inputPer1k/1000 + max_output * outputPer1k/1000`
   - `max_output_tokens` priority: agent's `max_tokens` config > catalog's `maxOutputTokens` > safe default (16384)
   - Unknown models use safe defaults ($0.003/1k input, $0.015/1k output)

### after_llm_call

1. Settles actual usage against the lock with raw usage records
2. Releases the lock immediately after settlement
3. BYOK sessions settle with `charge_type='byok_llm'`

## Tool Pricing with @pricing Decorator

Tools define their own pricing via the `@pricing` decorator. Tools without `@pricing` are free — no lock is created.

```python
from webagents import tool
from webagents.agents.skills.robutler.payments import pricing, PricingInfo

@tool
@pricing(credits_per_call=0.05, lock=0.10, reason="Database query")
async def query_database(sql: str) -> dict:
    """Query database - costs 0.05 credits per call, locks 0.10"""
    return {"results": [...]}

@tool
@pricing()  # Dynamic pricing — must return PricingInfo
async def analyze_data(data: str) -> tuple:
    """Analyze data with variable pricing based on complexity"""
    complexity = len(data)
    result = f"Analysis of {complexity} characters"
    
    credits = max(0.01, complexity * 0.001)
    
    pricing_info = PricingInfo(
        credits=credits,
        reason=f"Data analysis of {complexity} chars",
        metadata={"character_count": complexity}
    )
    return result, pricing_info
```

### Pricing Options

1. **Fixed Pricing**: `@pricing(credits_per_call=0.05)` — 0.05 credits per call
2. **Fixed with Lock**: `@pricing(credits_per_call=0.05, lock=0.10)` — lock 0.10, settle 0.05
3. **Dynamic Pricing**: `@pricing()` + return `(result, PricingInfo(credits=..., ...))`
4. **No Pricing**: Omit `@pricing` entirely — tool is free

### Per-Tool Lock/Settle Flow

For each priced tool call:

1. **`before_toolcall`**: Creates a new lock using the tool's `@pricing` metadata (`lock` amount, or `credits_per_call` as fallback)
2. Tool executes (if lock fails, `tool_skipped=True` and the tool is blocked)
3. **`after_toolcall`**: Settles actual cost against the lock, then releases

Free tools (no `@pricing`) skip the entire payment flow.

## Hook Integration

| Hook | Behavior |
|------|----------|
| **`on_connection`** | Verify payment token and balance. Raises 402 if billing enabled and no token. **No lock created.** |
| **`before_llm_call`** | Acquire dynamic LLM lock based on model pricing and estimated tokens |
| **`after_llm_call`** | Settle actual LLM cost, release lock |
| **`before_toolcall`** | Acquire per-tool lock (priced tools only) |
| **`after_toolcall`** | Settle actual tool cost, release lock |
| **`finalize_connection`** | Safety net: release any orphaned locks from errors, set `payment_successful` |

## Context Namespacing

The PaymentSkill stores data in the `payments` namespace of the request context:

```python
from webagents.server.context.context_vars import get_context

context = get_context()
payments_data = getattr(context, 'payments', None)
payment_token = getattr(payments_data, 'payment_token', None) if payments_data else None
```

Per-call lock IDs are stored as `_llm_lock_id` and `_tool_lock_id` on the context and cleared after each settle/release cycle.

## Error Semantics (402)

- Missing token while `enable_billing` and `minimum_balance > 0` → 402 Payment Required
- Invalid or expired token → 402 Payment Token Invalid
- Insufficient balance → 402 Insufficient Balance
- Lock failure on tool call → `tool_skipped=True` (tool is not executed)

## Transport-Agnostic Payments

PaymentSkill extracts the payment token in a **transport-agnostic** manner.
The skill reads `context.payment_token` first (set by any transport), then falls back to HTTP
headers (`X-Payment-Token`, `X-PAYMENT`) and query parameters as a legacy path.

### Token extraction priority

1. **`context.payment_token`** — set by the transport (UAMP `session.update`, portal `payment.submit`, etc.)
2. **HTTP header** — `X-Payment-Token` or `X-PAYMENT` (Completions, A2A)
3. **Query parameter** — `?payment_token=...` (legacy)

### PaymentTokenRequiredError

When billing is enabled and no token is found, the skill raises `PaymentTokenRequiredError`
(HTTP status 402). Each transport catches this and maps it to its protocol:

| Transport | Behavior |
|-----------|----------|
| **Completions** | Returns 402 JSON before streaming (pre-flight check) |
| **UAMP** | Sends `payment.required` event, waits for `payment.submit`, retries, sends `payment.accepted` |
| **A2A** | Returns `task.failed` with `code: "payment_required"` and `accepts` array |
| **ACP** | Returns JSON-RPC error `-32402` with payment data |
| **Realtime** | Sends `payment.required` event over audio WebSocket |

### Example: UAMP inline payment negotiation

```
Client                      Agent (UAMP)
  │                             │
  ├─ input.text ───────────────►│
  │                             ├─ (skill raises PaymentTokenRequiredError)
  │◄── payment.required ───────┤
  │                             │
  ├─ payment.submit ───────────►│  (token from facilitator)
  │                             ├─ (retry with context.payment_token)
  │◄── response.delta ─────────┤
  │◄── response.done ──────────┤
  │◄── payment.accepted ───────┤
```

## Dependencies

- **AuthSkill**: Required for user identity headers (`X-Origin-User-ID`, `X-Peer-User-ID`, `X-Agent-Owner-User-ID`). The Payment Skill reads them from the auth namespace on the context.

Implementation: `webagents/agents/skills/robutler/payments/skill.py`.
