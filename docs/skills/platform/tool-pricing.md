# Tool Pricing

The Roborum platform supports monetization of MCP tools and agent skills.

## Overview

Each tool or skill defines its own pricing via the `@pricing` decorator or platform UI configuration. When a monetized tool is called, the PaymentSkill handles the full per-call payment lifecycle:

1. **Lock** — Reserve funds based on `@pricing` metadata (`lock` amount or `credits_per_call`)
2. **Execute** — Run the tool
3. **Settle** — Charge actual cost against the lock
4. **Release** — Return unused locked funds

Tools without pricing metadata are **free** — no lock is created, no payment flow runs.

## @pricing Decorator

Python agent tools use the `@pricing` decorator to declare costs:

```python
from webagents import tool
from webagents.agents.skills.robutler.payments import pricing

@tool
@pricing(credits_per_call=0.05, lock=0.10, reason="Image generation")
async def generate_image(prompt: str) -> dict:
    """Generate an image — costs 0.05 credits, locks 0.10"""
    ...
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `credits_per_call` | `float` | Fixed cost per invocation (in USD) |
| `lock` | `float` | Amount to lock before execution (defaults to `credits_per_call` if omitted) |
| `reason` | `str` | Human-readable description shown in billing |
| `supports_dynamic` | `bool` | Set automatically when no `credits_per_call` is given |

### Dynamic Pricing

For tools where cost depends on the result, use dynamic pricing:

```python
@tool
@pricing()  # No fixed price — dynamic
async def process_data(data: str) -> tuple:
    result = expensive_operation(data)
    cost = compute_actual_cost(data)
    return result, PricingInfo(credits=cost, reason="Data processing")
```

## _metering Convention

MCP tools can report fine-grained usage by returning a `_metering` object:

```json
{
  "result": { ... },
  "_metering": {
    "tokens": 1500,
    "images": 2
  }
}
```

The platform uses `_metering` dimensions combined with per-unit pricing (configured in the platform UI) to calculate the actual cost. The `_metering` key is stripped before the response reaches the caller.

## Per-Call Lock/Settle Flow

With the per-call architecture, each tool invocation gets its own lock:

```
before_toolcall           Tool Execution           after_toolcall
      │                        │                        │
      ├─ lock(amount) ────►    │                        │
      │                   execute()                     │
      │                        │    ├── settle(lockId, actual_cost)
      │                        │    ├── release(lockId)
```

If the lock fails (insufficient balance), the tool is **skipped** — `tool_skipped=True` is set on the context and the tool does not execute.

## Commission Distribution

Agent-side payment skills do not compute fee splits. A single `settle(amount)` call handles all distribution server-side:

- Work amount goes to the tool/skill provider
- Platform commission (configurable) goes to the platform
- Agent commissions go to each agent in the delegation chain

## External MCP Tool Pricing

External MCP tools connected through the platform UI can be priced via the **Monetization > Paid Tools** panel. The platform wraps these tools with the same lock/settle lifecycle — the MCP server itself does not need to be aware of payments.
