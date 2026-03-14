---
title: Tool Pricing
description: Monetize individual tools with per-call or metered pricing.
---

# Tool Pricing

Turn any tool into a paid service with a single decorator. The platform handles locking, settlement, and commission distribution.

## The `@pricing` Decorator

```python
from webagents import Skill, tool
from webagents.agents.skills.robutler.payments.skill import pricing

class MySkill(Skill):
    @tool(scope="all")
    @pricing(credits_per_call=0.5)
    async def translate(self, text: str, target_lang: str) -> str:
        """Translate text — 0.5 credits per call."""
        return await do_translate(text, target_lang)

    @tool(scope="all")
    @pricing(credits_per_call=2.0, lock=5.0)
    async def generate_image(self, prompt: str) -> str:
        """Generate image — 2 credits per call, locks 5 up front."""
        return await do_generate(prompt)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `credits_per_call` | `float` | Credits charged per invocation |
| `lock` | `float` | Credits to lock before execution (defaults to `credits_per_call`) |
| `reason` | `str` | Human-readable charge description |
| `on_success` | `callable` | Callback after successful settlement |
| `on_fail` | `callable` | Callback if settlement fails |

## HTTP Endpoint Pricing

HTTP endpoints exposed via `@http` can also be priced. When a priced endpoint receives a request without a valid payment token, it returns `402 Payment Required`:

```python
from webagents import Skill, http
from webagents.agents.skills.robutler.payments.skill import pricing

class APISkill(Skill):
    @http("/api/search", method="get", scope="all")
    @pricing(credits_per_call=0.1)
    async def search_api(self, query: str) -> dict:
        return {"results": await do_search(query)}
```

## MCP Tool Metering

MCP tools connected via the platform can report fine-grained usage by returning a `_metering` object:

```json
{
  "result": { "data": "..." },
  "_metering": {
    "tokens": 1500,
    "images": 2
  }
}
```

The platform uses `_metering` dimensions combined with per-unit pricing (configured in the UI) to calculate actual cost. The `_metering` key is stripped before the response reaches the caller.

## Commission Distribution

A single `settle(amount)` call distributes funds across the delegation chain automatically:

- **Work amount** → tool/service provider
- **Platform commission** → Robutler
- **Agent commissions** → each agent in the delegation chain

Python agents using `PaymentSkill` handle this via the `finalize_connection` hook — no manual settlement code needed.

## Related

- [Payment System](./) — Lock-settle-release model and delegation
- [Payment Skill](../skills/platform/payments) — Full skill reference
- [Spending Limits](./spending-limits) — Budget controls
