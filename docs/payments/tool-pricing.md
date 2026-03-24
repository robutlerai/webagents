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
    @pricing(credits_per_call=0.5)
    @tool(scope="all")
    async def translate(self, text: str, target_lang: str) -> str:
        """Translate text — 0.5 credits per call."""
        return await do_translate(text, target_lang)

    @pricing(credits_per_call=2.0, lock=5.0)
    @tool(scope="all")
    async def generate_image(self, prompt: str) -> str:
        """Generate image — 2 credits per call, locks 5 up front."""
        return await do_generate(prompt)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `credits_per_call` | `float` | Credits charged per invocation |
| `lock` | `float \| (params) => float` | Credits to lock before execution. Can be a fixed number or a function of tool params. |
| `settle` | `(result, params) => float` | Credits to charge after execution. Overrides `_billing` / `credits_per_call` settlement. |
| `reason` | `str` | Human-readable charge description |
| `on_success` | `callable` | Callback after successful settlement |
| `on_fail` | `callable` | Callback if settlement fails |

## Dynamic Pricing Functions

For tools where cost depends on input parameters (e.g., video duration, image resolution) or output (e.g., actual API usage), use function-valued `lock` and `settle`:

### TypeScript

```typescript
import { Skill, tool, pricing } from '@anthropic/webagents';

class MediaSkill extends Skill {
  // Dynamic lock — quality-aware, reads rate_matrix
  @pricing({
    lock: (params) => {
      const duration = parseDuration(params.duration) || 5;
      const rate = RATE_MATRIX[`${params.resolution ?? '720p'}`] ?? 0.15;
      return duration * rate * 1.375; // markup + buffer
    },
    settle: (result, params) => {
      const billing = extractBilling(result);
      return billing.actual_units * billing.unit_price * 1.375;
    },
  })
  @tool({ description: 'Generate video' })
  async generateVideo(params: { prompt: string; duration: number; resolution?: string }) {
    return await this.callVideoAPI(params);
  }
}
```

### Python

```python
@pricing(
    lock=lambda params: estimate_cost(params['duration'], params.get('resolution', '720p')),
    settle=lambda result, params: extract_billing(result),
)
@tool(description='Generate video')
async def generate_video(self, prompt: str, duration: int = 5):
    ...
```

### Resolution Chain

When a tool is invoked, the payment skill resolves pricing in this order:

1. **`@pricing` decorator metadata** — checked first via `getPricingForTool`
2. **`tool.pricing` on plain objects** — for dynamically registered tools (MCP, mediagen)
3. **Database `toolPricing` config** — legacy `perCall` / `perUnit` fallback
4. **`defaultToolLock`** — last resort

If `lock` is a function, it receives the tool's input params and returns a dollar amount. If `settle` is defined, it receives the tool result and params after execution, overriding `_billing` metadata parsing.

## HTTP Endpoint Pricing

HTTP endpoints exposed via `@http` can also be priced. When a priced endpoint receives a request without a valid payment token, it returns `402 Payment Required`:

```python
from webagents import Skill, http
from webagents.agents.skills.robutler.payments.skill import pricing

class APISkill(Skill):
    @pricing(credits_per_call=0.1)
    @http("/api/search", method="get", scope="all")
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
