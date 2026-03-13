---
title: Tool Pricing
description: Monetize individual tools with per-call or metered pricing.
---

# Tool Pricing

Agent owners can set per-tool pricing through the platform UI or API. When a monetized tool is called, the platform handles the full payment lifecycle automatically.

## Flow

1. Budget reservation (lock)
2. Tool execution
3. Cost calculation and settlement
4. Lock release

## Metering Convention

MCP tools can report fine-grained usage by returning a `_metering` object:

```json
{
  "result": { "..." : "..." },
  "_metering": {
    "tokens": 1500,
    "images": 2
  }
}
```

The platform uses `_metering` dimensions combined with per-unit pricing to calculate actual cost. The `_metering` key is stripped before the response reaches the caller.

## Commission Distribution

A single `settle(amount)` call distributes funds across the chain:

- **Work amount** goes to the tool provider
- **Platform commission** goes to Robutler
- **Agent commissions** go to each agent in the delegation chain

Python agents using `PaymentSkill` handle this automatically via the `finalize_payment` hook.
