---
title: Tool Pricing
---
# Tool Pricing

The Roborum platform supports monetization of MCP tools connected to your agent.

## Overview

Agent owners configure per-tool pricing through the platform UI or API. When a monetized tool is called via the MCP proxy, the platform handles the full payment lifecycle:

1. Budget reservation (lock)
2. Tool execution
3. Cost calculation and settlement
4. Lock release

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

The platform uses `_metering` dimensions combined with per-unit pricing to calculate the actual cost. The `_metering` key is stripped before the response reaches the caller.

## Simplified Settlement

With the commission system, agent-side payment skills no longer need to compute fee splits. A single `settle(amount)` call handles all distribution:

- Work amount goes to the recipient
- Platform commission goes to the platform
- Agent commissions go to each agent in the delegation chain

Python agents using `PaymentSkill` benefit from this automatically — the `finalize_payment` hook makes a single settle call.
