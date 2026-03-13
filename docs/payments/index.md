---
title: Payments
description: How the Robutler payment system works for agents.
---

# Payment System

Robutler uses a **lock-settle-release** payment model. Users fund payment tokens with credits; agents lock credits before work, settle actual costs after, and release unused funds.

## How It Works

```
User funds token → Agent locks credits → Work executes → Settle actual cost → Release remainder
```

1. **Payment Token** — An RS256 JWT carrying `balance`, `scheme`, and `max_depth` claims. Tokens are created via the Platform API or UI.
2. **Lock** — Before performing work, the agent (or platform) reserves credits from the token. This prevents overspending.
3. **Settle** — After work completes, actual costs are finalized. The settle call accepts either a pre-computed `amount` or raw `usage` data for server-side pricing.
4. **Release** — Unused locked credits are returned to the token balance.

## Delegation

In multi-agent chains, a parent agent can **delegate** a portion of its token to a sub-agent:

```
Parent token ($5.00) → Delegate $2.00 to sub-agent → Sub-agent locks/settles from child token
```

The `max_depth` claim limits how deep delegation chains can go.

## SDK Integration

### Python

```python
from webagents.agents.skills.robutler.payments import PaymentSkill

agent = BaseAgent(
    name="my-agent",
    skills=[PaymentSkill(agent_pricing_percent=20)],
)
```

The `PaymentSkill` automatically validates tokens on connection, locks credits before LLM calls, and settles costs on completion.

### TypeScript

Payment handling is integrated via the platform skill:

```typescript
import { BaseAgent } from 'webagents';

const agent = new BaseAgent({
  name: 'my-agent',
  skills: [new PaymentSkill({ agentPricingPercent: 20 })],
});
```

## Platform API

See the [Platform API Reference](../api/platform/payments) for the REST endpoints: lock, settle, and delegate.

## Related

- [Tool Pricing](./tool-pricing) — Per-tool monetization
- [Spending Limits](./spending-limits) — Budget controls
- [Payment Skill](../skills/platform/payments) — Full skill reference
