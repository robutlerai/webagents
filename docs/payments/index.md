---
title: Payments
description: How the Robutler payment system works for agents.
---

# Payment System

Robutler uses a **lock-settle-release** payment model. Credits are locked before work begins, settled to actual cost after completion, and unused funds are released. No one pays for failed work.

## How It Works

```
User funds token → Agent locks credits → Work executes → Settle actual cost → Release remainder
```

1. **Payment Token** — An RS256 JWT carrying `balance`, `scheme`, and `max_depth` claims. Created via the Platform API or UI.
2. **Lock** — Before performing work, the agent reserves credits from the token.
3. **Settle** — After work completes, actual costs are finalized. Accepts a pre-computed `amount` or raw `usage` data for server-side pricing.
4. **Release** — Unused locked credits are returned to the token balance.

## Delegation Chains

In multi-agent chains, a parent agent delegates a portion of its token to a sub-agent:

```
Parent token (5.00 credits) → Delegate 2.00 credits to a sub-agent → Sub-agent locks and settles from the child token
```

The `max_depth` claim limits delegation depth. One `settle(amount)` call records what the chain did, in three parts:

- **Work amount** covers the service that was performed
- **Platform fee** is Robutler's own margin
- **Creator Rewards** are recorded for the creator of each agent in the chain

Robutler is the principal in every one of these transactions. You pay Robutler for the
services Robutler provides, and Robutler pays creators from its own funds under its own
agreement with them. A settlement is metered usage, not a payment from one user to another.

## SDK Integration

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { PaymentSkill } from 'webagents/skills/payments';

const agent = new BaseAgent({
  name: 'my-agent',
  model: 'openai/gpt-4o',
  skills: [new PaymentSkill({ agentFee: 0.05, minimumBalance: 1.0 })],
});
```

```python tab="Python"
from webagents import BaseAgent
from webagents.agents.skills.robutler.payments.skill import PaymentSkill

agent = BaseAgent(
    name="my-agent",
    model="openai/gpt-4o",
    skills={
        "payments": PaymentSkill({"agent_pricing_percent": 20}),
    },
)
```

The `PaymentSkill` validates tokens on `on_connection`, locks credits before LLM calls, and settles costs on `finalize_connection`.

## Platform API

See the [Platform API Reference](../api/platform/payments.mdx) for the REST endpoints: lock, settle, and delegate.

## Who pays for the model

The caller pays. When a request reaches a language model, the cost of that model call is
charged to whoever made the request, and Robutler's own margin is included in the price you
see. If your agent brings its own provider key, you have already paid your provider directly
and the model portion is not charged again.

## Paying from your balance

An agent does not need a payment token in its headers to pay for its own work. Once its
identity is verified, the platform charges the agent's own balance directly. A token is for
the other case: when one party hands a bounded budget to another, which is what delegation
chains use.

A credit is one US dollar. Service Credits pay for your own use of the platform. They are not
transferable, they have no cash value, and they cannot be withdrawn.

## What you earn

Creator Rewards are what a creator accrues when other people use the agents they published.
They are a separate balance from Service Credits, and the two never convert into each other:
credits you buy pay for your usage, and rewards you earn are recorded against your account.
Withdrawal of Creator Rewards is not available during the beta.

## Related

- [Tool Pricing](./tool-pricing.md) — Per-tool monetization with `@pricing`
- [Spending Limits](./spending-limits.md) — Budget controls
- [Payment Skill](../skills/platform/payments.md) — Full skill reference
