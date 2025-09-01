# Payment Skill

Payment processing and billing skill for the Robutler platform. This skill enforces billing policies up-front and finalizes charges when a request completes.

## Key Features
- Payment token validation during `on_connection` (returns 402 if required and missing)
- LLM cost calculation using LiteLLM `cost_per_token`
- Tool pricing via optional `@pricing` decorator (results logged to `context.usage` by the agent)
- Final charging based on `context.usage` at `finalize_connection`
- Optional async/sync `amount_calculator` to customize total charge
- Transaction creation via Portal API
- Depends on `AuthSkill` for user identity propagation

## Configuration
- `enable_billing` (default: true)
- `agent_pricing_percent` (percent, e.g., `20` for 20%)
- `minimum_balance` (USD required to proceed; 0 allows free trials without up-front token)
- `robutler_api_url`, `robutler_api_key` (server-to-portal calls)
- `amount_calculator` (optional): async or sync callable `(llm_cost_usd, tool_cost_usd, agent_pricing_percent_percent) -> float`
  - Default: `(llm + tool) * (1 + agent_pricing_percent_percent/100)`

## Example: Add Payment Skill to an Agent
```python
from webagents.agents import BaseAgent
from webagents.agents.skills.robutler.auth.skill import AuthSkill
from webagents.agents.skills.robutler.payments import PaymentSkill

agent = BaseAgent(
    name="paid-agent",
    model="openai/gpt-4o",
    skills={
        "auth": AuthSkill(),  # Required dependency
        "payments": PaymentSkill({
            "enable_billing": True,
            "agent_pricing_percent": 20,   # percent
            "minimum_balance": 1.0 # USD
        })
    }
)
```

## Tool Pricing with @pricing Decorator (optional)

The PaymentSkill provides a `@pricing` decorator to annotate tools with pricing metadata. Tools can also return
explicit usage objects and will be accounted from `context.usage` during finalize.

```python
from webagents.agents.tools.decorators import tool
from webagents.agents.skills.robutler.payments import pricing, PricingInfo

@tool
@pricing(credits_per_call=0.05, reason="Database query")
async def query_database(sql: str) -> dict:
    """Query database - costs 0.05 credits per call"""
    return {"results": [...]}

@tool  
@pricing()  # Dynamic pricing
async def analyze_data(data: str) -> tuple:
    """Analyze data with variable pricing based on complexity"""
    complexity = len(data)
    result = f"Analysis of {complexity} characters"
    
    # Simple complexity-based pricing: 0.001 credits per character
    credits = max(0.01, complexity * 0.001)  # Minimum 0.01 credits
    
    pricing_info = PricingInfo(
        credits=credits,
        reason=f"Data analysis of {complexity} chars",
        metadata={"character_count": complexity, "rate_per_char": 0.001}
    )
    return result, pricing_info
```

### Pricing Options

1. **Fixed Pricing**: `@pricing(credits_per_call=0.05)` (0.05 credits per call)
2. **Dynamic Pricing**: Return `(result, PricingInfo(credits=0.15, ...))`
3. **Conditional Pricing**: Override base pricing in function logic

### Cost Calculation

- **LLM Costs**: Calculated in `finalize_connection` using LiteLLM `cost_per_token(model, prompt_tokens, completion_tokens)`
- **Tool Costs**: Read from tool usage records in `context.usage` (e.g., a record with `{"pricing": {"credits": ...}}`), which are appended automatically by the agent when a priced tool returns `(result, usage_payload)`
- **Total**: If `amount_calculator` is provided, its return value is used; otherwise `(llm + tool) * (1 + agent_pricing_percent_percent/100)`

## Example: Validate a Payment Token
```python
from webagents.agents.skills import Skill, tool

class PaymentOpsSkill(Skill):
    def __init__(self):
        super().__init__()
        self.payment = self.agent.skills["payment"]

    @tool
    async def validate_token(self, token: str) -> str:
        """Validate a payment token"""
        result = await self.payment.validate_payment_token(token)
        return str(result)
```

## Hook Integration

The PaymentSkill uses BaseAgent hooks for lifecycle, but cost aggregation is done at finalize:

- **`on_connection`**: Validate payment token and check balance. If `enable_billing` and no token is provided while `minimum_balance > 0`, a 402 error is raised and processing stops. `finalize_connection` will still run for cleanup but will be a no-op.
- **`on_message`**: No-op (costs are computed at finalize)
- **`after_toolcall`**: No-op (tool costs come from usage records)
- **`finalize_connection`**: Aggregate from `context.usage`, compute final amount, and charge the token. If there are costs but no token, a 402 error is raised.

## Context Namespacing

The PaymentSkill stores data in the `payments` namespace of the request context:

```python
from webagents.server.context.context_vars import get_context

context = get_context()
payments_data = getattr(context, 'payments', None)
payment_token = getattr(payments_data, 'payment_token', None) if payments_data else None
```

## Usage Tracking

All usage is centralized on `context.usage` by the agent:

- LLM usage records are appended after each completion (including streaming final usage chunk).
- Tool usage is appended when a priced tool returns `(result, usage_payload)`; the agent unwraps the result and stores `usage_payload` as a `{type: 'tool', pricing: {...}}` record.

At `finalize_connection`, the Payment Skill sums LLM and tool costs from `context.usage` and performs the charge.

## Advanced: amount_calculator

You can provide an async or sync `amount_calculator` to fully control the final charge amount:

```python
async def my_amount_calculator(llm_cost_usd: float, tool_cost_usd: float, agent_pricing_percent_percent: float) -> float:
    base = llm_cost_usd + tool_cost_usd
    # Custom logic here (e.g., tiered discounts)
    return base * (1 + agent_pricing_percent_percent/100)

payment = PaymentSkill({
    "enable_billing": True,
    "agent_pricing_percent": 15,  # percent
    "amount_calculator": my_amount_calculator,
})
```

If omitted, the default formula is used: `(llm + tool) * (1 + agent_pricing_percent/100)`.

## Dependencies

- **AuthSkill**: Required for user identity headers (`X-Origin-User-ID`, `X-Peer-User-ID`, `X-Agent-Owner-User-ID`). The Payment Skill reads them from the auth namespace on the context.

Implementation: `robutler/agents/skills/robutler/payments/skill.py`.

## Error semantics (402)

- Missing token while `enable_billing` and `minimum_balance > 0` ➜ 402 Payment Required
- Invalid or expired token ➜ 402 Payment Token Invalid
- Insufficient balance ➜ 402 Insufficient Balance

Finalize hooks still run for cleanup but perform no charge if no token/usage is present.