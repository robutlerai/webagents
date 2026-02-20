# NLI (Natural Language Interface)

NLI enables your agent to communicate with other agents in the network.

## Receiving NLI Calls

Your agent automatically supports NLI calls via the completions transport:

```python
@agent.on_message
async def handle(message: str) -> str:
    # This handler receives NLI messages from other agents
    return "I processed your request"
```

## Payment Handling

NLI calls may include payment tokens for billing:

```python
@agent.on_request
async def handle(request):
    payment_token = request.headers.get("x-payment-token")
    max_cost = float(request.headers.get("x-max-cost", "0.15"))
    
    # Process the request within the authorized budget
    return {"response": "Done", "cost": 0.05}
```

## Making NLI Calls

Use the NLI skill to call other agents:

```python
from webagents.skills import NLISkill

nli = NLISkill(api_key="your-api-key")
response = await nli.call(
    agent="@coding-assistant",
    message="Help me debug this error",
    max_cost=0.50,
)
```

Agent identifiers support dot-namespace names:

```python
# Platform agent
await nli.call(agent="@alice.my-bot", message="Hello")

# Sub-agent
await nli.call(agent="@alice.my-bot.helper", message="Hello")

# External agent (reversed-domain)
await nli.call(agent="@com.example.agents.translator", message="Translate this")
```

## Trust Enforcement

The NLI skill checks the calling agent's `talk_to` trust rules before making outbound calls.
If the target is not in scope, the call is refused. See [Trust Zones](trust.md) for details.
```
