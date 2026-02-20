# UAMP (Universal Agentic Message Protocol)

UAMP provides a standardized event-based protocol for agent communication.

## Event Types

| Event | Direction | Description |
|-------|-----------|-------------|
| `session.create` | Client → Agent | Initialize session |
| `session.created` | Agent → Client | Session confirmed |
| `input.text` | Client → Agent | Send text message |
| `response.create` | Client → Agent | Request response |
| `response.text.delta` | Agent → Client | Streaming text chunk |
| `response.done` | Agent → Client | Response complete |
| `payment.required` | Agent → Client | Payment needed |
| `payment.accepted` | Client → Agent | Payment confirmed |

## Adapter Layer

The UAMP adapter maps between UAMP events and your agent's message handler:

```python
from webagents.transports import UampTransport

transport = UampTransport(
    ws_url="ws://localhost:8080/uamp",
)

# UAMP events are automatically translated to/from your handler
@agent.on_message
async def handle(message: str) -> str:
    return "Response via UAMP"
```

## Payment Events

When an agent requires payment, it sends a `payment.required` event:

```json
{
  "type": "payment.required",
  "amount": 0.25,
  "currency": "USD",
  "description": "Code review service"
}
```

The platform handles payment authorization through the spending limits system.
