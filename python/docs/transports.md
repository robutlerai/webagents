# Transports

WebAgents Python supports five transport protocols:

## Completions

OpenAI-compatible `/chat/completions` endpoint. Default transport.

```python
agent = WebAgent(name="my-agent")
agent.run(port=8080)
# Accessible at http://localhost:8080/chat/completions
```

## UAMP

Universal Agentic Message Protocol — event-based communication.

```python
from webagents.transports import UampTransport

agent = WebAgent(
    name="my-agent",
    transports=[UampTransport()],
)
```

## A2A

Google Agent-to-Agent protocol.

```python
from webagents.transports import A2ATransport

agent = WebAgent(
    name="my-agent",
    transports=[A2ATransport()],
)
```

## Realtime

WebSocket-based real-time communication.

## ACP

Agent Communication Protocol for structured task orchestration.
