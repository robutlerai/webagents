# UCP Skill - Universal Commerce Protocol

The UCP Skill enables WebAgents to participate in the [Universal Commerce Protocol](https://ucp.dev) ecosystem, allowing agents to:

- **Discover** UCP-compliant merchants and their capabilities
- **Create** checkout sessions and manage purchases
- **Process** payments using multiple handlers (Stripe, Google Pay, Robutler)

## Installation

The UCP skill is part of the WebAgents ecosystem skills. Ensure you have the required dependencies:

```bash
# Core dependencies
pip install aiohttp pydantic

# Optional: For Stripe payments
pip install stripe

# Optional: UCP SDK for schema validation
pip install ucp-sdk
```

## Quick Start

### 1. Add to Agent Configuration

```yaml
# agent.yaml
name: shopping-assistant
skills:
  ucp:
    enabled_handlers:
      - ai.robutler.token
      - com.stripe.payments.card
      - google.pay
    default_currency: USD
```

### 2. Use in Agent Code

```python
from webagents.agents.skills.ecosystem.ucp import UCPSkill

# Create skill with configuration
ucp_skill = UCPSkill(config={
    "enabled_handlers": ["ai.robutler.token", "com.stripe.payments.card"],
    "stripe_api_key": "sk_test_...",  # Optional
})

# Add to agent
agent.add_skill(ucp_skill)
```

### 3. Agent Tools Available

Once initialized, the agent has access to these tools:

| Tool | Description |
|------|-------------|
| `discover_merchant` | Discover a merchant's UCP capabilities |
| `create_checkout` | Create a checkout session |
| `complete_purchase` | Complete checkout with payment |
| `get_checkout_status` | Get current checkout status |
| `list_payment_handlers` | List available payment handlers |

## Operating Modes

The UCP skill supports three operating modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `client` | Discover merchants, create checkouts, pay | Agents that purchase services |
| `server` | Accept checkouts, verify payments, sell services | Agents that sell capabilities |
| `both` | Full bidirectional commerce | Agents that both buy and sell |

### Client Mode (Default)

```yaml
skills:
  ucp:
    mode: client
    enabled_handlers:
      - ai.robutler.token
      - com.stripe.payments.card
```

### Server Mode (Merchant)

```yaml
skills:
  ucp:
    mode: server
    agent_description: "AI assistant for data analysis"
    accepted_handlers:
      - ai.robutler.token
      - google.pay
    services:
      - id: data_analysis
        title: Data Analysis
        description: Comprehensive data analysis report
        price: 2500  # $25.00 in cents
        tool_name: analyze_data  # Tool to execute after purchase
      - id: quick_summary
        title: Quick Summary
        description: Brief summary of your data
        price: 500  # $5.00
```

### Both Modes

```yaml
skills:
  ucp:
    mode: both
    # Client config
    enabled_handlers:
      - ai.robutler.token
    # Server config  
    agent_description: "Multi-purpose AI agent"
    services:
      - id: my_service
        title: My Service
        price: 1000
```

## Server Mode Features

When running in server mode, the agent exposes UCP-compliant HTTP endpoints:

### Endpoints Exposed

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/ucp` | GET | Agent's UCP profile for discovery |
| `/ucp/services` | GET | List services for sale |
| `/checkout-sessions` | POST | Create checkout session |
| `/checkout-sessions/{id}` | GET | Get checkout status |
| `/checkout-sessions/{id}` | PUT | Update checkout |
| `/checkout-sessions/{id}/complete` | POST | Complete with payment |

### Server Mode Commands

| Command | Description |
|---------|-------------|
| `/ucp/server` | Show server mode status |
| `/ucp/services` | List services for sale |
| `/ucp/orders` | List received orders |
| `/ucp/profile` | Show UCP profile |

### Server Mode Tools

| Tool | Description |
|------|-------------|
| `register_service` | Register a new service for sale |
| `list_services` | List available services |
| `list_orders` | List orders received |

### Example: Agent Selling Services

```python
from webagents.agents.skills.ecosystem.ucp import UCPSkill

# Create skill in server mode
ucp_skill = UCPSkill(config={
    "mode": "server",
    "agent_description": "Expert data analyst",
    "services": [
        {
            "id": "full_analysis",
            "title": "Full Data Analysis",
            "description": "Complete analysis with insights and recommendations",
            "price": 5000,  # $50.00
            "tool_name": "run_analysis"  # Tool executed after payment
        }
    ]
})

agent.add_skill(ucp_skill)

# Other agents can now:
# 1. Discover this agent at /.well-known/ucp
# 2. See services at /ucp/services
# 3. Create checkout and pay
```

### Example: Agent-to-Agent Commerce

```python
# Agent A (buyer) discovers Agent B (seller)
discovery = await agent_a.discover_merchant("https://agent-b.webagents.ai")
# Returns services Agent B offers

# Agent A creates checkout
checkout = await agent_a.create_checkout(
    merchant_url="https://agent-b.webagents.ai",
    items=[{"id": "full_analysis", "quantity": 1}],
    buyer_email="agent-a@webagents.ai"
)

# Agent A pays with Robutler tokens
result = await agent_a.complete_purchase(
    checkout_id=checkout["checkout_id"],
    payment_handler="ai.robutler.token"
)

# Agent B receives payment and can fulfill the service
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled_handlers` | list | All handlers | Payment handler namespaces to enable |
| `agent_profile_url` | str | webagents.ai/profile | Agent's UCP profile URL |
| `default_currency` | str | "USD" | Default currency for transactions |
| `stripe_api_key` | str | None | Stripe secret API key |
| `robutler_token` | str | None | Pre-configured Robutler token |

### Environment Variables

- `UCP_AGENT_PROFILE_URL` - Agent profile URL
- `STRIPE_API_KEY` - Stripe API key

## Payment Handlers

### Robutler Credits (`ai.robutler.token`)

Uses Robutler payment tokens from the WebAgents platform.

```python
# Token from context (automatic)
result = await agent.complete_purchase(
    checkout_id="...",
    payment_handler="ai.robutler.token"
)

# Or with explicit token
result = await agent.complete_purchase(
    checkout_id="...",
    payment_handler="ai.robutler.token",
    payment_credentials={"token": "tok_id:tok_secret"}
)
```

### Stripe (`com.stripe.payments.card`)

Supports multiple credential types:

```python
# With PaymentMethod ID (recommended)
result = await agent.complete_purchase(
    checkout_id="...",
    payment_handler="com.stripe.payments.card",
    payment_credentials={"payment_method_id": "pm_..."}
)

# With Stripe token (from Stripe.js)
result = await agent.complete_purchase(
    checkout_id="...",
    payment_handler="com.stripe.payments.card",
    payment_credentials={"token": "tok_..."}
)
```

### Google Pay (`google.pay`)

Accepts Google Pay payment data from the web SDK:

```python
# With payment data from Google Pay SDK
result = await agent.complete_purchase(
    checkout_id="...",
    payment_handler="google.pay",
    payment_credentials={"payment_data": google_pay_response}
)
```

## Slash Commands

Available commands for interactive use:

| Command | Description |
|---------|-------------|
| `/ucp` | Show help and subcommands |
| `/ucp discover <url>` | Discover merchant capabilities |
| `/ucp checkout <url> <items>` | Create checkout session |
| `/ucp status [id]` | Get status or list sessions |
| `/ucp complete <id>` | Complete checkout |
| `/ucp handlers` | List payment handlers |

### Examples

```bash
# Discover a merchant
/ucp discover https://shop.example.com

# Create checkout
/ucp checkout https://shop.example.com '[{"id": "widget-1", "title": "Widget", "quantity": 2}]'

# Check status
/ucp status abc123-checkout-id

# Complete purchase
/ucp complete abc123-checkout-id
```

## Workflow Example

### Complete Purchase Flow

```python
# 1. Discover merchant
discovery = await agent.discover_merchant(
    merchant_url="https://flowers.example.com"
)
print(f"Can transact: {discovery['can_transact']}")
print(f"Payment methods: {discovery['payment_handlers']}")

# 2. Create checkout
checkout = await agent.create_checkout(
    merchant_url="https://flowers.example.com",
    items=[
        {"id": "bouquet_roses", "title": "Red Roses", "quantity": 1}
    ],
    buyer_email="customer@example.com",
    buyer_name="John Doe",
    discount_codes=["10OFF"]
)
print(f"Checkout ID: {checkout['checkout_id']}")
print(f"Total: {checkout['total_formatted']}")

# 3. Complete with payment
if checkout['ready_for_payment']:
    result = await agent.complete_purchase(
        checkout_id=checkout['checkout_id'],
        payment_handler="ai.robutler.token"
    )
    
    if result.get('payment_successful'):
        print("Purchase complete!")
    elif result.get('requires_escalation'):
        print(f"Complete in browser: {result['continue_url']}")
```

## Handling Escalation

When a checkout requires user interaction (browser handoff):

```python
result = await agent.complete_purchase(checkout_id="...")

if result.get('requires_escalation'):
    # Provide URL to user for browser completion
    continue_url = result['continue_url']
    print(f"Please complete your purchase: {continue_url}")
```

## Testing

### Using UCP Playground

Test against the official UCP playground:

```bash
# Run integration tests
python -m pytest tests/test_ucp_integration.py -v

# Or test manually
python -c "
import asyncio
from webagents.agents.skills.ecosystem.ucp import UCPClient

async def test():
    client = UCPClient()
    # Test with UCP sample server
    result = await client.discover_and_negotiate('http://localhost:8182')
    print(result)

asyncio.run(test())
"
```

### Running Unit Tests

```bash
pytest tests/test_ucp_skill.py -v
```

## Architecture

```
ucp/
├── __init__.py          # Public exports
├── skill.py             # Main UCPSkill class
├── client.py            # UCP REST client
├── discovery.py         # Merchant discovery
├── schemas.py           # Pydantic models
├── exceptions.py        # Error classes
├── handlers/            # Payment handlers
│   ├── base.py          # Abstract interface
│   ├── robutler.py      # Robutler tokens
│   ├── stripe.py        # Stripe payments
│   └── google_pay.py    # Google Pay
└── README.md            # This file
```

## UCP Specification

This skill implements [UCP v2026-01-11](https://ucp.dev/specification/overview/).

Key concepts:
- **Capabilities**: What operations a merchant supports (checkout, discounts, fulfillment)
- **Payment Handlers**: How payments are processed (Stripe, Google Pay, etc.)
- **Negotiation**: Agent and merchant agree on compatible capabilities
- **Escalation**: Browser handoff when agent can't complete autonomously

## Troubleshooting

### "Handler not available"

Ensure the handler is in `enabled_handlers` and properly configured:

```python
UCPSkill(config={
    "enabled_handlers": ["ai.robutler.token"],
    # For Stripe, also need:
    "stripe_api_key": "sk_..."
})
```

### "Cannot transact with merchant"

The merchant may not support capabilities or payment handlers the agent supports. Check discovery result:

```python
result = await agent.discover_merchant("https://merchant.com")
print(result['unsupported'])  # Missing capabilities
print(result['payment_handlers'])  # Available handlers
```

### Checkout stuck in "incomplete"

Missing required information. Check the checkout response for messages:

```python
status = await agent.get_checkout_status(checkout_id)
print(status)  # Check for missing fields
```

## References

- [UCP Specification](https://ucp.dev/specification/overview/)
- [UCP Python SDK](https://pypi.org/project/ucp-sdk/)
- [UCP Playground](https://ucp.dev/playground/)
- [Google Blog: Under the Hood UCP](https://developers.googleblog.com/under-the-hood-universal-commerce-protocol-ucp/)
- [Shopify: Building UCP](https://shopify.engineering/ucp)
