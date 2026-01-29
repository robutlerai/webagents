# UCP Commerce Example

This example demonstrates agent-to-agent commerce using the Universal Commerce Protocol (UCP).

## Agents

### Merchant Agent (`ucp-merchant`)
- **Mode**: Server
- **Sells**: Data analysis services
- **Services**:
  - Quick Analysis ($5.00)
  - Deep Analysis ($25.00)
  - Summary Report ($10.00)
- **Accepts**: Robutler tokens, Google Pay

### Client Agent (`ucp-client`)
- **Mode**: Client
- **Can**: Discover merchants, create checkouts, complete purchases
- **Pays with**: Robutler tokens

## Running the Example

### 1. Start the Server

```bash
cd /Users/vs/dev/webagents
python examples/ucp_commerce/run_commerce_demo.py
```

This starts a server with both agents at:
- Merchant: `http://localhost:8000/ucp-merchant`
- Client: `http://localhost:8000/ucp-client`

### 2. Test Discovery

```bash
# Client discovers merchant
curl -X POST http://localhost:8000/ucp-client/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Discover what services http://localhost:8000/ucp-merchant offers"}]
  }'
```

### 3. Test Checkout Flow

```bash
# Client creates checkout
curl -X POST http://localhost:8000/ucp-client/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Buy the quick analysis service from http://localhost:8000/ucp-merchant"}]
  }'
```

### 4. Direct UCP API Access

```bash
# Get merchant's UCP profile
curl http://localhost:8000/ucp-merchant/.well-known/ucp

# List merchant's services
curl http://localhost:8000/ucp-merchant/ucp/services

# Create checkout directly
curl -X POST http://localhost:8000/ucp-merchant/checkout-sessions \
  -H 'Content-Type: application/json' \
  -d '{
    "line_items": [{"item": {"id": "quick_analysis"}, "quantity": 1}],
    "buyer": {"email": "client@example.com"}
  }'
```

## Running Tests

```bash
# Run UCP commerce integration tests
cd /Users/vs/dev/webagents
pytest tests/test_ucp_commerce.py -v
```

## Architecture

```
┌─────────────────┐         UCP Protocol         ┌─────────────────┐
│                 │                              │                 │
│  Client Agent   │◄────────────────────────────►│ Merchant Agent  │
│  (UCP Client)   │                              │  (UCP Server)   │
│                 │  1. /.well-known/ucp         │                 │
│  - discover     │  2. /checkout-sessions       │  - services     │
│  - checkout     │  3. /complete                │  - orders       │
│  - pay          │                              │  - verify       │
│                 │                              │                 │
└─────────────────┘                              └─────────────────┘
```

## Payment Flow

1. **Discovery**: Client calls `/.well-known/ucp` on merchant
2. **Negotiation**: Client and merchant agree on capabilities/handlers
3. **Checkout**: Client creates checkout session with items
4. **Payment**: Client provides payment instrument (Robutler token)
5. **Verification**: Merchant verifies payment
6. **Fulfillment**: Merchant marks order complete, can execute service
