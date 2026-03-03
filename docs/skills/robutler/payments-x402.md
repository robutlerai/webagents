# PaymentSkillX402 - x402 Protocol Support

Full x402 payment protocol integration for WebAgents, enabling agents to provide and consume paid APIs using multiple payment schemes.

## Overview

PaymentSkillX402 extends [PaymentSkill](payments.md) with complete x402 protocol support, enabling agents to provide and consume paid APIs using multiple payment schemes including blockchain cryptocurrencies.

**Key Features**:

- ✅ All PaymentSkill functionality (token validation, cost calculation, hooks)
- ✅ **Agent B**: Expose paid HTTP endpoints with `@http` + `@pricing`
- ✅ **Agent A**: Automatic payment handling via hooks (no manual tool calls needed)
- ✅ Multiple payment schemes: robutler tokens, blockchain (USDC), etc.
- ✅ Cross-token exchange: convert crypto to credits automatically
- ✅ Standard x402 protocol: `scheme: "token", network: "robutler"`

## What is x402?

x402 is a payments protocol for HTTP, built on blockchain concepts. It allows HTTP APIs to require payment before serving requests, with standardized payment verification and settlement.

**Core Concepts**:

- **402 Payment Required**: HTTP status code indicating payment needed
- **Payment Requirements**: Structured spec of what payment types are accepted
- **Payment Header**: Cryptographic proof of payment included in `X-PAYMENT` header
- **Facilitator**: Third-party service that verifies and settles payments
- **Multiple Schemes**: Support for various payment types (tokens, blockchain, etc.)

Learn more: [x402 Protocol Specification](https://docs.cdp.coinbase.com/x402/)

## Installation

```bash
pip install webagents[robutler]
```

For blockchain payment support (optional):

```bash
pip install eth-account web3  # For Ethereum-based chains (Base, Polygon, etc.)
pip install solana  # For Solana
```

## Quick Start

### Agent B: Providing Paid APIs

Create an agent that exposes a paid HTTP endpoint:

```python
from webagents import BaseAgent
from webagents.agents.skills.robutler import PaymentSkillX402, pricing

# Create agent with x402 payment support
agent_b = BaseAgent(
    name="weather-api",
    api_key="your_robutler_api_key",
    skills={
        "payments": PaymentSkillX402(config={
            "accepted_schemes": [
                {"scheme": "token", "network": "robutler"}
            ]
        })
    }
)

# Add paid endpoint
@agent_b.http("/weather", method="get")
@pricing(credits_per_call=0.05, reason="Weather API call")
async def get_weather(location: str) -> dict:
    """Get weather for a location (costs 0.05 credits)"""
    return {
        "location": location,
        "temperature": 72,
        "conditions": "sunny"
    }
```

When called without payment, returns HTTP 402 with x402 V2 requirements:

```bash
curl http://localhost:8080/weather-api/weather?location=SF

# Response: HTTP 402 Payment Required
{
  "x402Version": 2,
  "accepts": [
    {
      "scheme": "token",
      "network": "robutler",
      "amount": "0.05",
      "asset": "USD",
      "resource": "/weather",
      "description": "Weather API call",
      "mimeType": "application/json",
      "payTo": "agent_weather-api",
      "maxTimeoutSeconds": 60,
      "extra": {
        "tokenPricing": {
          "creditsPerCall": 0.05,
          "chargeTypes": ["platform_fee", "platform_llm", "agent_fee"]
        }
      }
    }
  ]
}
```

With valid `X-PAYMENT` header:

```bash
curl -H "X-PAYMENT: <base64_payment_header>" \
  http://localhost:8080/weather-api/weather?location=SF

# Response: HTTP 200 OK
{
  "location": "SF",
  "temperature": 72,
  "conditions": "sunny"
}
```

### Agent A: Consuming Paid APIs

Create an agent that can automatically pay for services:

```python
from webagents import BaseAgent
from webagents.agents.skills.robutler import PaymentSkillX402

# Create agent with automatic payment
agent_a = BaseAgent(
    name="consumer",
    api_key="your_robutler_api_key",
    skills={
        "payments": PaymentSkillX402()
    }
)

# When the agent makes HTTP requests to paid endpoints:
# 1. Gets 402 response with payment requirements
# 2. Skill automatically creates payment
# 3. Retries with X-PAYMENT header
# 4. Returns result
# All automatic - no explicit payment code needed!
```

## Configuration

### Basic Configuration

```python
PaymentSkillX402(config={
    # Facilitator URL (default: /api/payments for lock, verify, settle)
    "facilitator_url": "https://robutler.ai/api/payments",
    
    # Payment schemes to accept (for Agent B)
    "accepted_schemes": [
        {"scheme": "token", "network": "robutler"}
    ],
    
    # Payment schemes to use (for Agent A)  
    "payment_schemes": ["token"],
    
    # Maximum payment amount (safety limit)
    "max_payment": 10.0,
    
    # Auto-exchange: convert crypto to credits
    "auto_exchange": True
})
```

### Multi-Scheme Support (Agent B)

Accept both robutler tokens and blockchain payments:

```python
PaymentSkillX402(config={
    "accepted_schemes": [
        {
            "scheme": "token",
            "network": "robutler"
        },
        {
            "scheme": "exact",
            "network": "base-mainnet",
            "wallet_address": "0xYourWallet..."
        }
    ]
})
```

### Blockchain Support (Agent A)

Enable direct blockchain payments:

```python
PaymentSkillX402(config={
    "wallet_private_key": "0x...",  # Your wallet private key
    "auto_exchange": True,  # Auto-convert crypto to credits
    "payment_schemes": ["token", "exact"]
})
```

## JWKS verification flow

When the X-PAYMENT header contains a JWT (e.g. from `POST /api/payments/lock`):

1. Decode the JWT header (unverified) to get `kid` and read `iss` from claims.
2. Fetch the issuer's public keys from `{iss}/.well-known/jwks.json` (cached with TTL/ETag).
3. Verify the JWT signature with RS256 and validate `exp`, `aud`.
4. Read `payment.balance` from claims. If verification succeeds, the verify API call can be skipped.
5. Settlement still uses `POST /api/payments/settle` so the platform can deduct balance and credit the recipient.

This reduces latency when the payer uses JWT payment tokens.

## Payment Flow

### Lock-First Settlement Model

All payment flows use a **lock-first** model: a single lock (payment token) is created at the beginning of a conversation turn and reused for all settlements within that turn. Settlements happen in a fixed order:

1. **`platform_fee`** — platform margin (configurable via `PLATFORM_FEE_PERCENT`, default 20%)
2. **`platform_llm`** — LLM inference cost (when platform keys are used, not BYOK)
3. **`agent_fee`** — agent markup (configurable via `agent_pricing_percent`, default 100%)

### Two-Settle Model

Each tool invocation or LLM call may produce **two settlements** against the same lock:

- The **platform settle** (`platform_fee` + optionally `platform_llm`) happens first.
- The **agent settle** (`agent_fee`) happens second.

This ensures the platform always recovers its costs before the agent receives its markup.

### BYOK (Bring Your Own Key) Flow

When the user provides their own LLM API key:

1. The **native LLM skill** detects BYOK (user-supplied provider key exists).
2. LLM inference is routed through the user's key — no `platform_llm` charge.
3. Settlement routes to `agent_fee` only (plus `platform_fee` on the agent markup).
4. If no BYOK key exists, `platform_llm` is settled for the inference cost.

### Flow 1: Agent A → Agent B (Robutler Token)

```
1. Agent A calls Agent B's endpoint
   GET /weather?location=SF

2. Agent B returns HTTP 402 with x402 V2 payment requirements
   {
     "x402Version": 2,
     "accepts": [{"scheme": "token", "network": "robutler", "amount": "0.05", ...}]
   }

3. Agent A's PaymentSkillX402 hook:
   - Checks for compatible payment scheme
   - Uses existing token from context or API
   - Encodes payment header

4. Agent A retries request with X-PAYMENT header
   GET /weather?location=SF
   X-PAYMENT: <base64_payment_header>

5. Agent B's PaymentSkillX402 hook:
   - Verifies payment via facilitator /verify
   - Settles platform_fee first, then agent_fee
   - Allows request to proceed

6. Agent B returns result
   {"location": "SF", "temperature": 72}
```

### Flow 2: Agent A → Agent B (Crypto via Exchange)

```
1. Agent A calls Agent B, gets 402 with token scheme in accepts

2. Agent A has no token but has crypto wallet

3. Agent A's skill:
   - Calls facilitator /exchange GET (see rates)
   - Creates blockchain payment
   - Calls /exchange POST with crypto payment → gets robutler token

4. Agent A retries with new token in X-PAYMENT header

5. Agent B processes payment normally
```

### Flow 3: Agent A → Agent B (Direct Blockchain)

```
1. Agent B returns 402 with blockchain scheme (e.g., "exact:base-mainnet")

2. Agent A's PaymentSkillX402:
   - Creates blockchain payment using wallet
   - Includes x402 payment header

3. Agent B's PaymentSkillX402:
   - Validates/settles via CDP/x402.org proxy
   - Creates virtual token in Portal API

4. Subsequent requests use virtual token until depleted
```

## Payment Schemes

### Token (Robutler)

Platform credits with instant settlement:

- **Scheme**: `"token"`
- **Network**: `"robutler"`
- **Benefits**: Instant, no gas fees, best for agent-to-agent
- **Rate**: 1:1 USD

```json
{
  "scheme": "token",
  "network": "robutler",
  "amount": "0.05",
  "asset": "USD"
}
```

### Exact (Blockchain)

Direct USDC payments on various blockchains:

- **Scheme**: `"exact"`
- **Networks**: `"base-mainnet"`, `"solana"`, `"polygon"`, `"avalanche"`
- **Benefits**: Real blockchain settlement, decentralized
- **Note**: Gas fees covered by facilitator

```json
{
  "scheme": "exact",
  "network": "base-mainnet",
  "amount": "1.00",
  "asset": "USDC"
}
```

## Advanced Features

### The `@pricing` Decorator

The `@pricing` decorator supports a `lock` parameter for pre-authorization:

```python
# Static pricing with lock pre-auth
@agent.http("/weather", method="get")
@pricing(credits_per_call=0.05, reason="Weather API call", lock=True)
async def get_weather(location: str) -> dict:
    """Lock=True means the transport pre-authorizes the full amount before execution."""
    return {"location": location, "temperature": 72}

# Lock with LLM preauth — reserves enough for the tool + expected LLM cost
@agent.http("/analyze", method="post")
@pricing(credits_per_call=0.50, reason="Analysis", lock=True)
async def analyze(data: dict) -> dict:
    # The lock covers: platform_fee + platform_llm (estimated) + agent_fee
    return await self.llm.complete(f"Analyze: {data}")
```

When `lock=True`, the transport ensures a payment token with sufficient balance exists before the handler runs. If the token is insufficient, a mid-stream top-up flow is triggered (see UAMP Token Top-Up below).

### Dynamic Pricing

Use `PricingInfo` for dynamic pricing based on request params:

```python
from webagents.agents.skills.robutler import PricingInfo

@agent.http("/analyze", method="post")
@pricing(credits_per_call=None)  # Will be calculated dynamically
async def analyze_data(data: dict, complexity: str = "standard") -> dict:
    # Calculate price based on complexity
    base_price = 0.10
    if complexity == "advanced":
        base_price = 0.50
    elif complexity == "enterprise":
        base_price = 2.00
    
    # Return PricingInfo for dynamic pricing
    return PricingInfo(
        credits=base_price,
        reason=f"Data analysis ({complexity})",
        metadata={"complexity": complexity}
    )
```

### Payment Priority

Agent A tries payment methods in this order:

1. Existing robutler token from context
2. Robutler token from agent's token list (includes virtual tokens)
3. Exchange crypto for credits (if `auto_exchange=True` and wallet configured)
4. Direct blockchain payment (if wallet configured)

### Virtual Tokens

When Agent B receives direct blockchain payments, a "virtual token" is automatically created and tracked via the Robutler API. This allows subsequent requests to use the same payment source without repeated blockchain transactions.

## API Reference

### PaymentSkillX402

```python
class PaymentSkillX402(PaymentSkill):
    """Enhanced payment skill with x402 protocol support"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize PaymentSkillX402.
        
        Args:
            config: Configuration dict with:
                - facilitator_url: x402 facilitator endpoint
                - accepted_schemes: List of payment schemes to accept
                - payment_schemes: List of payment schemes to use
                - wallet_private_key: Private key for blockchain payments
                - auto_exchange: Enable automatic crypto-to-credits exchange
                - max_payment: Maximum payment amount (safety limit)
        """
```

### Hooks

#### `check_http_endpoint_payment`

```python
@hook("before_http_call", priority=10, scope="all")
async def check_http_endpoint_payment(self, context) -> Any:
    """
    Intercept HTTP endpoint calls requiring payment.
    
    For Agent B:
    - Checks if endpoint has @pricing decorator
    - If no X-PAYMENT header: raises PaymentRequired402
    - If X-PAYMENT present: verifies and settles via facilitator
    """
```

### Helper Methods

#### `_get_available_token`

```python
async def _get_available_token(self, context) -> Optional[str]:
    """
    Get available robutler payment token.
    
    Checks:
    1. Context payment token (from PaymentSkill)
    2. Agent's token list via API (includes virtual tokens)
    
    Returns:
        Token string or None
    """
```

#### `_create_payment`

```python
async def _create_payment(
    self,
    accepts: List[Dict],
    context
) -> tuple[str, str, float]:
    """
    Create payment for one of the accepted schemes.
    
    Priority:
    1. scheme='token', network='robutler' with existing token
    2. scheme='token' via exchange (if auto_exchange)
    3. Direct blockchain payment
    
    Returns:
        (payment_header, scheme_description, cost)
    """
```

#### `_exchange_for_credits`

```python
async def _exchange_for_credits(
    self,
    amount: float,
    context
) -> str:
    """
    Exchange cryptocurrency for robutler token.
    
    Args:
        amount: Amount of credits needed
        context: Request context
        
    Returns:
        Token string
        
    Raises:
        X402ExchangeFailed: If exchange fails
    """
```

## Exceptions

### PaymentRequired402

```python
class PaymentRequired402(X402Error):
    """
    HTTP 402 Payment Required exception.
    
    Raised when an HTTP endpoint requires payment.
    Contains x402 payment requirements in details.
    """
```

### X402UnsupportedScheme

```python
class X402UnsupportedScheme(X402Error):
    """Raised when no compatible payment scheme is available"""
```

### X402VerificationFailed

```python
class X402VerificationFailed(X402Error):
    """Raised when payment verification fails"""
```

### X402SettlementFailed

```python
class X402SettlementFailed(X402Error):
    """Raised when payment settlement fails"""
```

### X402ExchangeFailed

```python
class X402ExchangeFailed(X402Error):
    """Raised when crypto-to-credits exchange fails"""
```

## Examples

### Example 1: Simple Paid API

```python
from webagents import BaseAgent
from webagents.agents.skills.robutler import PaymentSkillX402, pricing

agent = BaseAgent(
    name="translator",
    skills={"payments": PaymentSkillX402()}
)

@agent.http("/translate", method="post")
@pricing(credits_per_call=0.10, reason="Translation service")
async def translate(text: str, target_lang: str) -> dict:
    # Translation logic here
    return {"translated": f"[{target_lang}] {text}"}
```

### Example 2: Multi-Tier Pricing

```python
@agent.http("/compute", method="post")
@pricing(credits_per_call=None)
async def compute(task: dict, tier: str = "basic") -> dict:
    prices = {
        "basic": 0.05,
        "standard": 0.20,
        "premium": 1.00
    }
    
    # Calculate result
    result = perform_computation(task)
    
    # Return with pricing
    return PricingInfo(
        credits=prices[tier],
        reason=f"Computation ({tier} tier)",
        metadata={"tier": tier, "task_size": len(task)}
    )
```

### Example 3: Consumer Agent with Auto-Exchange

```python
consumer = BaseAgent(
    name="api-consumer",
    skills={
        "payments": PaymentSkillX402(config={
            "wallet_private_key": os.getenv("WALLET_KEY"),
            "auto_exchange": True,
            "max_payment": 5.0
        })
    }
)

# This agent can automatically:
# - Use existing tokens
# - Exchange crypto for credits when needed
# - Make blockchain payments as fallback
```

## Roborum Payments API (`/api/payments/*`)

The Roborum platform exposes payment endpoints that implement the x402 flow. Payment tokens are **RS256-signed JWTs**; they can be verified locally via JWKS or via the verify endpoint.

### JWT payment tokens

- **Issuer**: `https://robutler.ai` (or `JWT_ISSUER`).
- **Claims**: `sub` (user id), `aud`, `exp`, `jti`, and `payment: { balance, scheme }`.
- **Verification**: Same RS256 public key as auth; fetch from `{iss}/.well-known/jwks.json`.

### POST /api/payments/lock

Create a payment authorization (lock funds, issue JWT):

```json
// Request
{ "amount": 0.05, "audience": ["agent-id"], "expiresIn": 3600 }

// Response
{ "token": "<JWT>", "expiresAt": "2025-11-01T00:00:00Z", "lockedAmount": 0.05 }
```

### POST /api/payments/verify

Validate a payment token (optional when using local JWKS verification):

```json
// Request (body or X-PAYMENT header)
{ "token": "<JWT>" }

// Response
{ "valid": true, "balance": 0.05, "expiresAt": "...", "issuer": "https://robutler.ai" }
```

### POST /api/payments/settle

Charge against the token (always required for settlement):

```json
// Request
{ "token": "<JWT>", "amount": 0.05, "recipientId": "agent-id", "description": "API call", "resource": "/path" }

// Response
{ "success": true, "charged": 0.05, "remaining": 0 }
```

### GET /api/payments/tokens

List current user's payment tokens (active, expired, depleted).

### GET /.well-known/jwks.json

Public keys for JWT verification (RS256). Used by the skill for **local verification** to avoid a network call when the X-PAYMENT header contains a JWT.

### GET /.well-known/ucp

UCP capability manifest (version, issuer, capabilities, endpoints: lock, verify, settle, tokens).

## Facilitator API (legacy /api/x402)

The skill also supports the legacy x402 facilitator interface:

### POST /x402/verify

Verify payment validity:

```python
# Request
{
  "paymentHeader": "<base64>",
  "paymentRequirements": {
    "scheme": "token",
    "network": "robutler",
    "maxAmountRequired": "0.05",
    ...
  }
}

# Response
{
  "isValid": true
}
```

### POST /x402/settle

Settle verified payment:

```python
# Response
{
  "success": true,
  "transactionHash": "robutler-tx-1234567890"
}
```

### GET /x402/supported

List supported payment schemes:

```python
# Response
{
  "schemes": [
    {"scheme": "token", "network": "robutler", "description": "..."},
    {"scheme": "exact", "network": "base-mainnet", "description": "..."}
  ]
}
```

### GET /x402/exchange

Get exchange rates (Robutler extension):

```python
# Response
{
  "supportedOutputTokens": [
    {"scheme": "token", "network": "robutler", ...}
  ],
  "exchangeRates": {
    "exact:base-mainnet:USDC": {
      "outputScheme": "token",
      "rate": "1.0",
      "minAmount": "0.01",
      "fee": "0.02"
    }
  }
}
```

### POST /x402/exchange

Exchange crypto for credits (Robutler extension):

```python
# Request
{
  "paymentHeader": "<base64_blockchain_payment>",
  "paymentRequirements": {...},
  "requestedOutput": {
    "scheme": "token",
    "network": "robutler",
    "amount": "9.80"
  }
}

# Response
{
  "success": true,
  "token": "tok_xxx:secret_yyy",
  "amount": "9.80",
  "expiresAt": "2025-11-01T00:00:00Z"
}
```

## Best Practices

### Security

1. **Never expose private keys**: Store wallet private keys in environment variables
2. **Set max_payment limits**: Prevent accidental overpayment
3. **Validate pricing**: Always verify pricing before accepting payments
4. **Use HTTPS**: Never send payment headers over unencrypted connections

### Performance

1. **Token reuse**: Existing tokens are cached and reused when possible
2. **Async operations**: All payment operations are async for non-blocking execution
3. **Connection pooling**: HTTP client uses connection pooling for efficiency

### Error Handling

```python
from webagents.agents.skills.robutler.payments_x402 import (
    PaymentRequired402,
    X402VerificationFailed,
    X402SettlementFailed
)

try:
    result = await agent.call_endpoint("/paid-api")
except PaymentRequired402 as e:
    print("Payment required:", e.payment_requirements)
except X402VerificationFailed as e:
    print("Payment verification failed:", e.message)
except X402SettlementFailed as e:
    print("Payment settlement failed:", e.message)
```

## Comparison with PaymentSkill

| Feature | PaymentSkill | PaymentSkillX402 |
|---------|--------------|------------------|
| Tool charging | ✅ Yes | ✅ Yes (inherited) |
| Cost calculation | ✅ Yes | ✅ Yes (inherited) |
| HTTP endpoint payments | ❌ No | ✅ Yes |
| x402 protocol | ❌ No | ✅ Yes |
| Multiple payment schemes | ❌ No | ✅ Yes |
| Blockchain payments | ❌ No | ✅ Yes (optional) |
| Crypto exchange | ❌ No | ✅ Yes |
| Automatic payment | ❌ No | ✅ Yes |

## UAMP Payment Flow

When agents are accessed over UAMP (WebSocket), payment negotiation happens **inline**
using UAMP payment events instead of HTTP 402 responses. The flow is:

1. Client sends `input.text` (or `response.create`)
2. Agent skill raises `PaymentTokenRequiredError` (no token in context)
3. UAMP transport catches the error and sends `payment.required` event to client
4. Client obtains a payment token (e.g. calls `/api/payments/lock` on Roborum)
5. Client sends `payment.submit` event with the token
6. UAMP transport sets `context.payment_token` and retries `process_uamp`
7. Agent processes successfully, streams `response.delta` / `response.done`
8. UAMP transport sends `payment.accepted` with remaining balance

### Mid-Stream UAMP Token Top-Up

When a lock's balance is insufficient mid-turn (e.g., an expensive tool call or long LLM generation), the transport triggers a **top-up flow**:

1. Agent (or native LLM skill) detects the token balance is too low for the next charge.
2. UAMP transport sends `payment.required` with `extra.action: "topup"` and the additional `amount` needed.
3. Client calls `POST /api/payments/tokens/{id}/topup` to add funds to the existing token.
4. Client sends `payment.submit` with the updated token (same `jti`, higher balance).
5. Transport resumes processing with the topped-up token — no retry, no lost state.

The top-up flow uses `wait_for_event("payment.submit")` to block the agent's execution until the client responds, preserving streaming state.

### UAMP Payment Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `payment.required` | Server → Client | Agent needs payment; includes `requirements.amount`, `requirements.currency`, `requirements.schemes` |
| `payment.submit` | Client → Server | Client provides token via `payment.scheme`, `payment.amount`, `payment.token` |
| `payment.accepted` | Server → Client | Payment verified; includes `payment_id`, `balance_remaining` |
| `payment.balance` | Server → Client | Balance update notification (low balance warning) |
| `payment.error` | Server → Client | Payment failed; includes `code`, `message`, `can_retry` |

### Pre-loading tokens via session.update

Clients can pre-load a payment token before sending any input:

```json
{
  "type": "session.update",
  "session": {
    "payment_token": "eyJhbGciOiJSUzI1NiIsI..."
  }
}
```

This sets `context.payment_token` so the first request doesn't trigger `payment.required`.

### Daemon bridge (Roborum → Agent)

When Roborum routes messages to agents via the UAMP daemon bridge:

1. Router calls `sendInputToAgentSession()` with `senderId` and `agentId`
2. If daemon returns `payment.required`, WS server calls `findOrCreatePaymentToken(senderId, { audience: [agentId] })`
3. WS server sends `payment.submit` back to daemon with the JWT token
4. Daemon retries the agent with the token in context
5. On success, daemon sends `response.done`; on payment failure, `payment.error`

### Python example

```python
from webagents.agents.skills.core.transport.uamp.skill import UAMPTransportSkill

# UAMP transport automatically handles payment negotiation.
# PaymentSkill reads context.payment_token (set by transport).
agent = BaseAgent(
    name="paid-agent",
    skills=[UAMPTransportSkill(), PaymentSkillX402({"enable_billing": True})]
)
```

### TypeScript example

```typescript
import { PaymentX402Skill } from 'webagents/skills/payments/x402';
import { PortalTransportSkill } from 'webagents/skills/transport/portal';

// Portal transport handles payment.required/submit/accepted over WS.
const agent = new Agent({
  name: 'paid-agent',
  skills: [new PortalTransportSkill(), new PaymentX402Skill()],
});
```

## See Also

- [PaymentSkill](payments.md) - Basic payment integration
- [Transport Payment Handling](../../agent/transports.md#payment-handling) - Per-transport payment behavior
- [x402 Protocol](https://docs.cdp.coinbase.com/x402/) - Official specification
- [Robutler Platform](https://robutler.ai) - Platform documentation
- [WebAgents Documentation](../../index.md) - Main documentation

