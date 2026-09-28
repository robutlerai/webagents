---
title: x402 Payments
description: Price an HTTP endpoint with @pricing and it answers standard x402 (v2 and v1) payment challenges, paid in Robutler credits or, on a self-hosted agent, on chain.
---

# x402 Payments

`@pricing` stacked on `@http` turns an agent's HTTP endpoint into a priced
API. The endpoint answers an unpaid request with a standard
[x402](https://docs.cdp.coinbase.com/x402/) payment challenge, verifies a
payment before the handler runs, and settles once after the handler answers.
Both SDKs answer the same requests the same way: x402 v2 in headers, with the
v1 body beside it for older clients.

## Pricing an Endpoint

Give the agent a payment skill, and stack `@pricing` on `@http`:

```typescript tab="TypeScript"
import { BaseAgent, Skill, http, pricing } from 'webagents';
import { PaymentSkill } from 'webagents/skills/payments';

class Quotes extends Skill {
  @pricing({ creditsPerCall: 0.01, reason: 'A quote' })
  @http({ path: '/quote', method: 'GET', description: 'A price quote' })
  async quote(): Promise<Response> {
    return Response.json({ price: 42 });
  }
}

const agent = new BaseAgent({
  name: 'quotes',
  instructions: 'You quote prices.',
  skills: [new Quotes(), new PaymentSkill({ x402: { nonceSecret: process.env.X402_NONCE_SECRET } })],
});
```

```python tab="Python"
from webagents import BaseAgent, Skill
from webagents.agents.skills.robutler.payments import pricing
from webagents.agents.skills.robutler.payments_x402 import PaymentSkillX402
from webagents.agents.tools.decorators import http


class Quotes(Skill):
    @http("/quote", method="get")
    @pricing(credits_per_call=0.01, reason="A quote")
    async def quote(self) -> dict:
        """A price quote"""
        return {"price": 42}


agent = BaseAgent(
    name="quotes",
    instructions="You quote prices.",
    skills={"quotes": Quotes(), "payments": PaymentSkillX402()},
)
```

The decorators stack in a different order in the two languages, as above:
`@pricing` outermost in TypeScript, `@http` outermost in Python. The price is
in credits. An endpoint without `@pricing` stays free, and a priced endpoint on
an agent with no payment skill is not served free: it answers `503`
`payment_not_configured`.

The payment skill finds the platform and the agent's key the way the other
platform skills do, so an agent published from its folder, or given
`WEBAGENTS_AGENT_TOKEN`, needs no more. `X402_NONCE_SECRET` is the secret
behind the challenges; set the same value on every process that serves the
agent, or a challenge one process issued is refused by another.

## What a Caller Sees

1. **No payment:** `402 Payment Required`, with the offers in the
   `PAYMENT-REQUIRED` header (x402 v2, base64 JSON) and a v1 JSON body, and
   `Cache-Control: no-store`. Each offer in `accepts` names a scheme, a
   network, an amount, an asset and who is paid.
2. **The retry** carries the payment for one of the offers, in
   `PAYMENT-SIGNATURE` (v2) or `X-PAYMENT` (v1). The paywall matches it to the
   offer, verifies it, and only then runs the handler.
3. **The handler answered below 400:** the payment is settled, once, and the
   answer goes out with `PAYMENT-RESPONSE` (v2) or `X-PAYMENT-RESPONSE` (v1).
   A handler that answers 400 or above is not charged, and its answer goes out
   as it is.
4. **The settle failed:** `402`, with the failed settle in the response header
   and an empty body.

A payment pays for one request. A challenge covers the resource, the amount
and the request itself (method, path, query and a hash of the body), so the
paid retry must be the request the `402` answered, and a payment sent again,
to the same process or another, is refused with a `402` and an empty body.
The payment headers are exposed over CORS, so a browser client can read them.

## The Credits Scheme

The first offer is always Robutler credits: scheme `robutler-credits` on
network `robutler:1`, asset `credits`, paid to `robutler`. The payload is the
caller's Robutler payment token (`{"token": "..."}`), which a person gets from
a signed-in session and an agent from the platform when it delegates. The
paywall verifies the token against the platform, and the settle charges it.

Robutler is the seller of record: the caller's credits pay Robutler for the
call, and the agent's creator earns Creator Rewards on it, as with priced
tools. See [Payments](../../payments/index.md).

A standard x402 client that only knows the on-chain schemes skips this offer
and pays on chain when the agent also offers that.

## Metered Endpoints

A price with `lock` above `creditsPerCall` (`credits_per_call` in Python) is
metered: the challenge asks for up to `lock`, and the handler names what the
call actually cost:

```python tab="Python"
from webagents import Skill
from webagents.agents.skills.robutler.payments import PricingInfo, pricing
from webagents.agents.tools.decorators import http


class Reports(Skill):
    @http("/report", method="get")
    @pricing(lock=0.05)
    async def report(self) -> tuple:
        """A report, priced by its size."""
        return {"rows": 7}, PricingInfo(credits=0.002, reason="7 rows")
```

Only the actual amount is settled.

## On-Chain Payments on a Self-Hosted Agent

A self-hosted agent can also offer the x402 chain schemes (`exact`, and `upto`
for a metered endpoint) to an address you choose, verified and settled through
an x402 facilitator. Robutler is not a party to such a payment. The simplest
switch is the environment:

| Variable | What it sets |
|---|---|
| `X402_PAY_TO` | The receiving address; setting it turns chain payments on |
| `X402_NETWORK` | The CAIP-2 network, `eip155:84532` (Base Sepolia) unless set |
| `X402_ASSET`, `X402_ASSET_DECIMALS` | The token contract and its decimals, Base Sepolia USDC unless set |
| `X402_ASSET_NAME`, `X402_ASSET_VERSION` | The token's EIP-712 domain |
| `X402_FACILITATOR_URL` | The facilitator; `https://x402.org/facilitator` (testnets) unless set |
| `CDP_API_KEY_ID`, `CDP_API_KEY_SECRET` | A CDP key pair; the CDP facilitator is then used unless `X402_FACILITATOR_URL` says otherwise |

The same settings go under `x402: { chain: {...} }` in the payment skill's
configuration (`payTo`, `network`, `asset`, `decimals`, `schemes`,
`facilitator`). A settle a facilitator reports as pending is not a failure:
the answer is delivered and the pending settle rides along with its
transaction hash.

## Describing the Endpoint

The challenge carries the endpoint's URL and its `description`. A priced
endpoint can also declare how it is called and what it returns
(`discovery` on `@http`), which is published in the challenge as x402's
Bazaar extension, so x402 indexes can list it.

## Hosted Agents

An agent hosted on Robutler prices a no-code HTTP endpoint with a `price` in
its configuration, and answers the same credits challenge. See
[Custom HTTP](../platform/custom-http.md).

## Chat Turns

A priced agent charges callers of its chat, A2A and UAMP routes through a
payment token, not a challenge like this one; see
[Payment Skill](../platform/payments.md) and
[Transports](../../agent/transports.md#payment-handling).

## The machine-payments buyer

The same package ships `MppBuyer`, the buying half of Machine Payments: it reads a `402` from a Robutler resource, pays the one challenge on it with a card or a stablecoin payment source, and replays the original request. Its twin is re-exported from `webagents/skills/payments` in TypeScript, and the two behave the same.

| Export | What it is |
|---|---|
| `MppBuyer` | The buyer. `paying_fetch` / `payingFetch` wraps a send, `purchase` buys without a request to replay. |
| `MppBuyerPolicy` | Spend ceilings, the purchase timeout and optional seller overrides. |
| `MppBuyerPersistence` | Where a pending credential and a purchase record are kept across a restart. |
| `CardPaymentSource`, `StablecoinPaymentSource` | The two ways to pay. Each is given the challenge and returns a credential. |
| `SptRequest`, `TempoTransferRequest`, `MppTermsRequest` | What a payment source is handed. |
| `MppPurchaseOutcome`, `MppPurchaseRecord`, `MppPendingCredential`, `MppBuyerRefusal` | The results. |
| `MppRedirectError` | Raised when the host answers with a redirect. |

Seven rules worth knowing before you wire a payment source to it:

- **A paid 2xx is handed to you unread.** A streamed answer is the common case for a paid call, so the buyer does not consume the body it is supposed to give back. It reads a 2xx only when the response declares a length, and then only through a clone, bounded at 64 KiB; anything else reaches you still streaming. Python puts back whatever it had to peek at.
- **A purchase pointer is followed only under a daily cap.** Some answers carry an `mpp` requirement that names a `purchase_url` and no challenge, because whatever sent it had not verified who was asking. That is a pointer, not a challenge: `purchaseAt` / `purchase_at` asks the purchase URL itself with a signed `POST`, is challenged there as its own identity, and pays under the same policy as any other purchase. Nothing is sent unless the hosts of both the purchase URL and whatever named it are on `policy.realms`, and nothing is sent unless a daily cap is set (`dailyCapCents` / `daily_cap_cents`); without one the refusal reason is `pointer_needs_daily_cap`. A pointer carries no secret, so any peer reachable through an allowed host can write one, and the per-call purchase limit bounds one call while only the daily cap bounds the total across calls.
- **The replayed call drops your payment token.** After a purchase the buyer re-sends the original request without the exhausted token, in all three of its spellings (`x-payment-token`, `x-payment`, `?payment_token=`), and keeps the body and your other headers. A resource serves a newly purchased balance to a signed request that names no token, so re-sending the spent one could only be refused again.
- **A redirect is never followed.** Every send refuses redirects outright, and a redirect that some other transport already followed is caught too. The refusal reason is `redirect_refused`. A challenge is bound to the resource that issued it, so following a redirect would offer your payment credential to whatever the `Location` header named.
- **Every send is bounded.** The default purchase timeout is 60 seconds, settable as `purchase_timeout_seconds` / `purchaseTimeoutSeconds` on the policy. In TypeScript the buyer enforces that bound itself; in Python it holds for a client the buyer opens, so a client you pass in brings its own timeout.
- **The seller is pinned for an hour.** The buyer reads the seller's `/openapi.json` once and keeps the Stripe profile id and the stablecoin deposit address it finds there for an hour, then checks every challenge against them. A challenge that names a different recipient is refused `seller_not_pinned` before any payment source is called. A document naming two values for one method pins nothing, and a stale pin is never used as a fallback when the document cannot be re-read.
- **A challenge naming a reserved header is dropped.** A `Payment` challenge may nominate the header its credential rides in. The buyer refuses about twenty names the signer, the buyer or the HTTP client owns (`signature`, `signature-input`, `signature-agent`, `content-digest`, `content-length`, `cookie`, the `robutler-*` covered headers and the hop-by-hop set), because writing one of them would either break the signature or leak something. The challenge is then simply unreadable and the refusal reason is the generic `no_challenge`.

## Platform API

The platform routes behind these flows (verify, lock, settle, delegate and the
token list) are in the [Platform API reference](../../api/platform/payments.mdx).
Payment tokens are RS256-signed JWTs, verifiable against the platform's
`/.well-known/jwks.json`.

## See Also

- [Payment Skill](../platform/payments.md): pricing tools and settling chat turns
- [Transports](../../agent/transports.md#payment-handling): how each transport asks for a token
- [x402](https://docs.cdp.coinbase.com/x402/): the protocol
