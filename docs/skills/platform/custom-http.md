# Custom HTTP

Expose a user-authored function as an HTTP endpoint on the agent.

## Entry shape

```yaml
skills:
  custom_http:
    endpoints:
      - id: stripe_webhook
        method: POST
        path: /webhooks/stripe
        auth: signature        # public | signature | session | portal_token
        use: stripeHandler
        description: Stripe webhook receiver
```

The dispatcher mounts the endpoint under `/api/agents/<idOrUsername>/<path>` and supports `:param` route templates (`/items/:id` matches `/items/abc` and yields `ctx.request.params.id = 'abc'`).

## Auth modes

| Mode           | Who verifies                              | When to use                                                |
| ---            | ---                                       | ---                                                        |
| `public`       | the function itself                       | unauth'd webhooks; internal IP allowlists                  |
| `signature`    | the function (HMAC via `ctx.portal.verifyHmac`) | provider webhooks (Stripe, Slack, Twilio, …)         |
| `session`      | the dispatcher (owner-session check)      | owner-only management endpoints                            |
| `portal_token` | the dispatcher (RS256 via JWKS)           | platform-issued bearer tokens (payment, AOAuth, service)   |

## ctx.request

Functions see the raw headers (lower-cased), the parsed body, optionally `rawBody` (when manifest declares `permissions.rawBody`), and the route-template params.

```js
export default async function handler(ctx) {
  const sig = ctx.request.headers['stripe-signature'];
  const ok = await ctx.portal.verifyHmac({
    algo: 'sha256',
    secretBinding: 'STRIPE_WEBHOOK_SECRET',
    payload: ctx.request.rawBody,
    expected: sig,
  });
  if (!ok) return new Response('bad signature', { status: 400 });
  const event = JSON.parse(new TextDecoder().decode(ctx.request.rawBody));
  // ... handle event ...
  return new Response('ok');
}
```

## See also

- [Functions](functions.md) — declaring the function
- [Portal helpers](portal-helpers.md) — `verifyToken`, `verifyHmac`
