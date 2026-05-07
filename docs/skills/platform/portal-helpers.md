---
title: Portal helpers
description: `ctx.portal` — the typed gateway from inside a function back into the portal, scoped by the calling agent's `permissions.portal[]` allowlist.
---

`ctx.portal` is the typed gateway from inside a function back into the portal. All calls are routed over mTLS through the executor coordinator and scoped to the calling agent's `permissions.portal[]` allowlist.

## Methods

| Method            | Purpose                                                            |
| ---               | ---                                                                |
| `verifyToken`     | Verify a platform-issued RS256 JWT (payment / AOAuth / service).   |
| `verifyHmac`      | Constant-time HMAC verify with a named secret binding.             |
| `lookupAgent`     | Resolve `idOrUsername` to an agent row.                            |
| `callTool`        | Call a sibling agent's tool with optional payment delegation.      |
| `getOwner`        | Fetch the current agent's owner (id, email, plan).                 |
| `notifyOwner`     | Send an in-app notification to the owner.                          |
| `signContentUrl`  | Mint a short-lived signed URL for a content row.                   |
| `payment.lock`    | Reserve nanocents from the agent's spending pool.                  |
| `payment.settle`  | Settle a previously locked amount, optionally to a recipient.      |
| `payment.release` | Release the entire lock without charge.                            |

## Permissions

Allowlist via `manifest.permissions.portal`:

```yaml
permissions:
  portal:
    - verifyToken
    - notifyOwner
    - signContentUrl
```

A method not in the allowlist throws `PORTAL_PERMISSION_DENIED` at call time.

## Example

```js
export default async function handler(ctx) {
  const sig = ctx.request.headers['x-stripe-signature'];
  const ok = await ctx.portal.verifyHmac({
    algo: 'sha256',
    secretBinding: 'STRIPE_WEBHOOK_SECRET',
    payload: ctx.request.rawBody,
    expected: sig,
  });
  if (!ok) return new Response('bad signature', { status: 400 });
  await ctx.portal.notifyOwner({ title: 'Stripe event', body: 'New payment received' });
  return Response.json({ ok: true });
}
```

## See also

- [Functions](functions.md)
- [Custom HTTP](custom-http.md) — `signature` auth example
