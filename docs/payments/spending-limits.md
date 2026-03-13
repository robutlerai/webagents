---
title: Spending Limits
description: Budget controls for access tokens and payment tokens.
---

# Spending Limits

Robutler enforces spending limits at multiple levels to prevent runaway costs.

## Access Token Limits

When creating an access token, you can set:

| Limit | Description |
|-------|-------------|
| `limitDaily` | Maximum spend per 24-hour period |
| `limitTotal` | Lifetime spending cap |
| `limitPerUse` | Maximum per-request charge |
| `expiresAt` | Token expiration date |

```bash
curl -X POST https://robutler.ai/api/access-tokens \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"name": "dev-token", "limitDaily": 500000000, "limitTotal": 5000000000}'
```

Amounts are in **nanocents** (1 dollar = 100,000,000 nanocents).

## Payment Token Limits

Payment tokens carry their balance as a JWT claim. The `balance` field caps total spending for that token. The `max_depth` field limits delegation chain depth.

## Platform Defaults

- **Daily cap**: $5.00 per access token (configurable)
- **Per-token limit**: Set at creation time
- **Delegation depth**: Max 5 levels by default

## Spending Overrides

Platform users can configure per-agent spending overrides via the API:

```
POST /api/balance/spending-overrides
GET  /api/balance/spending-overrides
```

This allows different limits for trusted vs. untrusted agents.
