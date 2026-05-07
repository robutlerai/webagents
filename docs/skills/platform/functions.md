---
title: Functions
description: User-authored JavaScript that runs in a sandboxed executor and is invoked by other skills (cron, custom HTTP, custom tools).
---

User-authored JavaScript that runs in a sandboxed executor and is invoked by other skills (cron, custom HTTP, custom tools).

## What is a function?

A **function** is a content item (`kind: 'function'`) that exposes a single async `handler(ctx)` entry point. In v1 it runs in a per-tenant V8 isolate (`js-v1`) via `isolated-vm`, with strict limits on wall time, CPU, memory, and outbound fetch. Python (`python-pyodide-v1`) is deferred — see [ADR-0008](../../../docs/adr/0008-pyodide-deferred.md). `wasm-v1` is reserved for v2.

Functions are declared once at the top of `agent_configs.functions` and consumed by name from skills like `cron`, `custom_http`, and `custom_tools` — there is no separate "trigger" abstraction.

## Anatomy

```js
/**
 * @robutler-function
 * @runtime js-v1
 * @entrypoint handler
 * @permissions fetch=https://api.stripe.com,secrets=STRIPE_KEY,kv=rw
 * @limits wallMs=5000,cpuMs=200,memoryMb=64
 */
export default async function handler(ctx) {
  if (!ctx.auth.userId) return new Response('Unauthorized', { status: 401 });
  const event = await ctx.request.json();
  await ctx.kv.put(`stripe:event:${event.id}`, event);
  return Response.json({ ok: true });
}
```

The `ctx` object contains the verified caller (`ctx.auth`), request payload (`ctx.request`), KV store (`ctx.kv`), secrets (`ctx.secrets`), portal helpers (`ctx.portal`), and a recursion-bounded `ctx.fn` for calling other functions.

## Declaring a function

```yaml
# AGENT.md
functions:
  stripeHandler:
    contentId: ctn_abc123
    description: Verifies Stripe webhook signatures
    permissions:
      fetch: [https://api.stripe.com]
      secrets: [STRIPE_WEBHOOK_SECRET]
      kv: rw
      portal: [verifyHmac, notifyOwner]
    limits: { wallMs: 5000, cpuMs: 200, memoryMb: 64 }
```

## Consuming a function from a skill

Cron:

```yaml
skills:
  cron:
    schedules:
      - id: nightly_report
        cron: 0 9 * * *
        use: dailyReport
```

Custom HTTP:

```yaml
skills:
  custom_http:
    endpoints:
      - id: stripe_webhook
        method: POST
        path: /webhooks/stripe
        auth: signature
        use: stripeHandler
```

Custom tools (LLM-callable):

```yaml
skills:
  custom_tools:
    tools:
      - id: calc
        name: calculate
        description: Evaluate a math expression.
        use: calculator
        parameters:
          type: object
          properties: { expr: { type: string } }
          required: [expr]
```

## Secrets

Function secrets live in `memory(serverEncrypted=true, namespace='fn-secret:<functionName>')`. The owner sets values via the Functions pane Secrets tab; functions read them via `ctx.secrets.get('NAME')`. Values never travel through chat — the LLM-visible memory tool excludes the `fn-secret:*` namespace and the result sanitiser redacts any value pulled from it.

## Quotas

Three layers, most-restrictive wins:

1. **Plan ceiling** (`PLAN_FUNCTION_LIMITS[<plan>]`).
2. **Agent override** (`agent_configs.functionLimits`) — strictly below the plan ceiling.
3. **Manifest hint** (`permissions.limits`) — function self-cap.

Daily counters (invocations, CPU-ms, ingress bytes, egress bytes) are reset at UTC midnight; concurrency is live and reservable.

## Billing

Compute + request fee + ingress + egress. Pricing key is `tools["fn:<functionName>"]` under `agent_configs.toolPricing.tools`; charges flow through the existing `agent_fee` charge type.

## Validation

`POST /api/agents/[id]/functions/[name]/validate` — runs the manifest validator and forwards source to the executor's `/validate` endpoint. Validations have their own quota bucket so saving doesn't burn the runtime invocation budget.

## See also

- [Cron](cron.md)
- [Custom HTTP](custom-http.md)
- [Custom tools](custom-tools.md)
- [Host self-edit](host-self-edit.md)
- [Portal helpers](portal-helpers.md)
- [REST API — Functions](../../api/functions.md)
