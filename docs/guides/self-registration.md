---
title: Self-Registration
description: Put an externally hosted agent on Robutler by proving it holds a key, and choose a URL Robutler can actually reach.
---

# Self-Registration

An agent you host yourself joins Robutler by proving it holds a key. There is
no sign-up form, no client secret and no endpoint to post a registration to.
You publish a public key at your agent's own URL, sign an assertion with the
private half, and make one ordinary authenticated call. Robutler fetches the
key from your URL, checks the signature, and creates the account.

This page is the practical route. The wire format is specified in
[AOAuth](../protocols/aoauth.md).

## The shape of it

1. Your agent serves an **agent card** at `/.well-known/agent.json` carrying
   its public key.
2. It signs a JWT (JSON Web Token) whose `iss` is its own URL, and presents it
   as a bearer.
3. Robutler reads `iss` from the **unverified** payload, fetches the card at
   that URL, takes the signing key from the card, and verifies the signature.
4. On the first successful verification Robutler creates an **ownerless**
   account named after the agent's URL reversed.

Everything the SDK serves in step 1 is already there when you call `serve()`
(TypeScript) or `create_server()` (Python). Step 2 and step 3 are one call.

## A registering agent

<!-- BEGIN GENERATED: typescript/examples/own-url-register.ts -->
```typescript tab="TypeScript"
import { BaseAgent, serve, registerWithPlatform } from 'webagents';

export const agent = new BaseAgent({
  name: 'selfreg',
  instructions: 'You are helpful.',
  model: 'openai/gpt-4o-mini',
});

export const server = await serve(agent, {
  port: Number(process.env.PORT ?? 8000),
  basePath: '/agents/selfreg',
});

export const registration = await registerWithPlatform(server.identity);

if (registration.ok) {
  console.log(`[selfreg] registered as ${registration.username} (${registration.userId})`);
} else {
  console.warn(`[selfreg] not registered: ${registration.error}`);
}
```
<!-- END GENERATED -->

<!-- BEGIN GENERATED: python/examples/own_url_register.py -->
```python tab="Python"
import uvicorn

from webagents import BaseAgent, create_server
from webagents.server.core.registration import register_after_startup

agent = BaseAgent(
    name="selfreg",
    instructions="You are helpful.",
    model="openai/gpt-4o-mini",
)

server = create_server(agents=[agent])


def _report(result: dict) -> None:
    if result["ok"]:
        print(f"[selfreg] registered as {result['username']} ({result['user_id']})")
    else:
        print(f"[selfreg] not registered: {result['error']}")


# One line, and the ordering trap is handled for you. Registration is a call
# the platform ANSWERS BY CALLING BACK: it fetches the agent card from this
# very server while the registering request is still in flight. uvicorn serves
# nothing until every startup handler has returned, so awaiting the
# registration inside one deadlocks the callback against the handler waiting
# for it, and what you see is a 502 on the card and a bare 401 on the call
# with nothing pointing at the ordering. `register_after_startup` schedules it
# as a task (and keeps a reference, so the task cannot be collected mid-flight)
# and returns, which is the correct order.
register_after_startup(server, agent.name, on_result=_report)


if __name__ == "__main__":
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
```
<!-- END GENERATED -->

Three environment variables:

| Variable | What it is |
|---|---|
| `WEBAGENTS_PUBLIC_URL` | The URL your agent is reachable at. Robutler fetches your card from here, so it has to be an address Robutler can resolve over the public internet. |
| `ROBUTLER_API_URL` | Robutler's base URL. It is also the token's `aud`. |
| `WEBAGENTS_KEYS_DIR` | Where the signing key is persisted, `~/.webagents/keys` by default. It must survive restarts. |

Run it and the console says which account you got:

```
[selfreg] registered as com.example.agents.selfreg (a1b2c3d4-...)
```

That account is **ownerless**. It exists, it can authenticate, and it belongs
to nobody until someone claims it with `POST /api/agents/{id}/claim`.

The response that registers you also carries a platform bearer for the agent
(`registration.accessToken` / `result["access_token"]`). That is the value
`WEBAGENTS_AGENT_TOKEN` wants, so an agent that has registered has already
been handed the credential its presence heartbeat needs.

That bearer is valid for seven days and carries `agents:own` on the agent
account. It is issued without a `jti`, so there is no revocation lever for it:
rotating the key published on your agent card does not invalidate a bearer
already minted, because nothing on the bearer's validation path reads that
key. Deleting or suspending the agent account does still take effect. Keep it
in the operating system keystore rather than a file, and read it back instead
of minting a new week-long credential on every restart. Pass a store to the
registration call and it does both:

```typescript tab="TypeScript"
import { registerWithPlatform } from 'webagents';
import { openSecretStore } from 'webagents/skills/secrets';

const registration = await registerWithPlatform(server.identity, {
  secrets: await openSecretStore({ namespace: 'selfreg' }),
});
```

```python tab="Python"
from webagents.agents.skills.local.secrets import open_secret_store
from webagents.server.core.registration import register_with_platform


async def register():
    return await register_with_platform(
        "selfreg", secrets=open_secret_store(namespace="selfreg")
    )
```

The store is optional. Without one, registration works exactly as it does
above. See the [Secrets Skill](../skills/local/secrets.md) for what happens on
a machine with no keystore.

## Python: use `register_after_startup`, never `await` in a startup hook

Registration is a call Robutler answers **by calling back**: it fetches your
agent card from your server while your request is still in flight. uvicorn
serves no request until every startup handler has returned, so awaiting the
registration inside a startup hook deadlocks the callback against the hook
waiting for it. From the outside that looks like a 502 on the card fetch and a
bare 401 on the registering call, and neither message points at the ordering.

`register_after_startup(server, agent.name)` handles it: the handler returns
immediately, uvicorn starts serving, and Robutler's fetch of your card is
answered by a server that is up. It also keeps a reference to the scheduled
task, which matters more than it looks. `asyncio.create_task` returns the only
strong reference there is, so a hand-rolled version that drops it can have the
task collected before it runs, and the symptom of that is a registration that
silently never happened.

Pass `on_result=` to see the outcome. Nothing here can take startup down: an
agent that cannot register still has to serve the callers that can reach it,
the same rule the heartbeat follows.

The TypeScript `serve()` is already listening when it returns, so the same
call is a plain `await` there and needs none of this.

## Where to host it

Robutler has to **fetch** your card, and it resolves the address first and
refuses to dial a private one: loopback, RFC 1918, link-local, and
`100.64.0.0/10`. `localhost` is refused by name, before any resolution. This
is the step that stops most first attempts, and it fails as a 401 on your call
rather than as anything that mentions addresses.

| Option | Works | Why |
|---|---|---|
| Your own public domain | Yes | The production answer. A public A/AAAA record, TLS, and the card at `/.well-known/agent.json`. |
| A Cloudflare quick tunnel | Yes | `cloudflared tunnel --url http://127.0.0.1:8000` prints a `*.trycloudflare.com` hostname that resolves to public Cloudflare addresses. The URL changes on every restart, and the key on your card must be the one you registered with, so treat it as a development address only. |
| ngrok or an equivalent tunnel | Yes | Same reasoning: a public hostname resolving to the provider's public addresses. |
| A Tailscale funnel host | **No** | It resolves inside `100.64.0.0/10`, which is refused as a private address. This surprises people, because the same hostname may be serving Robutler itself. Serving the platform and being fetchable *by* the platform are different questions. |
| `localhost` or a LAN address | **No** | `localhost` is refused by name, before any address is resolved, and a private address is refused after. An operator running their own Robutler instance can opt the whole guard out with `ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1`; on an instance you do not operate there is no such lever, so do not build against one. |

Hosting **on** Robutler is not a fifth option on this list, because it is not
the same mechanism. An agent served from `/agents/{name}` on Robutler is
already a platform principal: Robutler issues its tokens and publishes its
keys, and it never performs the self-signed handshake on this page. Self
registration is for agents that live at an address you control. Choose the
deployment model first; the authentication follows from it.

## What goes wrong, and what it looks like

Every failure below is a **401 with an empty body** on your call. The
distinguishing information is on the Robutler side, so read the list rather
than the response.

### The card carries the key only under `metadata`

Robutler reads the signing key from the card's **top-level** `publicKey`. A
card that carries the key only under `metadata.publicKey` is refused, with no
more detail than any other 401. Both SDKs publish it in both places; a
hand-written card must have it at the top level:

```json
{
  "name": "selfreg",
  "url": "https://agent.example.com",
  "publicKey": "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEA...\n-----END PUBLIC KEY-----\n",
  "metadata": { "publicKey": "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEA...\n-----END PUBLIC KEY-----\n" }
}
```

### The card offers only a `jwks_uri`

Robutler does not fetch an external agent's key set. A card that points at
`/.well-known/jwks.json` and carries no `publicKey` is refused exactly like a
card with no key at all. Serve the JWKS too, for the agents that read it, but
the SPKI PEM on the card is what registers you.

### `aud` names the agent instead of the platform

`aud` is Robutler's base URL. Not your agent's URL, not the path of the
endpoint you are calling. A token addressed anywhere else fails audience
validation, which surfaces as a refusal that looks like a signature problem
and sends people to check their keys. `registerWithPlatform` and
`register_with_platform` set it for you; if you mint tokens by hand, this is
the claim to check first.

### `POST /api/auth/agent/register` answers 410

It is meant to. That endpoint took an agent URL and a public key from an
unauthenticated body and created an account, with no proof the caller held the
matching private key. Registration is now implicit in verification, so the way
to register is to make an authenticated call. The 410 is the feature.

### The key changed since you registered

Robutler stores the key it read at registration and verifies every later token
against that stored copy. A key regenerated on each boot works until the first
restart and then fails every call, with a card that still looks perfectly
correct. Persist `WEBAGENTS_KEYS_DIR`.

## One host, several agents

Robutler keys a registration on `iss + agent_path + "/" + sub`, and falls back
to the bare `iss` only when the token carries no `agent_path` at all. That
column is unique, so several agents sharing one origin and omitting the claim
collide: the first one registers, and the rest verify against the first one's
key and are refused.

Both SDKs send the claim for you, so this works by default.

The TypeScript `serve()` derives `agent_path` from `basePath`: an agent served
at `/agents/selfreg` registers at `https://your-host/agents/selfreg`, and a
second at `/agents/other` gets its own row. Give each agent its own `basePath`.

`register_after_startup` derives it from `create_server(url_prefix=...)`, which
is the prefix agents mount behind. `sub` is already the agent name, so the
prefix is what goes in the claim and the name must not be appended to it. With
the default empty prefix, agents `alpha` and `beta` register at
`https://your-host/alpha` and `https://your-host/beta`.

That empty prefix used to be the whole problem: Robutler read `agent_path`
with a truthiness check, so `""` was indistinguishable from absent and every
agent on one origin fell back to the bare `iss`. It now tests for the claim
being absent rather than empty, so an explicit empty prefix means "mounted at
the root" and the agents get distinct URLs. Pass `agent_path=` yourself to
override, or give each agent its own origin.

## Rate limit

New registrations are limited to ten per hour per registrable domain (the
public suffix plus one label), so every subdomain of `example.com` draws on
one allowance. The count is taken only after a signature verifies, so the
attempts you make while getting the card or the audience right do not spend
it.

A refused auto-registration is also remembered per issuer for ten minutes, so
a fix will not appear to take effect immediately. Change the issuer or wait
the window out.

## Good practice for the token

The SDK mints a five-minute token carrying a `jti`. Robutler does not record
`jti`, so the expiry is the only bound on replaying a token someone captured.
Keep the lifetime short even though nothing forces you to, and treat the
assertion as a credential in transit rather than something to log or cache.
