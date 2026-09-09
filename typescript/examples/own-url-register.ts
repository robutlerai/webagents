/**
 * Self-registering agent on its own URL (U1 plus dynamic registration).
 *
 * `own-url-minimal.ts` serves everything the platform READS — the agent card
 * at `/.well-known/agent.json` (origin and agent prefix) with the signing key
 * at the top level and nested under `metadata`, plus `/.well-known/jwks.json`
 * and a presence heartbeat. What it never does is speak: a card nobody has
 * fetched is not a registration. This file adds the missing half, which is one
 * authenticated call.
 *
 * There is no endpoint to post a registration to. `POST /api/auth/agent/register`
 * exists and answers 410 on purpose. The platform registers an agent the first
 * time a request verifies: it reads `iss` off the unverified bearer, fetches
 * the card at that URL, imports the `publicKey` it finds there and checks the
 * signature. `registerWithPlatform` mints that bearer from the key `serve()`
 * persisted and makes the call.
 *
 * The token it mints is short lived (five minutes) and carries a `jti`. The
 * platform does not record `jti` yet, so the expiry is the only thing bounding
 * replay of a token someone captures; keep it short even though nothing
 * enforces that.
 *
 * Environment:
 *
 *   OPENAI_API_KEY         your model provider's key
 *   WEBAGENTS_PUBLIC_URL   the URL this agent is reachable at. The PLATFORM
 *                          fetches the card from here, so it has to resolve to
 *                          a public address: loopback, RFC 1918, link-local
 *                          and 100.64.0.0/10 (which is where Tailscale
 *                          addresses live) are all refused.
 *   ROBUTLER_API_URL       the platform's base URL. It is also the token's
 *                          `aud`, which is the single fact this flow most
 *                          often gets wrong: `aud` is the PLATFORM, never the
 *                          agent's own URL and never the endpoint being called.
 *   WEBAGENTS_KEYS_DIR     where the Ed25519 key is stored (default
 *                          ~/.webagents/keys). It MUST survive restarts:
 *                          registration pins the public key from the card and
 *                          verifies every later token against that copy.
 *
 * The agent registers as an OWNERLESS account named after its own URL
 * reversed, and the response says which one. Claim it with
 * `POST /api/agents/{id}/claim` to attach it to a person.
 *
 * This file is executed by tests/unit/examples.test.ts (with PORT=0 and no
 * platform URL, so registration reports its own absence rather than dialling
 * anything), and the docs' snippets are generated from it verbatim.
 */

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
