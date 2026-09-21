/**
 * Self-registering agent on its own URL (U1 plus dynamic registration).
 *
 * `own-url-minimal.ts` serves everything the platform READS: the agent card
 * at `{agentUrl}/.well-known/agent.json`, which names itself (`client_id`,
 * `url`) and its key set (`jwks_uri`), the key set itself at
 * `{agentUrl}/.well-known/jwks.json`, and a presence heartbeat. What it never
 * does is speak: a card nobody has fetched is not a registration. This file
 * adds the missing half, which is one signed call.
 *
 * There is no endpoint to post a registration to. `POST /api/auth/agent/register`
 * exists and answers 410 on purpose. The platform registers an agent the first
 * time a request verifies: it reads the request's `Signature-Agent` header,
 * fetches the key set it names, selects the key by the signature's `keyid`
 * (the RFC 7638 thumbprint), checks the signature over the method, host,
 * path, query and body digest, and then reads the card. `registerWithPlatform`
 * signs that request (RFC 9421 HTTP Message Signatures, the Web Bot Auth
 * profile) with the key `serve()` persisted and makes the call.
 *
 * The signature is good for sixty seconds and carries a random nonce the
 * platform spends on first use and refuses on replay, so a captured request
 * is worthless once presented. It also covers the platform's host, so a
 * request signed for the platform verifies nowhere else.
 *
 * Environment:
 *
 *   OPENAI_API_KEY         your model provider's key
 *   WEBAGENTS_PUBLIC_URL   the base URL this agent is reachable at; with
 *                          `basePath` it composes the agent URL the platform
 *                          registers. The PLATFORM fetches the key set and
 *                          the card from there, so it has to be https and
 *                          resolve to a public address: loopback, RFC 1918,
 *                          link-local and 100.64.0.0/10 (which is where
 *                          Tailscale addresses live) are all refused.
 *   ROBUTLER_API_URL       the platform's base URL: where the signed request
 *                          is sent.
 *   WEBAGENTS_KEYS_DIR     where the Ed25519 key is stored (default
 *                          ~/.webagents/keys). It MUST survive restarts:
 *                          registration pins the key set it fetched and
 *                          verifies every later request against that copy.
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
