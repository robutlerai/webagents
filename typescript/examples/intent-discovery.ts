/**
 * Publish what an agent can do, and find an agent by what you need.
 *
 * `PortalDiscoverySkill` gives the agent two things. A `search` tool its
 * model calls with a description of a NEED ("translate a contract into
 * German"), matched by meaning against the intents every agent on the
 * platform has published, so the caller never has to know a name or a URL.
 * And `publishIntents()`, which lists this agent's own intents so other
 * agents can find it the same way.
 *
 * Both calls are SIGNED with the identity `serve()` gives the agent (RFC 9421
 * HTTP Message Signatures under the Web Bot Auth profile), the same key that
 * signs its registration, so no platform key is involved: the platform takes
 * the signature first and looks for a bearer only when there is none. Set
 * WEBAGENTS_API_KEY only for an agent that has no signing identity.
 *
 * Order matters once, on the very first run. The platform fetches the key set
 * from the agent URL the first time it sees the signature, so the agent has
 * to be listening before it publishes: `serve()` returns listening,
 * `registerWithPlatform` makes the first signed call, `publishIntents` comes
 * after. From the second run on the platform already holds the key.
 *
 * Environment:
 *
 *   OPENAI_API_KEY         your model provider's key
 *   WEBAGENTS_PUBLIC_URL   the https base URL this agent is reachable at; with
 *                          `basePath` it composes the agent URL the platform
 *                          fetches the key set from
 *   ROBUTLER_API_URL       the platform's base URL (the signature covers its host)
 *   WEBAGENTS_KEYS_DIR     where the Ed25519 key is stored (default
 *                          ~/.webagents/keys). It MUST survive restarts.
 *
 * This file is executed by tests/unit/skills/discovery/example.test.ts against
 * a stub platform, and the docs' snippets are generated from it verbatim.
 */

import { BaseAgent, OpenAISkill, PortalDiscoverySkill, registerWithPlatform, serve } from 'webagents';

// What this agent DOES, in the words someone who needs it would use. Each
// intent is one sentence; the platform embeds them and matches a searcher's
// need against them by meaning, not by keyword.
export const discovery = new PortalDiscoverySkill({
  intents: [
    'translate documents between English and German',
    'proofread German business correspondence',
  ],
  description: 'Translates and proofreads English and German text.',
});

export const agent = new BaseAgent({
  name: 'translator',
  instructions: 'You translate and proofread English and German text.',
  model: 'openai/gpt-4o-mini',
  skills: [new OpenAISkill({ model: 'gpt-4o-mini' }), discovery],
});

// `serve()` persists the agent's signing key, publishes it at
// {agentUrl}/.well-known/jwks.json, and hands the identity to the agent.
export const server = await serve(agent, {
  port: Number(process.env.PORT ?? 8000),
  basePath: '/agents/translator',
});

// The first signed request registers the agent; this one also reports it.
export const registration = await registerWithPlatform(server.identity);
if (!registration.ok) console.warn(`[translator] not registered: ${registration.error}`);

// Now the platform can verify this agent's signature, so publish.
export const published = await discovery.publishIntents();
if (!published.ok) console.warn(`[translator] intents not published: ${published.error}`);

// The other side. The agent's model calls the `search` tool itself when it
// needs another agent; this is the same call made directly, for code that
// wants the answer rather than the model. Each result carries the matched
// intent, the agent that published it and a similarity score.
export async function findAgentFor(need: string) {
  return discovery.search({ query: need, types: ['intents'] });
}
