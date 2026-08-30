/**
 * Minimal agent on its own URL (U1).
 *
 * Build an agent, serve it. Nothing is hidden behind a helper: `serve()`
 * mounts `POST /agents/mini/chat/completions` (the endpoint the platform
 * dials) AND the platform registration surface — the agent card at
 * `/.well-known/agent.json` (origin *and* agent prefix) carrying
 * `metadata.publicKey` as an SPKI PEM, plus `/.well-known/jwks.json` and a
 * 60s presence heartbeat.
 *
 * Environment:
 *
 *   OPENAI_API_KEY         your model provider's key
 *   WEBAGENTS_PUBLIC_URL   the URL this agent is reachable at (card `url`)
 *   WEBAGENTS_KEYS_DIR     where the Ed25519 key is stored (default
 *                          ~/.webagents/keys). It MUST survive restarts:
 *                          registration pins the public key from the card.
 *   WEBAGENTS_AGENT_TOKEN  per-agent key; with ROBUTLER_API_URL set, this is
 *                          what the heartbeat presents
 *
 * `POST /agents/mini/chat/completions` requires an Authorization header — it
 * runs the model on your credit. Add an AuthSkill to have the credential
 * verified rather than merely required.
 *
 * This file is executed by tests/unit/examples.test.ts (with PORT=0), and the
 * docs' snippets are generated from it verbatim.
 */

import { BaseAgent, serve } from 'webagents';

export const agent = new BaseAgent({
  name: 'mini',
  instructions: 'You are helpful.',
  model: 'openai/gpt-4o-mini',
});

export const server = await serve(agent, {
  port: Number(process.env.PORT ?? 8000),
  basePath: '/agents/mini',
});
