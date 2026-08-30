/**
 * Minimal platform-connected agent (reverse WebSocket bridge, U2).
 *
 * The same two steps as the own-URL example, plus one skill.
 * `PortalConnectSkill` dials the platform, so no public URL and no inbound
 * port are needed; `serve()` is still here because it owns the lifecycle that
 * opens the socket (and it gives you /health and the agent card for free).
 *
 * The skill reads its own configuration — attaching it IS the configuration:
 *
 *   WEBAGENTS_PORTAL_URL   e.g. https://robutler.ai (`/ws` appended)
 *   WEBAGENTS_AGENT_TOKEN  a PER-AGENT key from POST /api/agents/{id}/api-key
 *   OPENAI_API_KEY         (or your model provider's key)
 *
 * The token must be per-agent: its JWT carries an `agent_id` claim. A generic
 * owner key connects successfully and then never receives a single turn, so
 * the skill refuses it at start with the fix in the message.
 *
 * For a process with no HTTP surface at all, see portal-connect-socket-only.ts.
 *
 * This file is executed by tests/unit/examples.test.ts against a stub portal,
 * and the docs' snippets are generated from it verbatim.
 */

import { BaseAgent, PortalConnectSkill, serve } from 'webagents';

export const agent = new BaseAgent({
  name: 'mini',
  instructions: 'You are helpful.',
  model: 'openai/gpt-4o-mini',
  skills: [new PortalConnectSkill()],
});

export const server = await serve(agent, {
  port: Number(process.env.PORT ?? 8000),
  basePath: '/agents/mini',
});
