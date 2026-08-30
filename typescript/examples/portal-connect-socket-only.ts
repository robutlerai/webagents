/**
 * Platform-connected agent with NO HTTP server at all.
 *
 * The honest minimal form when nothing will ever dial this process: there is
 * no port to bind, so there is no server — just the skill's own lifecycle,
 * started explicitly and then kept alive. Written out rather than hidden
 * behind a one-word call, because what the process is doing (open a socket,
 * then wait) is the whole point.
 *
 * Prefer portal-connect-minimal.ts unless you specifically want no HTTP
 * surface: `serve()` gives you /health and the agent card, and it owns the
 * same lifecycle for you.
 *
 * Environment: WEBAGENTS_PORTAL_URL, WEBAGENTS_AGENT_TOKEN, OPENAI_API_KEY.
 *
 * This file is executed by tests/unit/examples.test.ts against a stub portal.
 */

import { BaseAgent, PortalConnectSkill } from 'webagents';

export const portal = new PortalConnectSkill();

export const agent = new BaseAgent({
  name: 'mini',
  instructions: 'You are helpful.',
  model: 'openai/gpt-4o-mini',
  skills: [portal],
});

export async function main(): Promise<void> {
  await portal.initialize(); // reads the env, opens the socket
  try {
    await new Promise(() => {}); // the bridge lives on the socket, not a port
  } finally {
    await portal.stop();
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
