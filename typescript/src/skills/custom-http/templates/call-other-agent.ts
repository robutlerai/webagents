/**
 * Template: `call_other_agent`.
 *
 * Caller side of agent-to-agent. Mints a portal token, posts to the
 * target agent's `agent_to_agent_endpoint`, and returns the parsed
 * JSON response. The target agent verifies the portal token and
 * checks its allowlist.
 */

import type { WebappTemplate } from './types';

export const callOtherAgent: WebappTemplate = {
  name: 'call_other_agent',
  description: 'Caller side: invoke another agent\'s `agent_to_agent_endpoint`.',
  when_to_use:
    "Your agent needs to delegate a sub-task to a peer agent that exposes a portal_token-authed endpoint.",
  security_notes:
    '- Use `ctx.portal.signRequest(url)` (or the framework helper your runtime exposes) — never hard-code a token; the issuance is short-lived.\n- Treat the response as untrusted input. Validate the JSON shape before destructuring.\n- Allowlist the agents you call. A typo on the target id can leak data to the wrong agent.',
  required_permissions: {
    fetch: ['https://robutler.ai/agents/*', 'https://robutler.ai/api/agents/*'],
  },
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: { fetch: ['https://robutler.ai/agents/*', 'https://robutler.ai/api/agents/*'] }
const TARGET_AGENT = '<paste the target agent UUID or username here>';
const TARGET_PATH  = '/api'; // path the target agent registered

export async function handler(ctx) {
  const url = 'https://robutler.ai/agents/' + TARGET_AGENT + TARGET_PATH;
  const signed = await ctx.portal.signRequest(url, { method: 'POST' });
  const resp = await ctx.fetch(signed.url, {
    method: 'POST',
    headers: { ...signed.headers, 'content-type': 'application/json' },
    body: JSON.stringify({ from: ctx.auth.agentId, payload: ctx.request.body ?? null }),
  });
  if (!resp.ok) {
    return { status: 502, body: 'upstream agent returned ' + resp.status };
  }
  const data = await resp.json();
  return { status: 200, headers: { 'content-type': 'application/json' }, body: JSON.stringify(data) };
}
`,
};
