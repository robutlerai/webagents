/**
 * Template: `agent_to_agent_endpoint`.
 *
 * Receive a call from another agent. Validates the calling agent's
 * portal-token JWT (set `auth: 'portal_token'` on the endpoint
 * declaration so the dispatcher does the verification) and returns
 * a JSON payload.
 *
 * Pair this with `call_other_agent` on the caller side.
 */

import type { WebappTemplate } from './types';

export const agentToAgentEndpoint: WebappTemplate = {
  name: 'agent_to_agent_endpoint',
  description: 'Receive a call from another agent (portal_token-authed).',
  when_to_use:
    "You're publishing a small machine API for other agents on the platform to call (e.g. a tool another agent's LLM might invoke).",
  security_notes:
    "- Use `auth: 'portal_token'` on the endpoint declaration. The dispatcher verifies the calling agent's identity and populates `ctx.auth.agentId`.\n- Whitelist the agents you accept calls from — `ctx.auth.agentId` is verified, but the FACT that another agent CAN call you is not authorisation.\n- Charge for usage explicitly via `ctx.payment.lock(...)` if the work is non-trivial; portal-token calls aren't auto-billed.",
  required_permissions: {},
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: {}
const ALLOWED_CALLERS = new Set([
  // Add agent IDs (UUIDs) you want to accept calls from.
  // Empty set = reject every call.
]);

export async function handler(ctx) {
  const callerAgentId = ctx.auth?.agentId;
  if (!callerAgentId || !ALLOWED_CALLERS.has(callerAgentId)) {
    return { status: 403, headers: { 'content-type': 'application/json' }, body: JSON.stringify({ error: 'caller not in allowlist' }) };
  }
  const body = ctx.request.body ?? {};
  return {
    status: 200,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      ok: true,
      caller: callerAgentId,
      echo: body,
    }),
  };
}
`,
};
