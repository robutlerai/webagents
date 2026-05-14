/**
 * Template: `kv_visitor_state`.
 *
 * Read + write per-visitor state through the unified `ctx.kv` API.
 * Demonstrates the `{ user_id, key, scope }` shape that gates writes
 * to `permissions.kv.visitor`.
 *
 * Storage namespace is automatically derived: visitor + function-scope
 * → `u:<visitorId>:fn:<functionName>`. Other agents and other
 * visitors cannot read these rows even if they share the agent.
 */

import type { WebappTemplate } from './types';

export const kvVisitorState: WebappTemplate = {
  name: 'kv_visitor_state',
  description: 'Per-visitor key/value state (`ctx.kv` with `user_id`).',
  when_to_use:
    'You want visitor-specific persistence (preferences, draft state, last-seen timestamp) WITHOUT building a database table.',
  security_notes:
    "- `permissions.kv` MUST be the OBJECT form `{ self: 'rw', visitor: 'rw' }` — the legacy bare-string form (`kv: 'rw'`) only grants self access.\n- `ctx.kv.put({ user_id, ... })` MUST pass `ctx.auth.userId` for the visitor — passing any other id (including a guess) returns PERMISSION_DENIED.\n- For agent-wide state shared across visitors, use `scope: 'agent'` and add `agent_scope: true` to permissions.kv.",
  required_permissions: {
    kv: { self: 'rw', visitor: 'rw' },
  },
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: { kv: { self: 'rw', visitor: 'rw' } }
export async function handler(ctx) {
  if (!ctx.auth?.userId) {
    return jsonResponse(401, { error: 'sign in required' });
  }
  const userId = ctx.auth.userId;

  if (ctx.request.method === 'GET') {
    const prefs = await ctx.kv.get({ user_id: userId, key: 'prefs' });
    return jsonResponse(200, prefs ?? {});
  }
  if (ctx.request.method === 'POST' || ctx.request.method === 'PUT') {
    const body = ctx.request.body ?? {};
    await ctx.kv.put({ user_id: userId, key: 'prefs' }, body);
    return jsonResponse(200, { ok: true });
  }
  return jsonResponse(405, { error: 'method not allowed' });
}

function jsonResponse(status, payload) {
  return { status, headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload) };
}
`,
};
