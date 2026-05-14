/**
 * Template: `logout`.
 *
 * Clear the agent-scoped session by setting an `agt_<agentId>_session`
 * cookie with `Max-Age=0`. The platform Robutler session is NOT
 * affected — it lives on the parent robutler.ai domain and the agent
 * function never sees it.
 *
 * If the agent maintains its own session table in `ctx.kv`, the
 * function should also delete that row.
 */

import type { WebappTemplate } from './types';

export const logout: WebappTemplate = {
  name: 'logout',
  description: 'Clear the agent-scoped session cookie + drop the kv row.',
  when_to_use:
    "You're running an agent-scoped session (e.g. after `oauth_google_login`) and want a clean sign-out.",
  security_notes:
    "- Cookie name MUST be `agt_<ctx.auth.agentId>_<...>` — the dispatcher hard-rejects (502) any Set-Cookie with a different prefix.\n- NEVER set `Domain=` — the dispatcher rejects that too. Cookies stay host-scoped automatically.\n- Always pair with deleting any persisted session row in `ctx.kv`; just clearing the cookie leaves a leak if the cookie is reused.",
  required_permissions: {
    kv: { self: 'rw', visitor: 'rw' },
  },
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: { kv: { self: 'rw', visitor: 'rw' } }
export async function handler(ctx) {
  const cookieName = 'agt_' + ctx.auth.agentId + '_session';
  if (ctx.auth?.userId) {
    try { await ctx.kv.delete({ user_id: ctx.auth.userId, key: 'session' }); } catch {}
  }
  return {
    status: 302,
    headers: {
      location: '/',
      'set-cookie': cookieName + '=; Max-Age=0; Path=/; HttpOnly; Secure; SameSite=Lax',
    },
    body: '',
  };
}
`,
};
