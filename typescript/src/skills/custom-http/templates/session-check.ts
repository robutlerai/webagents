/**
 * Template: `session_check`.
 *
 * Tiny JSON endpoint that returns whether the visitor is signed in to
 * Robutler and, if so, their user id. Useful as a same-origin XHR
 * the agent's frontend JS can call to decide which UI to render
 * without doing a full page reload.
 */

import type { WebappTemplate } from './types';

export const sessionCheck: WebappTemplate = {
  name: 'session_check',
  description: '`{ authenticated, userId }` JSON for client-side UI gating.',
  when_to_use:
    "Frontend JS needs a quick 'am I logged in?' check. Pair with `visitor_session_personalized` for the actual personalised payload.",
  security_notes:
    "- `auth: 'visitor_session'` — same-origin XHR only (no permissive CORS by design).\n- Never include `email` here unless `permissions.visitor_profile` includes `'email'` — and only return what your UI actually needs.",
  required_permissions: {},
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: {}
export async function handler(ctx) {
  return {
    status: 200,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      authenticated: !!ctx.auth?.authenticated,
      userId: ctx.auth?.userId ?? null,
    }),
  };
}
`,
};
