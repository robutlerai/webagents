/**
 * Template: `csrf_protected_form`.
 *
 * Same-origin POST form with a per-session CSRF token stored in
 * `ctx.kv` and rendered as a hidden input. The token is rotated on
 * every successful submit.
 *
 * Even with `visitor_session` (cookie auth), CSRF is a real risk
 * because the platform cookie is sent on cross-origin form POSTs by
 * default (SameSite=Lax). The token check closes that gap.
 */

import type { WebappTemplate } from './types';

export const csrfProtectedForm: WebappTemplate = {
  name: 'csrf_protected_form',
  description: 'POST form with rotating CSRF token in `ctx.kv`.',
  when_to_use:
    'You need a state-changing form (settings update, comment submit) and want defence-in-depth on top of `visitor_session`.',
  security_notes:
    "- The CSRF token is keyed under the visitor's `user_id` in `ctx.kv` — never store it in a global key, or one visitor can submit on behalf of another.\n- Compare with constant-time string equality (the helper here uses a length-and-loop pattern). DON'T use `===` on attacker-controlled tokens in production.\n- Rotate after every successful POST so a leaked token can't be replayed.",
  required_permissions: {
    kv: { self: 'rw', visitor: 'rw' },
  },
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: { kv: { self: 'rw', visitor: 'rw' } }
export async function handler(ctx) {
  if (!ctx.auth?.userId) return { status: 401, body: 'sign in required' };
  const userId = ctx.auth.userId;

  if (ctx.request.method === 'GET') {
    const token = randomToken();
    await ctx.kv.put({ user_id: userId, key: 'csrf' }, token);
    return htmlResponse(200, formHtml(token));
  }

  if (ctx.request.method === 'POST') {
    const submitted = ctx.request.body?.csrf;
    const expected = await ctx.kv.get({ user_id: userId, key: 'csrf' });
    if (!constantTimeEq(submitted, expected)) {
      return { status: 403, body: 'csrf check failed' };
    }
    await ctx.kv.put({ user_id: userId, key: 'csrf' }, randomToken());
    return htmlResponse(200, '<p>Saved!</p>');
  }
  return { status: 405, body: 'method not allowed' };
}

function htmlResponse(status, body) {
  return { status, headers: { 'content-type': 'text/html; charset=utf-8' }, body };
}

function formHtml(token) {
  return \`<form method="post"><input type="hidden" name="csrf" value="\${token}"><input name="message"><button>Save</button></form>\`;
}

function randomToken() {
  const buf = new Uint8Array(24);
  crypto.getRandomValues(buf);
  return Array.from(buf, (b) => b.toString(16).padStart(2, '0')).join('');
}

function constantTimeEq(a, b) {
  if (typeof a !== 'string' || typeof b !== 'string' || a.length !== b.length) return false;
  let acc = 0;
  for (let i = 0; i < a.length; i++) acc |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return acc === 0;
}
`,
};
