/**
 * Template: `visitor_session_personalized`.
 *
 * Robutler-as-IdP at its simplest. Endpoint declared with
 * `auth: 'visitor_session'`; the dispatcher resolves the platform
 * cookie to a verified user id and (when `permissions.visitor_profile`
 * is declared) loads the requested profile fields.
 *
 * No OAuth, no cookies set by the agent — the platform's own session
 * does the work. The function decides whether to show a "sign in"
 * prompt or the personalised view.
 */

import type { WebappTemplate } from './types';

export const visitorSessionPersonalized: WebappTemplate = {
  name: 'visitor_session_personalized',
  description: 'Use the platform Robutler session to greet the logged-in visitor by name.',
  when_to_use:
    "You want the visitor's name / avatar without rolling your own login. Works only for visitors signed in to robutler.ai (NOT custom-domain agents — see `signin_with_robutler`).",
  security_notes:
    "- `visitor_session` is COOKIE-AUTHENTICATED. The dispatcher refuses permissive CORS, so cross-origin XHR with `credentials: 'include'` will fail by design — that's the protection.\n- Only `name` / `avatar` are surfaced by default. Add `'email'` to `permissions.visitor_profile` ONLY if you genuinely need the email — it's PII.\n- On a custom domain, the platform cookie is NOT present (host-scoped to robutler.ai). Anonymous fallback or `signin_with_robutler` redirect is required.",
  required_permissions: {
    visitor_profile: ['name', 'avatar'] as const,
  },
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: { visitor_profile: ['name', 'avatar'] }
export async function handler(ctx) {
  if (!ctx.auth?.authenticated) {
    return htmlResponse(200, \`<p>Hi! <a href="/login">Sign in with Robutler</a> to personalise this page.</p>\`);
  }
  const name = ctx.auth.profile?.displayName ?? 'friend';
  const avatar = ctx.auth.profile?.avatarUrl;
  return htmlResponse(200, \`
    <h1>Welcome back, \${escapeHtml(name)}!</h1>
    \${avatar ? \`<img src="\${escapeAttr(avatar)}" width="48" height="48" alt="" />\` : ''}
  \`);
}

function htmlResponse(status, body) {
  return { status, headers: { 'content-type': 'text/html; charset=utf-8' }, body };
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function escapeAttr(s) { return escapeHtml(s); }
`,
};
