/**
 * Template: `signin_with_robutler`.
 *
 * Redirect-style "Sign in with Robutler" link from a custom-domain
 * agent (or any anonymous page) to the platform login page, then
 * back to the original page once a session exists.
 *
 * The platform login page sets the platform cookie (host-scoped to
 * robutler.ai), so the round-trip ends with a session that
 * `visitor_session_personalized` can use — but only if the agent's
 * page is also served from robutler.ai (e.g. via the `/agents/<id>/`
 * canonical URL or the `/d/<custom-domain>/` proxy that lands on a
 * robutler.ai-origin response).
 */

import type { WebappTemplate } from './types';

export const signinWithRobutler: WebappTemplate = {
  name: 'signin_with_robutler',
  description: '"Sign in with Robutler" link that bounces to the platform login + back.',
  when_to_use:
    "You want a one-click identity flow that piggybacks on Robutler's auth without your agent handling OAuth. Works on robutler.ai-origin pages.",
  security_notes:
    "- The `redirect` param MUST be a same-origin URL — never trust a visitor-supplied `redirect=` query param verbatim or attackers can bounce victims to evil.com after login. Validate it parses as a relative path or as `https://robutler.ai/...`.\n- The platform cookie is host-scoped to `robutler.ai`. After login the visitor MUST land on a robutler.ai URL for the session to be readable.",
  required_permissions: {},
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: {}
export async function handler(ctx) {
  const returnTo = sanitiseReturnTo(ctx.request.query?.return_to ?? '/');
  const url = 'https://robutler.ai/login?redirect=' + encodeURIComponent(returnTo);
  return { status: 302, headers: { location: url }, body: '' };
}

function sanitiseReturnTo(value) {
  try {
    if (!value || typeof value !== 'string') return '/';
    if (value.startsWith('/') && !value.startsWith('//')) return value;
    const u = new URL(value, 'https://robutler.ai');
    if (u.hostname === 'robutler.ai') return u.pathname + u.search;
  } catch { /* fallthrough */ }
  return '/';
}
`,
};
