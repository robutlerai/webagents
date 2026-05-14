/**
 * Template: `oauth_google_login` — Google API ACCESS, not identity.
 *
 * Kick off Google's OAuth dance to obtain a Google API access token
 * (Calendar, Gmail, Drive, etc.). Use this ONLY when you need Google
 * APIs on the visitor's behalf; for plain "who is the visitor?",
 * prefer `visitor_session_personalized` — it's free and friction-less.
 *
 * The state parameter binds the redirect to the visitor + intended
 * return URL. Stored under `ctx.kv` keyed by the visitor for O(1)
 * lookup in the callback.
 */

import type { WebappTemplate } from './types';

export const oauthGoogleLogin: WebappTemplate = {
  name: 'oauth_google_login',
  description: 'Start Google OAuth — for Google API ACCESS, not visitor identity.',
  when_to_use:
    'You need to call Google APIs (Calendar, Gmail, Drive) on the visitor\'s behalf. For "who is logged in?" use `visitor_session_personalized` instead.',
  security_notes:
    "- The `state` nonce MUST be cryptographically random and stored server-side (`ctx.kv`); never trust an unsigned `state` round-trip.\n- Add ONLY the OAuth scopes you actually need — Gmail/Drive scopes show a scary consent screen.\n- The Google client_id and secret live in `ctx.secrets`; declare them under `permissions.secrets`.",
  required_permissions: {
    secrets: ['google_client_id', 'google_client_secret'],
    kv: { self: 'rw', visitor: 'rw' },
    fetch: ['https://oauth2.googleapis.com/*', 'https://www.googleapis.com/*'],
  },
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: { secrets: ['google_client_id', 'google_client_secret'], kv: { self: 'rw', visitor: 'rw' }, fetch: ['https://oauth2.googleapis.com/*', 'https://www.googleapis.com/*'] }
export async function handler(ctx) {
  if (!ctx.auth?.userId) return { status: 401, body: 'sign in with Robutler first' };
  const state = randomToken();
  await ctx.kv.put(
    { user_id: ctx.auth.userId, key: 'oauth_state' },
    { state, returnTo: ctx.request.query?.return_to ?? '/' },
    { ttlSeconds: 600 },
  );
  const clientId = await ctx.secrets.get('google_client_id');
  const redirectUri = 'https://' + ctx.request.headers.host + '/oauth/google/callback';
  const scopes = ['https://www.googleapis.com/auth/calendar.readonly'];
  const url = new URL('https://accounts.google.com/o/oauth2/v2/auth');
  url.searchParams.set('client_id', clientId);
  url.searchParams.set('redirect_uri', redirectUri);
  url.searchParams.set('response_type', 'code');
  url.searchParams.set('scope', scopes.join(' '));
  url.searchParams.set('state', state);
  url.searchParams.set('access_type', 'offline');
  url.searchParams.set('prompt', 'consent');
  return { status: 302, headers: { location: url.toString() }, body: '' };
}

function randomToken() {
  const buf = new Uint8Array(32);
  crypto.getRandomValues(buf);
  return Array.from(buf, (b) => b.toString(16).padStart(2, '0')).join('');
}
`,
};
