/**
 * Template: `oauth_google_callback`.
 *
 * Companion to `oauth_google_login`. Validates the `state` nonce,
 * exchanges the auth code for a Google access + refresh token, and
 * stores them under the visitor's KV namespace for later API calls.
 */

import type { WebappTemplate } from './types';

export const oauthGoogleCallback: WebappTemplate = {
  name: 'oauth_google_callback',
  description: 'Google OAuth callback — exchange code, store refresh token in `ctx.kv`.',
  when_to_use:
    'Pair with `oauth_google_login`. Mount at `/oauth/google/callback` to match the redirect URI you registered with Google.',
  security_notes:
    "- Compare `state` against the value stored in `ctx.kv` (constant-time). Mismatch → reject.\n- Refresh tokens are long-lived secrets — store them under the visitor's KV namespace, NEVER in a global key.\n- Set `ttlSeconds` only on the access token; refresh tokens should persist indefinitely.",
  required_permissions: {
    secrets: ['google_client_id', 'google_client_secret'],
    kv: { self: 'rw', visitor: 'rw' },
    fetch: ['https://oauth2.googleapis.com/*'],
  },
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: { secrets: ['google_client_id', 'google_client_secret'], kv: { self: 'rw', visitor: 'rw' }, fetch: ['https://oauth2.googleapis.com/*'] }
export async function handler(ctx) {
  if (!ctx.auth?.userId) return { status: 401, body: 'sign in with Robutler first' };
  const code = ctx.request.query?.code;
  const submittedState = ctx.request.query?.state;
  if (!code || !submittedState) return { status: 400, body: 'missing code/state' };

  const stored = await ctx.kv.get({ user_id: ctx.auth.userId, key: 'oauth_state' });
  if (!stored || stored.state !== submittedState) {
    return { status: 400, body: 'invalid state' };
  }
  await ctx.kv.delete({ user_id: ctx.auth.userId, key: 'oauth_state' });

  const [clientId, clientSecret] = await Promise.all([
    ctx.secrets.get('google_client_id'),
    ctx.secrets.get('google_client_secret'),
  ]);
  const redirectUri = 'https://' + ctx.request.headers.host + '/oauth/google/callback';
  const tokenResp = await ctx.fetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      code, client_id: clientId, client_secret: clientSecret,
      redirect_uri: redirectUri, grant_type: 'authorization_code',
    }).toString(),
  });
  if (!tokenResp.ok) return { status: 502, body: 'google token exchange failed' };
  const tokens = await tokenResp.json();

  await ctx.kv.put({ user_id: ctx.auth.userId, key: 'google_access_token' }, tokens.access_token, { ttlSeconds: tokens.expires_in ?? 3600 });
  if (tokens.refresh_token) {
    await ctx.kv.put({ user_id: ctx.auth.userId, key: 'google_refresh_token' }, tokens.refresh_token);
  }
  return { status: 302, headers: { location: stored.returnTo ?? '/' }, body: '' };
}
`,
};
