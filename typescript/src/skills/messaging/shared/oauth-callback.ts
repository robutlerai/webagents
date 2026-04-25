/**
 * Provider-agnostic OAuth code → token exchange used by per-skill
 * `@http('oauth/callback', GET)` handlers.
 *
 * The portal does NOT use this path — its `/api/auth/connect/[provider]/
 * callback` route owns the canonical exchange so it can stamp
 * `connected_accounts` rows with portal-specific bookkeeping (display
 * names, avatars, agent-binding propagation, etc.).
 *
 * Standalone hosts (a one-off Node daemon, an OSS deployment without
 * the portal monolith) DON'T have that route, so each messaging skill
 * exposes a fallback `oauth/callback` endpoint that:
 *
 *   1. Reads `code` and `state` from the redirect URL.
 *   2. POSTs to the provider's token endpoint with client_id /
 *      client_secret pulled from env (`<PROVIDER>_CLIENT_ID`, etc.).
 *   3. Calls the host-supplied `tokenWriter.setToken(...)` with the
 *      resulting access token (and optional refresh token / metadata).
 *   4. Returns a tiny success page so the user knows the link worked.
 *
 * Skills with provider-specific quirks (e.g. WhatsApp's BSP-vs-cloud
 * branching, Reddit's required `User-Agent`, Bluesky's app-password
 * model that has no OAuth at all) override the relevant helpers.
 */
import type { TokenWriter } from './options';

export interface OAuthExchangeParams {
  /** Lowercase provider id, used for env-var lookup and TokenWriter dispatch. */
  provider: string;
  /** Authorization code received in the redirect. */
  code: string;
  /** Redirect URI registered with the provider — must match exactly. */
  redirectUri: string;
  /** Provider's token endpoint (POSTed as application/x-www-form-urlencoded). */
  tokenUrl: string;
  /** Optional override for client id (defaults to env). */
  clientId?: string;
  /** Optional override for client secret (defaults to env). */
  clientSecret?: string;
  /** Extra params sent with the exchange (e.g. grant_type overrides). */
  extraParams?: Record<string, string>;
}

export interface OAuthExchangeResult {
  accessToken: string;
  refreshToken?: string;
  expiresAt?: Date;
  scope?: string[];
  tokenType?: string;
  /** Raw provider payload, in case the skill needs provider-specific fields. */
  raw: Record<string, unknown>;
}

/**
 * Perform the token exchange. Throws on non-2xx so the @http handler
 * can render a meaningful error page.
 */
export async function exchangeOAuthCode(
  params: OAuthExchangeParams,
): Promise<OAuthExchangeResult> {
  const { provider, code, redirectUri, tokenUrl, extraParams } = params;
  const envPrefix = provider.toUpperCase();
  const clientId =
    params.clientId ?? process.env[`${envPrefix}_CLIENT_ID`] ?? '';
  const clientSecret =
    params.clientSecret ?? process.env[`${envPrefix}_CLIENT_SECRET`] ?? '';
  if (!clientId || !clientSecret) {
    throw new Error(
      `${envPrefix}_CLIENT_ID / ${envPrefix}_CLIENT_SECRET not configured`,
    );
  }
  const body = new URLSearchParams({
    grant_type: 'authorization_code',
    code,
    redirect_uri: redirectUri,
    client_id: clientId,
    client_secret: clientSecret,
    ...(extraParams ?? {}),
  });
  const response = await fetch(tokenUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      Accept: 'application/json',
    },
    body: body.toString(),
  });
  const text = await response.text();
  let json: Record<string, unknown> = {};
  try {
    json = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`token endpoint returned non-JSON: ${text.slice(0, 200)}`);
  }
  if (!response.ok) {
    throw new Error(
      `token exchange failed (${response.status}): ${
        (json.error_description as string | undefined) ??
        (json.error as string | undefined) ??
        text.slice(0, 200)
      }`,
    );
  }
  const expiresIn = json.expires_in as number | undefined;
  const expiresAt = expiresIn ? new Date(Date.now() + expiresIn * 1000) : undefined;
  return {
    accessToken: String(json.access_token ?? ''),
    refreshToken: json.refresh_token ? String(json.refresh_token) : undefined,
    expiresAt,
    scope:
      typeof json.scope === 'string'
        ? (json.scope as string).split(/[,\s]+/).filter(Boolean)
        : undefined,
    tokenType: typeof json.token_type === 'string' ? (json.token_type as string) : undefined,
    raw: json,
  };
}

/**
 * Convenience: parse `code` + `state` from a redirect Request, run the
 * exchange, persist via the supplied TokenWriter, and return a tiny
 * HTML success / error page so the user has positive feedback.
 *
 * Skills layer their own metadata extraction on top by passing
 * `transformResult(json)` if they need to read e.g. `data.user.id` for
 * the Twitter response shape.
 */
export async function handleOAuthCallback(args: {
  request: Request;
  provider: string;
  redirectUri: string;
  tokenUrl: string;
  tokenWriter?: TokenWriter;
  agentId?: string;
  integrationId?: string;
  extraExchangeParams?: Record<string, string>;
  transformResult?: (raw: Record<string, unknown>) => {
    metadata?: Record<string, unknown>;
    providerUserId?: string;
    providerUsername?: string;
  };
}): Promise<Response> {
  const url = new URL(args.request.url);
  const code = url.searchParams.get('code');
  const error = url.searchParams.get('error');
  if (error) {
    return new Response(htmlPage(`OAuth error: ${error}`, false), {
      status: 400,
      headers: { 'Content-Type': 'text/html; charset=utf-8' },
    });
  }
  if (!code) {
    return new Response(htmlPage('Missing authorization code.', false), {
      status: 400,
      headers: { 'Content-Type': 'text/html; charset=utf-8' },
    });
  }
  try {
    const exchanged = await exchangeOAuthCode({
      provider: args.provider,
      code,
      redirectUri: args.redirectUri,
      tokenUrl: args.tokenUrl,
      extraParams: args.extraExchangeParams,
    });
    const extra = args.transformResult?.(exchanged.raw) ?? {};
    if (args.tokenWriter) {
      await args.tokenWriter.setToken({
        agentId: args.agentId,
        integrationId: args.integrationId,
        provider: args.provider,
        token: exchanged.accessToken,
        refreshToken: exchanged.refreshToken,
        metadata: extra.metadata,
        expiresAt: exchanged.expiresAt,
        scopes: exchanged.scope,
        providerUserId: extra.providerUserId,
        providerUsername: extra.providerUsername,
      });
    }
    return new Response(
      htmlPage(`Connected ${args.provider}. You can close this tab.`, true),
      { status: 200, headers: { 'Content-Type': 'text/html; charset=utf-8' } },
    );
  } catch (err) {
    return new Response(
      htmlPage(`Connection failed: ${(err as Error).message}`, false),
      { status: 500, headers: { 'Content-Type': 'text/html; charset=utf-8' } },
    );
  }
}

function htmlPage(message: string, ok: boolean): string {
  const color = ok ? '#16a34a' : '#dc2626';
  const safeMessage = message.replace(/[<>&]/g, (c) =>
    ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c] as string),
  );
  return `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>OAuth callback</title></head>
<body style="font-family:system-ui;max-width:480px;margin:96px auto;text-align:center;color:#111">
  <div style="font-size:48px;color:${color};margin-bottom:16px">${ok ? '\u2713' : '\u2717'}</div>
  <p style="font-size:16px;line-height:1.4">${safeMessage}</p>
</body></html>`;
}
