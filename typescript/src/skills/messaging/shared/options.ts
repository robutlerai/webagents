/**
 * Common option types for messaging skills.
 *
 * Skills are DB-agnostic: the host (portal, OSS daemon, custom runtime)
 * supplies a {@link TokenResolver} that returns the credential at call
 * time. The skill never imports a database client or assumes a specific
 * persistence layer.
 *
 * For standalone use (e.g. `tsx examples/telegram-echo.ts`) skills fall
 * back to {@link defaultEnvTokenResolver} which reads credentials from
 * `process.env`. That keeps the skill runnable without any binding.
 */

export interface ResolvedToken {
  /** The bearer / bot token / API key. */
  token: string;
  /** Optional refresh token, if applicable. */
  refreshToken?: string;
  /** Provider-side metadata (phoneNumberId, accountSid, etc.). */
  metadata?: Record<string, unknown>;
  /** Optional accountId from the host so the skill can correlate logs. */
  accountId?: string;
  /** When set, the host signals the credential should be re-checked. */
  expiresAt?: Date;
}

export interface TokenResolver {
  /**
   * Resolve a token for the calling agent. The skill calls this at the
   * top of every tool / endpoint that reaches the provider API. Returning
   * `null` means "no credential is bound to this agent" — the skill
   * surfaces a structured error rather than calling the provider.
   */
  getToken: (input: { agentId?: string; provider: string; integrationId?: string }) => Promise<ResolvedToken | null>;
}

export interface TokenWriter {
  /**
   * Persist a token freshly minted by an OAuth callback or refresh. The
   * portal mirror writes to `connected_accounts`; the OSS daemon mirror
   * writes to a local JSON file. Skills call this from `@http` callback
   * handlers and from internal refresh paths only.
   */
  setToken: (input: {
    agentId?: string;
    provider: string;
    integrationId?: string;
    token: string;
    refreshToken?: string;
    metadata?: Record<string, unknown>;
    expiresAt?: Date;
    scopes?: string[];
    providerUserId?: string;
    providerUsername?: string;
  }) => Promise<{ accountId: string }>;
}

/**
 * Structured per-call metadata passed by skills to the host's apiCall
 * wrapper. The host reads this to apply rate caps, kill-switches, customer
 * service window pre-vetoes, idempotency replay, and structured logs.
 *
 * Mirrors the portal's WrapMessagingOptions in shape so a single result
 * envelope can flow across the SDK <-> host boundary without translation.
 */
export interface ApiCallMeta {
  /** Provider id matching the connected_accounts row (e.g. 'whatsapp', 'x'). */
  provider: string;
  /** Free-form call type for metrics / logs (e.g. 'send_text', 'publish_video'). */
  type: string;
  agentId?: string;
  integrationId?: string;
  /** The exact connected_accounts.id used for this call (when known). */
  connectedAccountId?: string;
  /**
   * Pre-veto for the 24h customer-service window (WhatsApp, Messenger,
   * Instagram). The host resolves this to a chat-row lookup and short-
   * circuits with reason='window_expired' when the gate is closed.
   */
  windowCheck?: {
    contactExternalId: string;
    chatId?: string;
  };
  /**
   * Stable key used to dedupe retries on providers that support it (Meta
   * Graph, SendGrid Idempotency-Key; deterministic Twilio helper). The
   * skill computes this from (agentId, draftId, hash(payload)) etc.
   */
  idempotencyKey?: string;
  /**
   * Per-call extractor for the provider-side external message id. Each
   * platform reports the persisted message id under a different field
   * (Telegram → `result.message_id`, Twilio → `sid`, Slack → `ts`,
   * Meta WhatsApp → `messages[0].id`, Meta Messenger / Instagram →
   * `message_id`, Discord → `id`, X → `data.dm_event_id`, SendGrid →
   * the `X-Message-Id` response header, Google Chat → `name`). The skill
   * supplies the extractor so the host can populate
   * `ApiCallResult.providerMessageId` from a single uniform field, which
   * the agent runtime later lifts into `messages.metadata.externalMessageId`.
   * Returning `null` (or omitting the extractor) leaves
   * `providerMessageId` undefined.
   */
  extractProviderMessageId?: (data: unknown) => string | null;
}

/**
 * Discriminated result envelope. The success branch carries the typed
 * payload returned by the wrapped fn; the failure branch carries a
 * uniform reason taxonomy that the agent loop can act on without learning
 * per-platform schemas.
 *
 * Shape mirrors lib/messaging/api-helpers.ts MessagingApiResult exactly so
 * the host's wrapper output flows through unchanged.
 */
export type ApiCallResult<T> =
  | { ok: true; data: T; providerMessageId?: string }
  | {
      ok: false;
      retriable: boolean;
      reason:
        | 'integrations_disabled'
        | 'spend_cap_reached'
        | 'reconnect_required'
        | 'window_expired'
        | 'rate_limited'
        | 'account_suspended'
        | 'invalid_input'
        | 'provider_api_error'
        | 'unknown';
      message: string;
      retryAfterSeconds?: number;
      code?: string | number;
    };

export interface ApiCallWrapper {
  /**
   * Kill-switch / spend-cap / metric wrapping. The skill's provider call
   * goes through this if supplied. The default fallback (used in
   * standalone hosts) wraps the fn into an ApiCallResult envelope and
   * surfaces thrown errors as `provider_api_error`.
   */
  wrapApiCall?: <T>(
    meta: ApiCallMeta,
    fn: () => Promise<T>,
  ) => Promise<ApiCallResult<T>>;
}

/**
 * Optional approval-gate hook used by publish-only skills (LinkedIn,
 * Bluesky, Reddit, Slack channel posts, Discord channel posts, Meta page
 * publishing, Instagram publishing). When the host supplies this callback
 * AND the skill is configured with `requirePostApproval=true`, the skill
 * routes the publish through the host's pending-draft store instead of
 * calling the provider API directly. Returning `null` means "publish
 * immediately" (no approval needed for this specific call).
 */
export interface PostApprovalGate {
  requestApproval?: (input: {
    provider: string;
    platform: string;
    payload: Record<string, unknown>;
    agentId?: string;
    integrationId?: string;
  }) => Promise<{ pending: true; draftId: string } | null>;
}

/**
 * Resolved outbound media reference returned by {@link OutboundMediaResolver}.
 *
 * The skill hands `url` to the messaging platform first (one HTTP hop, no
 * buffering). When the platform reports a URL-fetch failure (`failed to
 * get HTTP URL content`, `wrong remote file identifier`, transport
 * timeout, …) and `fetchBytes` is provided, the skill calls it lazily and
 * retries the same send via multipart/form-data. Hosts that cannot stream
 * bytes (e.g. very large files, remote-only content store) leave
 * `fetchBytes` undefined and accept that URL-fetch failures bubble up.
 */
export type ResolvedOutboundMedia = {
  url: string;
  fetchBytes?: () => Promise<{
    buffer: Uint8Array;
    contentType: string;
    filename?: string;
  }>;
};

/**
 * Host-supplied resolver that turns a Robutler content UUID into a
 * messenger-friendly reference. Returning `null` signals "unknown content
 * id" — the skill surfaces a structured `invalid_input` error instead of
 * relaying garbage to the platform.
 */
export type OutboundMediaResolver = (contentId: string) => Promise<ResolvedOutboundMedia | null>;

export interface MessagingSkillOptions extends ApiCallWrapper, PostApprovalGate {
  /** Logical agent id this skill instance serves (passed through to TokenResolver). */
  agentId?: string;
  /** Bound integration id (set by the portal factory). */
  integrationId?: string;
  /**
   * Capabilities enabled for this agent. Skills self-disable any tool
   * whose capability is not in the set; defaults to "all advertised".
   */
  enabledCapabilities?: string[];
  /** Host-supplied credential resolver (preferred). */
  getToken?: TokenResolver['getToken'];
  /** Host-supplied token writer (used by OAuth callback @http endpoints). */
  setToken?: TokenWriter['setToken'];
  /**
   * When true, publish-style tools route through {@link PostApprovalGate.requestApproval}
   * instead of calling the provider directly. When the gate callback is
   * not supplied, the skill falls back to direct publishing and logs a
   * warning so the host owner notices the misconfiguration.
   */
  requirePostApproval?: boolean;
  /**
   * Public base URL for `@http` endpoints owned by this agent. Used by
   * `webhookUrl()` helpers when generating signup URLs to give the
   * provider. Defaults to PORTAL_URL / BASE_URL env, falling back to
   * `http://localhost:3000` in dev.
   */
  httpEndpointBaseUrl?: string;
  /**
   * Optional resolver that turns a Robutler content UUID into an
   * absolute (signed) URL with an optional lazy bytes fallback. Wired by
   * the portal bridge to `signContentUrl(BASE_URL)` + the local content
   * store; standalone hosts can omit it and the skill will reject any
   * `content_id` argument with `invalid_input`.
   */
  resolveMediaForOutbound?: OutboundMediaResolver;
}

/**
 * Resolve the public base URL for self-referential `@http` endpoints.
 * Order: explicit option → BASE_URL → PORTAL_URL → localhost dev.
 */
export function resolveHttpEndpointBaseUrl(opts: MessagingSkillOptions): string {
  if (opts.httpEndpointBaseUrl) return opts.httpEndpointBaseUrl.replace(/\/+$/, '');
  const fromEnv = process.env.BASE_URL ?? process.env.PORTAL_URL;
  return (fromEnv ?? 'http://localhost:3000').replace(/\/+$/, '');
}
