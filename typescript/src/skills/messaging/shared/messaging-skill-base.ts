/**
 * Base class for all messaging skills.
 *
 * Centralises:
 *   - capability gating (`enabledCapabilities`)
 *   - token resolution with env fallback
 *   - apiCall wrapping (kill-switch, spend cap, metrics)
 *   - bridge-context lookup
 */
import { Skill } from '../../../core/skill';
import type { Context } from '../../../core/types';
import { extractContentId } from './content-id';
import { defaultEnvTokenResolver } from './env-resolver';
import { noopTokenWriter } from './env-writer';
import type {
  ApiCallMeta,
  ApiCallResult,
  ApiCallWrapper,
  MessagingSkillOptions,
  OutboundMediaResolver,
  ResolvedOutboundMedia,
  ResolvedToken,
  TokenResolver,
  TokenWriter,
} from './options';
import { resolveHttpEndpointBaseUrl } from './options';
import { isUrlFetchFailure } from './url-failure';

export abstract class MessagingSkill extends Skill {
  abstract readonly provider: string;

  protected readonly tokenResolver: TokenResolver;
  protected readonly tokenWriter: TokenWriter;
  protected readonly wrapApiCall: NonNullable<ApiCallWrapper['wrapApiCall']>;
  protected readonly enabledCapabilities: Set<string>;
  protected readonly httpBaseUrl: string;
  protected readonly agentId?: string;
  protected readonly integrationId?: string;
  protected readonly requirePostApproval: boolean;
  protected readonly approvalGate?: NonNullable<MessagingSkillOptions['requestApproval']>;
  protected readonly outboundMediaResolver?: OutboundMediaResolver;

  constructor(skillName: string, opts: MessagingSkillOptions = {}) {
    super({ name: skillName });
    this.agentId = opts.agentId;
    this.integrationId = opts.integrationId;
    this.tokenResolver = opts.getToken ? { getToken: opts.getToken } : defaultEnvTokenResolver();
    this.tokenWriter = opts.setToken ? { setToken: opts.setToken } : noopTokenWriter();
    const hostWrap: NonNullable<ApiCallWrapper['wrapApiCall']> =
      opts.wrapApiCall ??
      (async <T>(_meta: ApiCallMeta, fn: () => Promise<T>): Promise<ApiCallResult<T>> => {
        try {
          const data = await fn();
          let providerMessageId: string | undefined;
          try {
            const extracted = defaultExtractProviderMessageId(data);
            if (typeof extracted === 'string' && extracted.length > 0) {
              providerMessageId = extracted;
            }
          } catch {
            // Ignore extractor failures - the send already succeeded.
          }
          return providerMessageId !== undefined
            ? { ok: true, data, providerMessageId }
            : { ok: true, data };
        } catch (err) {
          return {
            ok: false,
            retriable: false,
            reason: 'provider_api_error',
            message: err instanceof Error ? err.message : String(err),
          };
        }
      });
    /**
     * Inject a default `extractProviderMessageId` into every meta so the
     * host wrapper (`wrapMessagingApiCall`) gets a consistent way to surface
     * `providerMessageId` on `ApiCallResult` even for tools that only ship
     * the wrapped `{ externalMessageId }` shape. Per-call extractors win
     * (a skill that wants to walk a raw provider response can pass its
     * own); the default just pulls `data.externalMessageId` which every
     * messaging skill already populates on success.
     */
    this.wrapApiCall = async <T>(meta: ApiCallMeta, fn: () => Promise<T>): Promise<ApiCallResult<T>> => {
      const enrichedMeta: ApiCallMeta = meta.extractProviderMessageId
        ? meta
        : { ...meta, extractProviderMessageId: defaultExtractProviderMessageId };
      return hostWrap(enrichedMeta, fn);
    };
    this.enabledCapabilities = new Set(opts.enabledCapabilities ?? []);
    this.httpBaseUrl = resolveHttpEndpointBaseUrl(opts);
    this.requirePostApproval = opts.requirePostApproval === true;
    this.approvalGate = opts.requestApproval;
    this.outboundMediaResolver = opts.resolveMediaForOutbound;
  }

  /**
   * Helper for publish-style tools. When approval is required AND a gate
   * is configured, defer to the host. Otherwise return null and the tool
   * proceeds with the direct provider call.
   */
  protected async maybeRequestApproval(
    platform: string,
    payload: Record<string, unknown>,
  ): Promise<{ pending: true; draftId: string } | null> {
    if (!this.requirePostApproval) return null;
    if (!this.approvalGate) {
      console.warn(
        `[${this.provider}] requirePostApproval=true but no requestApproval callback configured — falling back to direct publish`,
      );
      return null;
    }
    return this.approvalGate({
      provider: this.provider,
      platform,
      payload,
      agentId: this.agentId,
      integrationId: this.integrationId,
    });
  }

  /**
   * Returns true when the skill should expose the named capability for
   * this agent. When `enabledCapabilities` was not configured, every
   * advertised capability is considered enabled (legacy behaviour).
   */
  protected capabilityEnabled(name: string): boolean {
    if (this.enabledCapabilities.size === 0) return true;
    return this.enabledCapabilities.has(name);
  }

  protected async resolveToken(): Promise<ResolvedToken> {
    const t = await this.tokenResolver.getToken({
      agentId: this.agentId,
      provider: this.provider,
      integrationId: this.integrationId,
    });
    if (!t) {
      throw new Error(`${this.provider}_credential_not_found`);
    }
    return t;
  }

  /** Helper for skills that resolve their primary recipient from the bridge. */
  protected bridgeRecipient(ctx: Context | undefined): string | undefined {
    const bridge = (ctx?.metadata as Record<string, unknown> | undefined)?.bridge as
      | { source?: string; contactExternalId?: string }
      | undefined;
    if (!bridge || bridge.source !== this.provider) return undefined;
    return bridge.contactExternalId;
  }

  /**
   * Pre-veto helpers - skills use these for guard-clause failures
   * (capability gate, missing recipient, missing metadata) so every tool
   * returns the same `ApiCallResult<T>` envelope regardless of where the
   * failure originated.
   */
  protected invalidInput<T>(message: string, code?: string): ApiCallResult<T> {
    return { ok: false, retriable: false, reason: 'invalid_input', message, code };
  }

  protected capabilityDisabled<T>(capability: string): ApiCallResult<T> {
    return {
      ok: false,
      retriable: false,
      reason: 'invalid_input',
      message: `Capability '${capability}' is not enabled for this agent`,
      code: 'capability_disabled',
    };
  }

  protected reconnectRequired<T>(message?: string): ApiCallResult<T> {
    return {
      ok: false,
      retriable: false,
      reason: 'reconnect_required',
      message: message ?? `${this.provider} integration needs to be reconnected.`,
    };
  }

  // ---------------------------------------------------------------------------
  // Outbound media — content_id resolution + URL-first / bytes-fallback send
  // ---------------------------------------------------------------------------

  /**
   * Translate a (`content_id` | `url`) input into a {@link ResolvedOutboundMedia}.
   *
   * - Absolute external URLs (`http://` / `https://`) pass through with
   *   no resolver call and no `fetchBytes` (the host has no way to
   *   download them, and the platform can fetch them directly).
   * - A bare UUID, or a UUID embedded in a relative `/api/content/<uuid>`
   *   path, is normalised and handed to the host's
   *   {@link OutboundMediaResolver}.
   * - When neither input is provided, returns an `invalid_input` envelope.
   * - When the resolver returns `null` (unknown / unauthorised content),
   *   returns an `invalid_input` envelope so we never relay garbage to
   *   the platform.
   *
   * Per-provider tools call this from their `send_image` / `send_document`
   * handler before invoking {@link sendMediaWithFallback} (or, for
   * URL-only providers, before just passing `media.url` to the API).
   */
  protected async resolveOutboundMedia(input: {
    contentId?: string | null;
    url?: string | null;
  }): Promise<{ media: ResolvedOutboundMedia } | { error: ApiCallResult<never> }> {
    const url = input.url?.trim();

    if (url && /^https?:\/\//i.test(url)) {
      return { media: { url } };
    }

    const contentId = extractContentId(input);
    if (!contentId) {
      return {
        error: this.invalidInput(
          'Provide either `content_id` (Robutler UUID) or an absolute https URL. Relative `/api/content/...` URLs without a recognisable UUID are not supported.',
          'invalid_media_reference',
        ),
      };
    }

    if (!this.outboundMediaResolver) {
      return {
        error: this.invalidInput(
          'This host is not configured to resolve Robutler `content_id` references. Pass an absolute https URL instead.',
          'media_resolver_unavailable',
        ),
      };
    }

    const resolved = await this.outboundMediaResolver(contentId);
    if (!resolved) {
      return {
        error: this.invalidInput(
          `Content ${contentId} was not found, has been deleted, or is not accessible to this agent.`,
          'content_not_found',
        ),
      };
    }
    return { media: resolved };
  }

  /**
   * Send a media-bearing message using the URL-first → bytes-fallback
   * strategy. Used by providers whose API supports both inline URLs
   * (Telegram `sendPhoto`, WhatsApp `image.link`) and a multipart upload
   * fallback (Telegram multipart, WhatsApp `/media`).
   *
   * Behaviour:
   *   1. call `sendByUrl(media.url)`,
   *   2. on failure where {@link isUrlFetchFailure} matches and both
   *      `media.fetchBytes` and `sendByBytes` are present, fetch the
   *      bytes once and retry via `sendByBytes`,
   *   3. otherwise return the original failure unchanged.
   *
   * Logs the fallback once per call so we can see how often it fires in
   * production.
   *
   * Pure URL-only providers (Messenger, Instagram DM, Twilio MMS) skip
   * this helper and just pass `media.url` directly to their API. Pure
   * bytes-only providers (Discord, Slack) skip it too and always
   * upload bytes.
   */
  protected async sendMediaWithFallback<T>(input: {
    callType: string;
    media: ResolvedOutboundMedia;
    sendByUrl: (url: string) => Promise<ApiCallResult<T>>;
    sendByBytes?: (bytes: {
      buffer: Uint8Array;
      contentType: string;
      filename?: string;
    }) => Promise<ApiCallResult<T>>;
    extraUrlFailurePatterns?: RegExp[];
  }): Promise<ApiCallResult<T>> {
    const { callType, media, sendByUrl, sendByBytes, extraUrlFailurePatterns } = input;

    const urlAttempt = await sendByUrl(media.url);
    if (urlAttempt.ok) return urlAttempt;

    const failedOnUrlFetch = isUrlFetchFailure(urlAttempt.message, extraUrlFailurePatterns);
    if (!failedOnUrlFetch || !media.fetchBytes || !sendByBytes) {
      return urlAttempt;
    }

    console.log(
      `[${this.provider}] ${callType} URL fetch failed (${truncate(urlAttempt.message, 120)}) — retrying via multipart upload`,
    );
    let bytes: Awaited<ReturnType<NonNullable<ResolvedOutboundMedia['fetchBytes']>>>;
    try {
      bytes = await media.fetchBytes();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn(`[${this.provider}] ${callType} bytes fallback failed to load content: ${msg}`);
      return urlAttempt;
    }

    return sendByBytes(bytes);
  }
}

function truncate(s: string | undefined, max: number): string {
  if (!s) return '';
  return s.length <= max ? s : `${s.slice(0, max - 1)}…`;
}

/**
 * Default `extractProviderMessageId` shared by every messaging skill.
 *
 * Each skill's send tool wraps its provider response into the canonical
 * `{ externalMessageId, ... }` shape (see telegram, slack, whatsapp,
 * instagram, messenger, twilio, x, sendgrid, google-chat skills). The
 * extractor surfaces that single field onto `ApiCallResult.providerMessageId`
 * so hosts can persist a uniform external id (lifted into
 * `messages.metadata.externalMessageId` by the agent runtime) without
 * learning a per-platform schema.
 *
 * Per-provider raw extractors documented in the plan (Phase 10):
 *   - telegram:               data.message_id
 *   - twilio:                 data.sid
 *   - slack:                  data.ts
 *   - whatsapp:               data.messages[0].id
 *   - messenger / instagram:  data.message_id
 *   - discord:                data.id
 *   - x:                      data.dm_event.id  /  data.data.id
 *   - sendgrid:               response header `X-Message-Id`
 *   - google-chat:            data.name
 *
 * The extractor below covers all of those shapes in case a skill ever
 * returns the raw provider payload directly without wrapping it first.
 */
export function defaultExtractProviderMessageId(data: unknown): string | null {
  if (!data || typeof data !== 'object') return null;
  const obj = data as Record<string, unknown>;
  const direct = pickString(obj.externalMessageId);
  if (direct) return direct;

  const messageId = pickString(obj.message_id);
  if (messageId) return messageId;

  const messages = obj.messages;
  if (Array.isArray(messages) && messages.length > 0) {
    const first = messages[0];
    if (first && typeof first === 'object') {
      const id = pickString((first as Record<string, unknown>).id);
      if (id) return id;
    }
  }

  const sid = pickString(obj.sid);
  if (sid) return sid;

  const ts = pickString(obj.ts);
  if (ts) return ts;

  const id = pickString(obj.id);
  if (id) return id;

  const name = pickString(obj.name);
  if (name) return name;

  const dmEvent = obj.dm_event;
  if (dmEvent && typeof dmEvent === 'object') {
    const dmId = pickString((dmEvent as Record<string, unknown>).id);
    if (dmId) return dmId;
  }

  const innerData = obj.data;
  if (innerData && typeof innerData === 'object') {
    const dataObj = innerData as Record<string, unknown>;
    const dmEventId = pickString(dataObj.dm_event_id);
    if (dmEventId) return dmEventId;
    const inner = pickString(dataObj.id);
    if (inner) return inner;
  }

  const result = obj.result;
  if (result && typeof result === 'object') {
    const r = result as Record<string, unknown>;
    const inner = pickString(r.message_id) ?? pickString(r.id);
    if (inner) return inner;
  }

  return null;
}

function pickString(v: unknown): string | null {
  if (typeof v === 'string' && v.length > 0) return v;
  if (typeof v === 'number') {
    const s = String(v);
    return s.length > 0 ? s : null;
  }
  return null;
}
