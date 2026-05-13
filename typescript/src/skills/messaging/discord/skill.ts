/**
 * DiscordSkill — Discord Bot API + interactions endpoint.
 *
 * Token resolution: ResolvedToken.token = bot token (NOT a user token).
 * Metadata carries `applicationId` and `publicKey` (used to verify
 * Ed25519-signed inbound interactions).
 *
 * Reference: https://discord.com/developers/docs/intro
 */
import { http, prompt, tool } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  handleOAuthCallback,
  MessagingSkill,
  verifyDiscordSignature,
  type MessagingSkillOptions,
} from '../shared';

const PROVIDER = 'discord';
const SEND_TOOL = 'discord_send_dm';

interface DiscordMetadata {
  applicationId?: string;
  publicKey?: string;
  credentialType?: string;
}

export class DiscordSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  /**
   * Tracks which slash-command interaction tokens have already had their
   * deferred "@original" placeholder filled. The first agent reply for a
   * given interaction MUST be a `PATCH /webhooks/{app_id}/{token}/messages/@original`
   * to replace Discord's "Robutler is thinking…" loading bubble — every
   * subsequent reply is a `POST /webhooks/{app_id}/{token}` follow-up.
   * Without the PATCH the loading bubble never resolves and Discord
   * eventually renders it as "The application did not respond" once the
   * 15-min token TTL expires, even when follow-up POSTs have landed.
   *
   * Keyed by interaction token; values store the expiry timestamp so we
   * can prune stale entries lazily (tokens are valid for 15 min).
   */
  private slashOriginalFilledAt: Map<string, number> = new Map();

  constructor(opts: MessagingSkillOptions = {}) {
    super('discord', opts);
  }

  private markSlashOriginalFilled(token: string): void {
    this.slashOriginalFilledAt.set(token, Date.now() + 16 * 60 * 1000);
    if (this.slashOriginalFilledAt.size > 2048) {
      const now = Date.now();
      for (const [k, exp] of this.slashOriginalFilledAt) {
        if (exp < now) this.slashOriginalFilledAt.delete(k);
      }
    }
  }

  private slashOriginalIsFilled(token: string): boolean {
    const exp = this.slashOriginalFilledAt.get(token);
    if (!exp) return false;
    if (exp < Date.now()) {
      this.slashOriginalFilledAt.delete(token);
      return false;
    }
    return true;
  }

  @tool({
    name: SEND_TOOL,
    description:
      'Send a DM to a Discord user (opens a DM channel first via POST /users/@me/channels per ' +
      'Discord API v10). THIS IS THE ONLY WAY TO REPLY TO A BRIDGED DISCORD DM CONTACT — a plain ' +
      'assistant message stays in the portal and never reaches the user.',
    parameters: {
      type: 'object',
      properties: {
        user_id: { type: 'string' },
        content: { type: 'string' },
      },
      required: ['content'],
    },
    // DM tool: only surfaces when the current chat is a Discord DM
    // bridge OR a Discord slash-command bridge. In a guild @-mention
    // bridge the user is in a public channel, so DM-ing them out of
    // band is almost never what they want — `discord_send_in_channel`
    // is the right reply path there. For slash commands we want this
    // tool surfaced even in guild context (the agent's reply lands in
    // the original interaction webhook, not the channel).
    requiresBridge: [
      { source: 'discord', kind: 'dm' },
      { source: 'discord', kind: 'slash_command' },
    ],
  })
  async sendDm(args: { user_id?: string; content: string }, ctx?: Context) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    // Slash-command path: replace the deferred "Robutler is thinking..."
    // bubble (or post a follow-up) via the interaction webhook. The
    // outbound DM tool is the canonical reply path even from a guild
    // channel slash command — Discord routes the response back to the
    // original interaction context for us.
    const slash = this.activeSlashInteraction(ctx);
    if (slash) {
      return this.discordCall('send_dm', async () => {
        return this.sendInteractionFollowup(slash, { content: args.content });
      });
    }
    const userId = args.user_id ?? this.bridgeRecipient(ctx);
    if (!userId) return this.invalidInput('user_id required');
    return this.discordCall('send_dm', async (token) => {
      const open = await this.discordFetchRaw(token, '/users/@me/channels', {
        method: 'POST',
        body: JSON.stringify({ recipient_id: userId }),
      });
      const channelId = (open as { id?: string }).id;
      if (!channelId) throw new Error('open_dm_failed');
      return this.discordFetchRaw(token, `/channels/${channelId}/messages`, {
        method: 'POST',
        body: JSON.stringify({ content: args.content }),
      });
    });
  }

  @tool({
    name: 'discord_send_dm_photo',
    description:
      'Send an image attachment via DM to a Discord user. Pass either ' +
      '`content_id` (Robutler UUID) or `image_url` (any reachable URL). ' +
      'Discord requires multipart upload — the skill fetches external URLs ' +
      'server-side and uploads bytes. Optional `content` text is included ' +
      'in the same message.',
    parameters: {
      type: 'object',
      properties: {
        user_id: { type: 'string' },
        content_id: { type: 'string' },
        image_url: { type: 'string' },
        content: { type: 'string' },
        filename: { type: 'string' },
      },
      required: [],
    },
    requiresBridge: [
      { source: 'discord', kind: 'dm' },
      { source: 'discord', kind: 'slash_command' },
    ],
  })
  async sendDmPhoto(
    args: {
      user_id?: string;
      content_id?: string;
      image_url?: string;
      content?: string;
      filename?: string;
    },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    return this.sendDmAttachment({ ...args, url: args.image_url, kind: 'image' }, ctx);
  }

  @tool({
    name: 'discord_send_dm_document',
    description:
      'Send a file (document) attachment via DM to a Discord user. Same ' +
      'multipart-upload behaviour as `discord_send_dm_photo`.',
    parameters: {
      type: 'object',
      properties: {
        user_id: { type: 'string' },
        content_id: { type: 'string' },
        document_url: { type: 'string' },
        content: { type: 'string' },
        filename: { type: 'string' },
      },
      required: [],
    },
    requiresBridge: [
      { source: 'discord', kind: 'dm' },
      { source: 'discord', kind: 'slash_command' },
    ],
  })
  async sendDmDocument(
    args: {
      user_id?: string;
      content_id?: string;
      document_url?: string;
      content?: string;
      filename?: string;
    },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    return this.sendDmAttachment({ ...args, url: args.document_url, kind: 'document' }, ctx);
  }

  @tool({
    name: 'discord_send_in_channel',
    description:
      'Reply in a Discord channel the bot has access to. When invoked from a ' +
      'guild @-mention bridge, `channel_id` defaults to the channel the user ' +
      "pinged the bot in — the LLM can call this with just `content` and the " +
      'reply lands in the right place, with the original author auto-pinged.',
    parameters: {
      type: 'object',
      properties: {
        channel_id: { type: 'string' },
        content: { type: 'string' },
        embeds: {
          type: 'array',
          items: { type: 'object', additionalProperties: true },
        },
      },
      required: ['content'],
    },
    // Channel posts are visible to everyone in the channel — gate
    // behind the loaded NotificationSkill so a non-owner cannot
    // ask the bot to broadcast without owner sign-off. The
    // PortalNotificationSkill auto-approves when the requester is
    // the agent owner (already in-loop).
    //
    // Auto-bypass when this run is a reply to a guild @-mention to the
    // *same* channel: the user explicitly invoked the bot in public,
    // which is itself the consent signal for a public reply. Outside
    // that path (unsolicited broadcasts, replies to a different
    // channel) the gate stands.
    requiresConfirmation: (args, ctx) => {
      const a = args as { channel_id?: string };
      const bridge = (ctx?.metadata as Record<string, unknown> | undefined)?.bridge as
        | { source?: string; kind?: 'dm' | 'mention'; channelId?: string }
        | undefined;
      if (
        bridge?.source === PROVIDER &&
        bridge.kind === 'mention' &&
        bridge.channelId &&
        (!a.channel_id || a.channel_id === bridge.channelId)
      ) {
        return false;
      }
      return true;
    },
  })
  async sendInChannel(
    args: { channel_id?: string; content: string; embeds?: unknown[] },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages') && !this.capabilityEnabled('publish_posts')) {
      return this.capabilityDisabled('send_messages');
    }
    // Slash-command path: same as `sendDm` — replies go through the
    // interaction webhook regardless of whether the LLM picked the
    // DM or in-channel send tool. The interaction reply lands in the
    // original "Robutler is thinking..." bubble in the channel.
    const slash = this.activeSlashInteraction(ctx);
    if (slash) {
      return this.discordCall('send_in_channel', async () => {
        return this.sendInteractionFollowup(slash, {
          content: args.content,
          embeds: args.embeds,
        });
      });
    }
    const channelId = args.channel_id ?? this.bridgeChannel(ctx);
    if (!channelId) return this.invalidInput('channel_id required');
    const content = this.maybePrependBridgeMention(ctx, channelId, args.content);
    return this.discordCall('send_in_channel', (token) =>
      this.discordFetchRaw(token, `/channels/${channelId}/messages`, {
        method: 'POST',
        body: JSON.stringify({ content, embeds: args.embeds }),
      }),
    );
  }

  @tool({
    name: 'discord_send_webhook',
    description:
      'Post a message to the Discord channel configured by a Discord incoming webhook URL. ' +
      'This lightweight mode can only post into that one channel; it cannot read messages, DM users, or manage slash commands.',
    parameters: {
      type: 'object',
      properties: {
        content: { type: 'string' },
        username: { type: 'string' },
        avatar_url: { type: 'string' },
        embeds: {
          type: 'array',
          items: { type: 'object', additionalProperties: true },
        },
      },
      required: ['content'],
    },
    requiresConfirmation: true,
  })
  async sendWebhook(args: {
    content: string;
    username?: string;
    avatar_url?: string;
    embeds?: unknown[];
  }) {
    if (!this.capabilityEnabled('webhook_post') && !this.capabilityEnabled('publish_posts')) {
      return this.capabilityDisabled('webhook_post');
    }
    return this.discordCall('send_webhook', async (webhookUrl, metadata) => {
      const meta = (metadata ?? {}) as DiscordMetadata;
      if (meta.credentialType !== 'webhook') throw new Error('discord_webhook_not_configured');
      return this.discordWebhookFetchRaw(webhookUrl, {
        content: args.content,
        ...(args.username ? { username: args.username } : {}),
        ...(args.avatar_url ? { avatar_url: args.avatar_url } : {}),
        ...(args.embeds ? { embeds: args.embeds } : {}),
      });
    });
  }

  @tool({
    name: 'discord_send_in_channel_photo',
    description:
      'Send an image attachment to a Discord channel. Pass either ' +
      '`content_id` or `image_url`; the skill uploads via multipart.',
    parameters: {
      type: 'object',
      properties: {
        channel_id: { type: 'string' },
        content_id: { type: 'string' },
        image_url: { type: 'string' },
        content: { type: 'string' },
        filename: { type: 'string' },
      },
      required: ['channel_id'],
    },
    requiresConfirmation: true,
  })
  async sendInChannelPhoto(args: {
    channel_id: string;
    content_id?: string;
    image_url?: string;
    content?: string;
    filename?: string;
  }) {
    if (!this.capabilityEnabled('send_messages') && !this.capabilityEnabled('publish_posts')) {
      return this.capabilityDisabled('send_messages');
    }
    return this.sendChannelAttachment({
      channel_id: args.channel_id,
      content_id: args.content_id,
      url: args.image_url,
      content: args.content,
      filename: args.filename,
      kind: 'image',
    });
  }

  @tool({
    name: 'discord_send_in_channel_document',
    description: 'Send a file attachment to a Discord channel via multipart upload.',
    parameters: {
      type: 'object',
      properties: {
        channel_id: { type: 'string' },
        content_id: { type: 'string' },
        document_url: { type: 'string' },
        content: { type: 'string' },
        filename: { type: 'string' },
      },
      required: ['channel_id'],
    },
    requiresConfirmation: true,
  })
  async sendInChannelDocument(args: {
    channel_id: string;
    content_id?: string;
    document_url?: string;
    content?: string;
    filename?: string;
  }) {
    if (!this.capabilityEnabled('send_messages') && !this.capabilityEnabled('publish_posts')) {
      return this.capabilityDisabled('send_messages');
    }
    return this.sendChannelAttachment({
      channel_id: args.channel_id,
      content_id: args.content_id,
      url: args.document_url,
      content: args.content,
      filename: args.filename,
      kind: 'document',
    });
  }

  @tool({
    name: 'discord_register_slash_command',
    description: 'Register a guild-scoped slash command for the bot.',
    parameters: {
      type: 'object',
      properties: {
        guild_id: { type: 'string' },
        name: { type: 'string' },
        description: { type: 'string' },
        options: {
          type: 'array',
          items: { type: 'object', additionalProperties: true },
        },
      },
      required: ['guild_id', 'name', 'description'],
    },
    // Slash command registration permanently changes the bot's API
    // surface in the guild and is owner-only configuration. Restrict
    // to the agent owner AND require explicit approval per call.
    audience: 'owner',
    requiresConfirmation: true,
  })
  async registerSlashCommand(args: {
    guild_id: string;
    name: string;
    description: string;
    options?: unknown[];
  }) {
    return this.discordCall('register_slash_command', async (token, metadata) => {
      const meta = (metadata ?? {}) as DiscordMetadata;
      if (!meta.applicationId) throw new Error('application_id_missing');
      return this.discordFetchRaw(
        token,
        `/applications/${meta.applicationId}/guilds/${args.guild_id}/commands`,
        {
          method: 'POST',
          body: JSON.stringify({
            name: args.name,
            description: args.description,
            options: args.options ?? [],
          }),
        },
      );
    });
  }

  @prompt({ name: 'discord-bridge-awareness' })
  async bridgePrompt(ctx?: Context): Promise<string | null> {
    const b = bridgeMatches(ctx, PROVIDER);
    if (!b) return null;
    // Pick the right send tool based on bridge kind so the prompt names
    // the tool the LLM should actually call. Guild @-mention bridges
    // route through `discord_send_in_channel` (DM-only `discord_send_dm`
    // isn't even registered in their tool list — see the
    // `requiresBridge: { kind: 'dm' }` gate above).
    const sendToolName =
      b.kind === 'mention' ? 'discord_send_in_channel' : SEND_TOOL;
    return buildBridgeAwarenessPrompt({
      provider: PROVIDER,
      contactDisplayName: b.contactDisplayName,
      sendToolName,
    });
  }

  /**
   * Discord interactions endpoint — verifies Ed25519 signature and replies
   * to PING with PONG (Discord's mandatory liveness probe). Slash command
   * dispatch is deferred to the host; the skill's role is verification +
   * ack.
   */
  @http({
    path: '/messaging/discord/interactions',
    method: 'POST',
    auth: 'signature',
    description: 'Discord interactions endpoint with Ed25519 verification.',
  })
  async interactions(req: Request): Promise<Response> {
    const t = await this.resolveToken();
    const meta = (t.metadata ?? {}) as DiscordMetadata;
    const publicKey = meta.publicKey ?? process.env.DISCORD_PUBLIC_KEY ?? '';
    const rawBody = await req.text();
    const ok = await verifyDiscordSignature({
      publicKey,
      signatureHex: req.headers.get('X-Signature-Ed25519') ?? '',
      timestamp: req.headers.get('X-Signature-Timestamp') ?? '',
      rawBody,
    });
    if (!ok) return new Response('invalid request signature', { status: 401 });
    const json = JSON.parse(rawBody) as { type: number };
    if (json.type === 1) {
      return new Response(JSON.stringify({ type: 1 }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    // Type 2/3/4/5 — defer + let the host pick it up via tail logs / hook.
    console.log(`[discord-interactions] type=${json.type}`);
    return new Response(JSON.stringify({ type: 5 }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  /** Standalone OAuth callback. Portal hosts use the central route. */
  @http({
    path: '/messaging/discord/oauth/callback',
    method: 'GET',
    auth: 'public',
    description: 'Discord OAuth2 redirect target for non-portal hosts.',
  })
  async oauthCallback(req: Request): Promise<Response> {
    return handleOAuthCallback({
      request: req,
      provider: PROVIDER,
      redirectUri: `${this.httpBaseUrl}/api/agents/${this.agentId ?? 'agent'}/messaging/discord/oauth/callback`,
      tokenUrl: 'https://discord.com/api/v10/oauth2/token',
      tokenWriter: this.tokenWriter,
      agentId: this.agentId,
      integrationId: this.integrationId,
    });
  }

  private async sendDmAttachment(
    args: {
      user_id?: string;
      content_id?: string;
      url?: string;
      content?: string;
      filename?: string;
      kind: 'image' | 'document';
    },
    ctx: Context | undefined,
  ) {
    const slash = this.activeSlashInteraction(ctx);
    const resolved = await this.resolveOutboundMedia({
      contentId: args.content_id,
      url: args.url,
    });
    if ('error' in resolved) return resolved.error;
    if (slash) {
      // Slash-command path: post a follow-up to the interaction
      // webhook with the file as multipart. The interaction webhook
      // accepts the same `files[N]` form Discord channels do.
      return this.discordCall(args.kind === 'image' ? 'send_dm_photo' : 'send_dm_document', async () => {
        return this.uploadAndSendInteractionFollowup({
          slash,
          media: resolved.media,
          content: args.content,
          filename: args.filename,
        });
      });
    }
    const userId = args.user_id ?? this.bridgeRecipient(ctx);
    if (!userId) return this.invalidInput('user_id required');
    return this.discordCall(args.kind === 'image' ? 'send_dm_photo' : 'send_dm_document', async (token) => {
      const open = await this.discordFetchRaw(token, '/users/@me/channels', {
        method: 'POST',
        body: JSON.stringify({ recipient_id: userId }),
      });
      const channelId = (open as { id?: string }).id;
      if (!channelId) throw new Error('open_dm_failed');
      return this.uploadAndSend({
        token,
        channelId,
        media: resolved.media,
        content: args.content,
        filename: args.filename,
      });
    });
  }

  private async sendChannelAttachment(args: {
    channel_id: string;
    content_id?: string;
    url?: string;
    content?: string;
    filename?: string;
    kind: 'image' | 'document';
  }) {
    const resolved = await this.resolveOutboundMedia({
      contentId: args.content_id,
      url: args.url,
    });
    if ('error' in resolved) return resolved.error;
    return this.discordCall(
      args.kind === 'image' ? 'send_channel_photo' : 'send_channel_document',
      (token) =>
        this.uploadAndSend({
          token,
          channelId: args.channel_id,
          media: resolved.media,
          content: args.content,
          filename: args.filename,
        }),
    );
  }

  private async uploadAndSend(input: {
    token: string;
    channelId: string;
    media: import('../shared').ResolvedOutboundMedia;
    content?: string;
    filename?: string;
  }): Promise<unknown> {
    const { token, channelId, media, content, filename } = input;
    if (!media.fetchBytes) {
      throw new Error('discord_requires_bytes');
    }
    const bytes = await media.fetchBytes();
    const form = new FormData();
    const payload: Record<string, unknown> = { ...(content ? { content } : {}) };
    form.set('payload_json', JSON.stringify(payload));
    // Discord's image / video / audio inline preview is gated on the
    // upload filename having a recognised media extension — uploads
    // without one (e.g. host-side `displayName: "Generated image"`)
    // render as a generic "📄 file" box even when the multipart MIME
    // is `image/png`. Always force a content-type-derived extension
    // when the resolved filename doesn't already carry a sensible one.
    const ext = guessExt(bytes.contentType);
    const ensureExt = (name: string | undefined): string => {
      const fallback = `upload.${ext}`;
      const trimmed = (name ?? '').trim();
      if (!trimmed) return fallback;
      // Has any extension at all? Keep it (the user / resolver may
      // have supplied something more descriptive, e.g. `chart.svg`).
      if (/\.[a-z0-9]{1,8}$/i.test(trimmed)) return trimmed;
      return `${trimmed}.${ext}`;
    };
    form.set(
      'files[0]',
      new Blob([bytes.buffer as BlobPart], { type: bytes.contentType }),
      ensureExt(filename ?? bytes.filename),
    );
    const r = await fetch(`https://discord.com/api/v10/channels/${channelId}/messages`, {
      method: 'POST',
      headers: { Authorization: `Bot ${token}` },
      body: form,
    });
    const text = await r.text();
    if (!r.ok) {
      const e = new Error(text.slice(0, 200)) as Error & { status?: number };
      e.status = r.status;
      throw e;
    }
    return text ? JSON.parse(text) : {};
  }

  /**
   * Discord interaction tokens (per `/robutler` invocation) are valid
   * for 15 minutes after the interaction was received. Outbound calls
   * past that window MUST fall back to a different transport — the
   * webhook returns 404 once the token expires.
   *
   * Returns the active slash-command interaction descriptor when:
   *   • the current bridge is a Discord slash command, AND
   *   • the interaction token + applicationId are present, AND
   *   • we're still inside the 15-minute TTL.
   * Otherwise returns null and the caller falls through to its
   * original transport (DM channel for `sendDm`, channel post for
   * `sendInChannel`).
   */
  private activeSlashInteraction(ctx: Context | undefined): {
    applicationId: string;
    token: string;
  } | null {
    const bridge = (ctx?.metadata as Record<string, unknown> | undefined)?.bridge as
      | {
          source?: string;
          kind?: 'dm' | 'mention' | 'slash_command';
          interactionToken?: string;
          interactionTokenIssuedAt?: number;
          applicationId?: string;
        }
      | undefined;
    if (!bridge || bridge.source !== PROVIDER) return null;
    if (bridge.kind !== 'slash_command') return null;
    if (!bridge.interactionToken || !bridge.applicationId) return null;
    const issuedAt = bridge.interactionTokenIssuedAt;
    if (typeof issuedAt === 'number' && Date.now() - issuedAt > 15 * 60 * 1000) {
      // Token expired — log and let the caller use the fallback path.
      console.warn(
        `[discord] interaction token expired (issued ${Math.round((Date.now() - issuedAt) / 1000)}s ago); falling back to DM transport`,
      );
      return null;
    }
    return { applicationId: bridge.applicationId, token: bridge.interactionToken };
  }

  /**
   * Send a follow-up message to the per-interaction webhook
   * (`POST /webhooks/{app_id}/{token}`). Used for chunked replies +
   * status updates after the initial deferred ack — every send becomes
   * a fresh follow-up message in Discord, which is what makes the
   * "send multiple short messages" UX work.
   *
   * No bot-token auth required: the webhook URL itself is the
   * authentication artifact for interaction follow-ups.
   */
  private async sendInteractionFollowup(
    slash: { applicationId: string; token: string },
    payload: { content?: string; embeds?: unknown[] },
  ): Promise<unknown> {
    const base = `https://discord.com/api/v10/webhooks/${slash.applicationId}/${slash.token}`;
    // First reply for this interaction must PATCH the deferred "@original"
    // placeholder (Discord's "Robutler is thinking…" bubble). Subsequent
    // replies POST as additional follow-up messages.
    const isFirst = !this.slashOriginalIsFilled(slash.token);
    const url = isFirst ? `${base}/messages/@original` : base;
    const method = isFirst ? 'PATCH' : 'POST';
    const previewSrc = payload.content ?? JSON.stringify(payload.embeds ?? null);
    const preview = previewSrc.slice(0, 80);
    console.info(
      '[discord/skill] interaction_followup_send',
      JSON.stringify({
        applicationId: slash.applicationId,
        tokenPrefix: slash.token.slice(0, 12),
        isFirst,
        method,
        contentLen: payload.content?.length ?? 0,
        embeds: payload.embeds?.length ?? 0,
        preview,
      }),
    );
    let r: Response;
    try {
      r = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      console.error(
        '[discord/skill] interaction_followup_fetch_throw',
        JSON.stringify({
          applicationId: slash.applicationId,
          tokenPrefix: slash.token.slice(0, 12),
          err: (err as Error).message,
        }),
      );
      throw err;
    }
    const text = await r.text();
    console.info(
      '[discord/skill] interaction_followup_response',
      JSON.stringify({
        applicationId: slash.applicationId,
        tokenPrefix: slash.token.slice(0, 12),
        method,
        status: r.status,
        ok: r.ok,
        bodyPreview: text.slice(0, 200),
      }),
    );
    if (!r.ok) {
      const e = new Error(text.slice(0, 200)) as Error & { status?: number };
      e.status = r.status;
      throw e;
    }
    if (isFirst) this.markSlashOriginalFilled(slash.token);
    return text ? JSON.parse(text) : {};
  }

  /**
   * Upload an attachment via interaction follow-up. Same multipart
   * `files[N]` shape as `uploadAndSend` but POSTs to the interaction
   * webhook URL (no bot-token auth needed; webhook URL itself is
   * the auth artifact).
   */
  private async uploadAndSendInteractionFollowup(input: {
    slash: { applicationId: string; token: string };
    media: import('../shared').ResolvedOutboundMedia;
    content?: string;
    filename?: string;
  }): Promise<unknown> {
    const { slash, media, content, filename } = input;
    if (!media.fetchBytes) {
      throw new Error('discord_requires_bytes');
    }
    const bytes = await media.fetchBytes();
    const form = new FormData();
    const payload: Record<string, unknown> = { ...(content ? { content } : {}) };
    form.set('payload_json', JSON.stringify(payload));
    const ext = guessExt(bytes.contentType);
    const ensureExt = (name: string | undefined): string => {
      const fallback = `upload.${ext}`;
      const trimmed = (name ?? '').trim();
      if (!trimmed) return fallback;
      if (/\.[a-z0-9]{1,8}$/i.test(trimmed)) return trimmed;
      return `${trimmed}.${ext}`;
    };
    form.set(
      'files[0]',
      new Blob([bytes.buffer as BlobPart], { type: bytes.contentType }),
      ensureExt(filename ?? bytes.filename),
    );
    const base = `https://discord.com/api/v10/webhooks/${slash.applicationId}/${slash.token}`;
    const isFirst = !this.slashOriginalIsFilled(slash.token);
    const url = isFirst ? `${base}/messages/@original` : base;
    const method = isFirst ? 'PATCH' : 'POST';
    const r = await fetch(url, { method, body: form });
    const text = await r.text();
    if (!r.ok) {
      const e = new Error(text.slice(0, 200)) as Error & { status?: number };
      e.status = r.status;
      throw e;
    }
    if (isFirst) this.markSlashOriginalFilled(slash.token);
    return text ? JSON.parse(text) : {};
  }

  /**
   * When the current run is a guild @-mention reply landing in the
   * same channel the user pinged us in, prepend `<@authorId> ` so the
   * user gets a Discord ping on the bot's reply (otherwise they'd have
   * to re-open the channel to notice). Idempotent: if the LLM already
   * included the mention we leave the content untouched.
   *
   * For non-mention bridges, an out-of-band `discord_send_in_channel`
   * call (different channel, or DM bridge), or a missing
   * `contactExternalId`, this is a no-op.
   */
  private maybePrependBridgeMention(
    ctx: Context | undefined,
    channelId: string,
    content: string,
  ): string {
    const bridge = (ctx?.metadata as Record<string, unknown> | undefined)?.bridge as
      | {
          source?: string;
          kind?: 'dm' | 'mention';
          channelId?: string;
          contactExternalId?: string;
        }
      | undefined;
    if (!bridge || bridge.source !== PROVIDER) return content;
    if (bridge.kind !== 'mention') return content;
    if (!bridge.channelId || bridge.channelId !== channelId) return content;
    const authorId = bridge.contactExternalId;
    if (!authorId || !/^\d+$/.test(authorId)) return content;
    const mention = `<@${authorId}>`;
    if (content.includes(mention) || content.includes(`<@!${authorId}>`)) return content;
    return `${mention} ${content}`;
  }

  private async discordCall(
    callType: string,
    fn: (token: string, metadata?: Record<string, unknown> | null) => Promise<unknown>,
  ) {
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: callType,
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        return fn(t.token, t.metadata);
      },
    );
  }

  private async discordFetchRaw(
    token: string,
    path: string,
    init: RequestInit,
  ): Promise<unknown> {
    const r = await fetch(`https://discord.com/api/v10${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bot ${token}`,
        ...((init.headers as Record<string, string>) ?? {}),
      },
    });
    const text = await r.text();
    if (!r.ok) {
      const e = new Error(text.slice(0, 200)) as Error & { status?: number };
      e.status = r.status;
      throw e;
    }
    return text ? JSON.parse(text) : {};
  }

  private async discordWebhookFetchRaw(
    webhookUrl: string,
    payload: Record<string, unknown>,
  ): Promise<unknown> {
    const separator = webhookUrl.includes('?') ? '&' : '?';
    const r = await fetch(`${webhookUrl}${separator}wait=true`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const text = await r.text();
    if (!r.ok) {
      const e = new Error(text.slice(0, 200)) as Error & { status?: number };
      e.status = r.status;
      throw e;
    }
    return text ? JSON.parse(text) : {};
  }
}

function guessExt(contentType: string): string {
  const t = contentType.toLowerCase();
  if (t.includes('jpeg') || t.includes('jpg')) return 'jpg';
  if (t.includes('png')) return 'png';
  if (t.includes('gif')) return 'gif';
  if (t.includes('webp')) return 'webp';
  if (t.includes('pdf')) return 'pdf';
  if (t.includes('mp4')) return 'mp4';
  if (t.includes('mpeg')) return 'mp3';
  return 'bin';
}
