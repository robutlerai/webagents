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
}

export class DiscordSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('discord', opts);
  }

  @tool({
    name: SEND_TOOL,
    description:
      'Send a DM to a Discord user (opens a DM channel first via POST /users/@me/channels per ' +
      'Discord API v10). THIS IS THE ONLY WAY TO REPLY TO A BRIDGED DISCORD CONTACT — a plain ' +
      'assistant message stays in the portal and never reaches the user.',
    parameters: {
      type: 'object',
      properties: {
        user_id: { type: 'string' },
        content: { type: 'string' },
      },
      required: ['content'],
    },
  })
  async sendDm(args: { user_id?: string; content: string }, ctx?: Context) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
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
    description: 'Send a message to a Discord channel the bot has access to.',
    parameters: {
      type: 'object',
      properties: {
        channel_id: { type: 'string' },
        content: { type: 'string' },
        embeds: { type: 'array' },
      },
      required: ['channel_id', 'content'],
    },
  })
  async sendInChannel(args: {
    channel_id: string;
    content: string;
    embeds?: unknown[];
  }) {
    if (!this.capabilityEnabled('send_messages') && !this.capabilityEnabled('publish_posts')) {
      return this.capabilityDisabled('send_messages');
    }
    return this.discordCall('send_in_channel', (token) =>
      this.discordFetchRaw(token, `/channels/${args.channel_id}/messages`, {
        method: 'POST',
        body: JSON.stringify({ content: args.content, embeds: args.embeds }),
      }),
    );
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
        options: { type: 'array' },
      },
      required: ['guild_id', 'name', 'description'],
    },
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
    return buildBridgeAwarenessPrompt({
      provider: PROVIDER,
      contactDisplayName: b.contactDisplayName,
      sendToolName: SEND_TOOL,
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
    const userId = args.user_id ?? this.bridgeRecipient(ctx);
    if (!userId) return this.invalidInput('user_id required');
    const resolved = await this.resolveOutboundMedia({
      contentId: args.content_id,
      url: args.url,
    });
    if ('error' in resolved) return resolved.error;
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
    form.set(
      'files[0]',
      new Blob([bytes.buffer as BlobPart], { type: bytes.contentType }),
      filename ?? bytes.filename ?? `upload.${guessExt(bytes.contentType)}`,
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
