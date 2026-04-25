/**
 * SlackSkill — Slack Web API + interactivity endpoint.
 *
 * Token resolution: ResolvedToken.token = bot token (xoxb-…). Metadata
 * carries `teamId`, `botUserId`, and `signingSecret` (used to verify
 * inbound interactivity payloads on the @http endpoint). When the host
 * has post-approval gating, `enabledCapabilities` should NOT include
 * `publish_posts` for the no-approval tools and the host instead routes
 * channel posts through its own approval queue.
 *
 * File upload uses files.getUploadURLExternal + completeUploadExternal
 * (files.upload was retired Nov 12 2025 per Slack's deprecation notice).
 */
import { http, prompt, tool } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  handleOAuthCallback,
  MessagingSkill,
  verifySlackSignature,
  type MessagingSkillOptions,
} from '../shared';

const PROVIDER = 'slack';
const SEND_TOOL = 'slack_send_dm';

interface SlackMetadata {
  teamId?: string;
  botUserId?: string;
  signingSecret?: string;
}

export class SlackSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('slack', opts);
  }

  @tool({
    name: SEND_TOOL,
    description:
      'Send a DM to a Slack user (opens an IM channel via conversations.open + chat.postMessage). ' +
      'THIS IS THE ONLY WAY TO REPLY TO A BRIDGED SLACK CONTACT — a plain assistant message ' +
      'stays in the portal and never reaches the user.',
    parameters: {
      type: 'object',
      properties: {
        user_id: { type: 'string' },
        text: { type: 'string' },
        blocks: { type: 'array' },
      },
      required: ['text'],
    },
  })
  async sendDm(args: { user_id?: string; text: string; blocks?: unknown[] }, ctx?: Context) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const user = args.user_id ?? this.bridgeRecipient(ctx);
    if (!user) return this.invalidInput('user_id required');
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'send_dm',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        const open = await fetch('https://slack.com/api/conversations.open', {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${t.token}`,
            'Content-Type': 'application/json; charset=utf-8',
          },
          body: JSON.stringify({ users: user }),
        });
        const oj = (await open.json()) as { ok: boolean; channel?: { id: string }; error?: string };
        if (!oj.ok || !oj.channel) throw new Error(oj.error ?? 'open_failed');
        return this.slackPostRaw(t.token, 'chat.postMessage', {
          channel: oj.channel.id,
          text: args.text,
          blocks: args.blocks,
        });
      },
    );
  }

  @tool({
    name: 'slack_send_dm_photo',
    description:
      'Send an image as a DM to a Slack user via the 3-step ' +
      '`files.getUploadURLExternal` flow (the legacy `files.upload` was ' +
      'retired). Pass either `content_id` or `image_url`; the skill ' +
      'fetches bytes server-side and shares the uploaded file in the IM ' +
      'channel.',
    parameters: {
      type: 'object',
      properties: {
        user_id: { type: 'string' },
        content_id: { type: 'string' },
        image_url: { type: 'string' },
        initial_comment: { type: 'string' },
        filename: { type: 'string' },
        title: { type: 'string' },
      },
      required: [],
    },
  })
  async sendDmPhoto(
    args: {
      user_id?: string;
      content_id?: string;
      image_url?: string;
      initial_comment?: string;
      filename?: string;
      title?: string;
    },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    return this.sendDmFile({ ...args, url: args.image_url, kind: 'image' }, ctx);
  }

  @tool({
    name: 'slack_send_dm_document',
    description: 'Send a file (document) as a DM to a Slack user via files.getUploadURLExternal.',
    parameters: {
      type: 'object',
      properties: {
        user_id: { type: 'string' },
        content_id: { type: 'string' },
        document_url: { type: 'string' },
        initial_comment: { type: 'string' },
        filename: { type: 'string' },
        title: { type: 'string' },
      },
      required: [],
    },
  })
  async sendDmDocument(
    args: {
      user_id?: string;
      content_id?: string;
      document_url?: string;
      initial_comment?: string;
      filename?: string;
      title?: string;
    },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    return this.sendDmFile({ ...args, url: args.document_url, kind: 'document' }, ctx);
  }

  @tool({
    name: 'slack_post_in_channel_photo',
    description:
      'Share an image to a Slack channel via the 3-step ' +
      '`files.getUploadURLExternal` flow.',
    parameters: {
      type: 'object',
      properties: {
        channel: { type: 'string' },
        content_id: { type: 'string' },
        image_url: { type: 'string' },
        initial_comment: { type: 'string' },
        filename: { type: 'string' },
        title: { type: 'string' },
        thread_ts: { type: 'string' },
      },
      required: ['channel'],
    },
  })
  async postInChannelPhoto(args: {
    channel: string;
    content_id?: string;
    image_url?: string;
    initial_comment?: string;
    filename?: string;
    title?: string;
    thread_ts?: string;
  }) {
    if (!this.capabilityEnabled('publish_posts') && !this.capabilityEnabled('send_messages')) {
      return this.capabilityDisabled('publish_posts');
    }
    return this.shareFileToChannel({
      channel: args.channel,
      content_id: args.content_id,
      url: args.image_url,
      initial_comment: args.initial_comment,
      filename: args.filename,
      title: args.title,
      thread_ts: args.thread_ts,
      kind: 'image',
    });
  }

  @tool({
    name: 'slack_post_in_channel_document',
    description: 'Share a file (document) to a Slack channel via files.getUploadURLExternal.',
    parameters: {
      type: 'object',
      properties: {
        channel: { type: 'string' },
        content_id: { type: 'string' },
        document_url: { type: 'string' },
        initial_comment: { type: 'string' },
        filename: { type: 'string' },
        title: { type: 'string' },
        thread_ts: { type: 'string' },
      },
      required: ['channel'],
    },
  })
  async postInChannelDocument(args: {
    channel: string;
    content_id?: string;
    document_url?: string;
    initial_comment?: string;
    filename?: string;
    title?: string;
    thread_ts?: string;
  }) {
    if (!this.capabilityEnabled('publish_posts') && !this.capabilityEnabled('send_messages')) {
      return this.capabilityDisabled('publish_posts');
    }
    return this.shareFileToChannel({
      channel: args.channel,
      content_id: args.content_id,
      url: args.document_url,
      initial_comment: args.initial_comment,
      filename: args.filename,
      title: args.title,
      thread_ts: args.thread_ts,
      kind: 'document',
    });
  }

  @tool({
    name: 'slack_post_in_channel',
    description: 'Post a message in a Slack channel.',
    parameters: {
      type: 'object',
      properties: {
        channel: { type: 'string' },
        text: { type: 'string' },
        blocks: { type: 'array' },
        thread_ts: { type: 'string' },
      },
      required: ['channel', 'text'],
    },
  })
  async postInChannel(args: {
    channel: string;
    text: string;
    blocks?: unknown[];
    thread_ts?: string;
  }) {
    if (!this.capabilityEnabled('publish_posts') && !this.capabilityEnabled('send_messages')) {
      return this.capabilityDisabled('publish_posts');
    }
    return this.slackPost('chat.postMessage', 'post_in_channel', args);
  }

  @tool({
    name: 'slack_reply_in_thread',
    description: 'Reply in an existing Slack thread.',
    parameters: {
      type: 'object',
      properties: {
        channel: { type: 'string' },
        thread_ts: { type: 'string' },
        text: { type: 'string' },
      },
      required: ['channel', 'thread_ts', 'text'],
    },
  })
  async replyInThread(args: { channel: string; thread_ts: string; text: string }) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    return this.slackPost('chat.postMessage', 'reply_in_thread', args);
  }

  @prompt({ name: 'slack-bridge-awareness' })
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
   * Slack interactivity endpoint — handles button clicks / view submissions.
   * Must respond within 3 seconds (Slack's hard limit) so the handler ack's
   * immediately and dispatches asynchronous work via the wrapApiCall layer.
   */
  @http({
    path: '/messaging/slack/interactivity',
    method: 'POST',
    auth: 'signature',
    description: 'Slack interactivity endpoint (block actions, view submissions). Verifies v0 signature.',
  })
  async interactivity(req: Request): Promise<Response> {
    const t = await this.resolveToken();
    const meta = (t.metadata ?? {}) as SlackMetadata;
    const signingSecret = meta.signingSecret ?? process.env.SLACK_SIGNING_SECRET ?? '';
    const rawBody = await req.text();
    const ok = verifySlackSignature({
      signingSecret,
      timestamp: req.headers.get('X-Slack-Request-Timestamp') ?? '',
      signature: req.headers.get('X-Slack-Signature') ?? '',
      rawBody,
    });
    if (!ok) return new Response('forbidden', { status: 403 });
    // The skill ack's immediately; consumers (host) read the structured log.
    console.log(`[slack-interactivity] body_len=${rawBody.length}`);
    return new Response('', { status: 200 });
  }

  /**
   * Standalone OAuth callback for Slack v2. Portal hosts use the
   * central /api/auth/connect/slack/callback route and ignore this.
   * Slack v2 returns a `team` and `authed_user` block — extract teamId
   * + bot user id into TokenWriter metadata so the SlackSkill can use
   * them on subsequent send calls.
   */
  @http({
    path: '/messaging/slack/oauth/callback',
    method: 'GET',
    auth: 'public',
    description: 'Slack OAuth2 v2 redirect target for non-portal hosts.',
  })
  async oauthCallback(req: Request): Promise<Response> {
    return handleOAuthCallback({
      request: req,
      provider: PROVIDER,
      redirectUri: `${this.httpBaseUrl}/api/agents/${this.agentId ?? 'agent'}/messaging/slack/oauth/callback`,
      tokenUrl: 'https://slack.com/api/oauth.v2.access',
      tokenWriter: this.tokenWriter,
      agentId: this.agentId,
      integrationId: this.integrationId,
      transformResult: (raw) => {
        const team = (raw.team as { id?: string } | undefined) ?? {};
        const authedUser = (raw.authed_user as { id?: string } | undefined) ?? {};
        return {
          metadata: { teamId: team.id, botUserId: authedUser.id },
          providerUserId: authedUser.id,
        };
      },
    });
  }

  // -------------------------------------------------------------------------

  private async sendDmFile(
    args: {
      user_id?: string;
      content_id?: string;
      url?: string;
      initial_comment?: string;
      filename?: string;
      title?: string;
      kind: 'image' | 'document';
    },
    ctx: Context | undefined,
  ) {
    const user = args.user_id ?? this.bridgeRecipient(ctx);
    if (!user) return this.invalidInput('user_id required');
    const resolved = await this.resolveOutboundMedia({
      contentId: args.content_id,
      url: args.url,
    });
    if ('error' in resolved) return resolved.error;
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: args.kind === 'image' ? 'send_dm_photo' : 'send_dm_document',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        const open = await fetch('https://slack.com/api/conversations.open', {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${t.token}`,
            'Content-Type': 'application/json; charset=utf-8',
          },
          body: JSON.stringify({ users: user }),
        });
        const oj = (await open.json()) as { ok: boolean; channel?: { id: string }; error?: string };
        if (!oj.ok || !oj.channel) throw new Error(oj.error ?? 'open_failed');
        const fileId = await this.uploadFile({
          token: t.token,
          media: resolved.media,
          filename: args.filename,
        });
        return this.completeUploadAndShare({
          token: t.token,
          fileId,
          channel: oj.channel.id,
          initialComment: args.initial_comment,
          title: args.title,
        });
      },
    );
  }

  private async shareFileToChannel(args: {
    channel: string;
    content_id?: string;
    url?: string;
    initial_comment?: string;
    filename?: string;
    title?: string;
    thread_ts?: string;
    kind: 'image' | 'document';
  }) {
    const resolved = await this.resolveOutboundMedia({
      contentId: args.content_id,
      url: args.url,
    });
    if ('error' in resolved) return resolved.error;
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: args.kind === 'image' ? 'post_in_channel_photo' : 'post_in_channel_document',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        const fileId = await this.uploadFile({
          token: t.token,
          media: resolved.media,
          filename: args.filename,
        });
        return this.completeUploadAndShare({
          token: t.token,
          fileId,
          channel: args.channel,
          initialComment: args.initial_comment,
          title: args.title,
          threadTs: args.thread_ts,
        });
      },
    );
  }

  private async uploadFile(input: {
    token: string;
    media: import('../shared').ResolvedOutboundMedia;
    filename?: string;
  }): Promise<string> {
    const { token, media } = input;
    if (!media.fetchBytes) {
      throw new Error('slack_requires_bytes');
    }
    const bytes = await media.fetchBytes();
    const filename = input.filename ?? bytes.filename ?? `upload.${guessExt(bytes.contentType)}`;
    const params = new URLSearchParams({
      filename,
      length: String(bytes.buffer.byteLength),
    });
    const stepOne = await fetch(
      `https://slack.com/api/files.getUploadURLExternal?${params.toString()}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const so = (await stepOne.json()) as {
      ok: boolean;
      upload_url?: string;
      file_id?: string;
      error?: string;
    };
    if (!so.ok || !so.upload_url || !so.file_id) {
      throw new Error(so.error ?? 'files_get_upload_url_failed');
    }
    const stepTwo = await fetch(so.upload_url, {
      method: 'POST',
      body: new Blob([bytes.buffer as BlobPart], { type: bytes.contentType }),
    });
    if (!stepTwo.ok) {
      throw new Error(`files_upload_failed_${stepTwo.status}`);
    }
    return so.file_id;
  }

  private async completeUploadAndShare(input: {
    token: string;
    fileId: string;
    channel: string;
    initialComment?: string;
    title?: string;
    threadTs?: string;
  }): Promise<{ externalMessageId?: string; ok: boolean; file_id?: string }> {
    const body: Record<string, unknown> = {
      files: [{ id: input.fileId, ...(input.title ? { title: input.title } : {}) }],
      channel_id: input.channel,
      ...(input.initialComment ? { initial_comment: input.initialComment } : {}),
      ...(input.threadTs ? { thread_ts: input.threadTs } : {}),
    };
    const r = await fetch('https://slack.com/api/files.completeUploadExternal', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${input.token}`,
        'Content-Type': 'application/json; charset=utf-8',
      },
      body: JSON.stringify(body),
    });
    const j = (await r.json()) as { ok: boolean; error?: string };
    if (!r.ok || !j.ok) {
      throw new Error(j.error ?? `files_complete_upload_failed_${r.status}`);
    }
    return { ok: true, file_id: input.fileId, externalMessageId: input.fileId };
  }

  private async slackPost(
    method: string,
    callType: string,
    body: Record<string, unknown>,
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
        return this.slackPostRaw(t.token, method, body);
      },
    );
  }

  private async slackPostRaw(
    token: string,
    method: string,
    body: Record<string, unknown>,
  ): Promise<{ externalMessageId?: string; ok: boolean; ts?: string; channel?: string }> {
    const r = await fetch(`https://slack.com/api/${method}`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json; charset=utf-8',
      },
      body: JSON.stringify(body),
    });
    const j = (await r.json()) as {
      ok: boolean;
      error?: string;
      ts?: string;
      channel?: string;
    };
    if (!r.ok || !j.ok) {
      const e = new Error(j.error ?? `slack ${r.status}`) as Error & {
        status?: number;
        code?: string;
      };
      e.status = r.status;
      e.code = j.error;
      throw e;
    }
    return { externalMessageId: j.ts ? `${j.channel ?? ''}:${j.ts}` : undefined, ...j };
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
