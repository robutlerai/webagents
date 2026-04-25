/**
 * InstagramSkill — Instagram Messaging API + Graph (publish, comments).
 *
 * Token resolution: ResolvedToken.token = Page access token (Instagram
 * Messaging) or IG-User token (Graph publishing). Metadata MUST carry
 * `igUserId` (IG Business / Creator account id). Optional `appSecret`
 * enables `appsecret_proof`.
 *
 * Reference: https://developers.facebook.com/docs/instagram-platform
 */
import { prompt, tool } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  computeAppsecretProof,
  MessagingSkill,
  metaGraph,
  metaInstagramGraphUrl,
  type MessagingSkillOptions,
} from '../shared';

const PROVIDER = 'instagram';
const SEND_TOOL = 'instagram_send_dm';

interface InstagramMetadata {
  igUserId?: string;
  pageId?: string;
  appSecret?: string;
}

export class InstagramSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('instagram', opts);
  }

  @tool({
    name: SEND_TOOL,
    description:
      'Reply to an Instagram Direct conversation within the 24h customer-service window. ' +
      'Routes via the connected Page using the Instagram Messaging API.',
    parameters: {
      type: 'object',
      properties: {
        recipient_id: { type: 'string', description: 'IG-scoped sender id. Optional when bridged.' },
        text: { type: 'string' },
      },
      required: ['text'],
    },
  })
  async sendDm(args: { recipient_id?: string; text: string }, ctx?: Context) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const recipient = args.recipient_id ?? this.bridgeRecipient(ctx);
    if (!recipient) return this.invalidInput('recipient_id required');
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'send_dm',
        agentId: this.agentId,
        integrationId: this.integrationId,
        windowCheck: { contactExternalId: recipient },
      },
      async () => {
        const { creds } = await this.creds();
        if (!creds.pageId) throw new Error('instagram_pageid_missing');
        const res = await metaGraph<{ message_id?: string; recipient_id?: string }>(
          `/${creds.pageId}/messages`,
          {
            accessToken: creds.accessToken,
            appsecretProof: creds.appsecretProof,
            method: 'POST',
            body: {
              recipient: { id: recipient },
              message: { text: args.text },
              messaging_type: 'RESPONSE',
            },
          },
        );
        return { externalMessageId: res.message_id };
      },
    );
  }

  @tool({
    name: 'instagram_send_image',
    description:
      'Send an image attachment in an Instagram Direct conversation. Pass ' +
      'either `content_id` (Robutler UUID for internally-hosted media) or ' +
      '`image_url` (absolute https URL Meta can fetch). Subject to the ' +
      'same 24h customer-service window as text DMs.',
    parameters: {
      type: 'object',
      properties: {
        recipient_id: { type: 'string' },
        content_id: { type: 'string' },
        image_url: { type: 'string' },
        caption: { type: 'string' },
      },
      required: [],
    },
  })
  async sendImage(
    args: {
      recipient_id?: string;
      content_id?: string;
      image_url?: string;
      caption?: string;
    },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const recipient = args.recipient_id ?? this.bridgeRecipient(ctx);
    if (!recipient) return this.invalidInput('recipient_id required');
    const resolved = await this.resolveOutboundMedia({
      contentId: args.content_id,
      url: args.image_url,
    });
    if ('error' in resolved) return resolved.error;
    if (!/^https?:\/\//i.test(resolved.media.url)) {
      return this.invalidInput(
        'Instagram requires a publicly-reachable https URL. Internally-hosted content must surface an absolute URL the Meta Graph can fetch.',
        'instagram_requires_public_url',
      );
    }
    if (args.caption && args.caption.trim().length > 0) {
      const ack = await this.sendDm({ recipient_id: recipient, text: args.caption }, ctx);
      if (!ack.ok) return ack;
    }
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'send_image',
        agentId: this.agentId,
        integrationId: this.integrationId,
        windowCheck: { contactExternalId: recipient },
      },
      async () => {
        const { creds } = await this.creds();
        if (!creds.pageId) throw new Error('instagram_pageid_missing');
        const res = await metaGraph<{ message_id?: string }>(
          `/${creds.pageId}/messages`,
          {
            accessToken: creds.accessToken,
            appsecretProof: creds.appsecretProof,
            method: 'POST',
            body: {
              recipient: { id: recipient },
              messaging_type: 'RESPONSE',
              message: {
                attachment: {
                  type: 'image',
                  payload: { url: resolved.media.url, is_reusable: false },
                },
              },
            },
          },
        );
        return { externalMessageId: res.message_id };
      },
    );
  }

  @tool({
    name: 'instagram_publish_image',
    description:
      'Publish an image post to the connected Instagram account using the two-step container flow. ' +
      'Pass either `content_id` (Robutler UUID for internally-hosted media) or `image_url` ' +
      '(absolute https URL Meta can fetch). Hosts SHOULD wrap this through an approval gate when ' +
      'the integration requires post approval.',
    parameters: {
      type: 'object',
      properties: {
        content_id: { type: 'string' },
        image_url: { type: 'string' },
        caption: { type: 'string' },
      },
      required: [],
    },
  })
  async publishImage(args: { content_id?: string; image_url?: string; caption?: string }) {
    if (!this.capabilityEnabled('publish_posts')) return this.capabilityDisabled('publish_posts');
    const resolved = await this.resolveOutboundMedia({
      contentId: args.content_id,
      url: args.image_url,
    });
    if ('error' in resolved) return resolved.error;
    if (!/^https?:\/\//i.test(resolved.media.url)) {
      return this.invalidInput(
        'Instagram publish requires a publicly-reachable https URL. Internally-hosted content must surface an absolute URL the Meta Graph can fetch.',
        'instagram_requires_public_url',
      );
    }
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'publish_image',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const { creds } = await this.creds();
        const container = await metaGraph<{ id: string }>(
          `/${creds.igUserId}/media`,
          {
            accessToken: creds.accessToken,
            appsecretProof: creds.appsecretProof,
            method: 'POST',
            body: {
              image_url: resolved.media.url,
              ...(args.caption ? { caption: args.caption } : {}),
            },
          },
          metaInstagramGraphUrl,
        );
        const published = await metaGraph<{ id: string }>(
          `/${creds.igUserId}/media_publish`,
          {
            accessToken: creds.accessToken,
            appsecretProof: creds.appsecretProof,
            method: 'POST',
            body: { creation_id: container.id },
          },
          metaInstagramGraphUrl,
        );
        return { externalMessageId: published.id, containerId: container.id };
      },
    );
  }

  @tool({
    name: 'instagram_reply_to_comment',
    description: "Reply to a public comment on one of the Instagram account's posts.",
    parameters: {
      type: 'object',
      properties: { comment_id: { type: 'string' }, message: { type: 'string' } },
      required: ['comment_id', 'message'],
    },
  })
  async replyToComment(args: { comment_id: string; message: string }) {
    if (!this.capabilityEnabled('manage_comments')) return this.capabilityDisabled('manage_comments');
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'reply_comment',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const { creds } = await this.creds();
        return metaGraph<{ id: string }>(
          `/${args.comment_id}/replies`,
          {
            accessToken: creds.accessToken,
            appsecretProof: creds.appsecretProof,
            method: 'POST',
            body: { message: args.message },
          },
        );
      },
    );
  }

  @prompt({ name: 'instagram-bridge-awareness' })
  async bridgePrompt(ctx?: Context): Promise<string | null> {
    const b = bridgeMatches(ctx, PROVIDER);
    if (!b) return null;
    return buildBridgeAwarenessPrompt({
      provider: PROVIDER,
      contactDisplayName: b.contactDisplayName,
      sendToolName: SEND_TOOL,
    });
  }

  private async creds() {
    const t = await this.resolveToken();
    const meta = (t.metadata ?? {}) as InstagramMetadata;
    if (!meta.igUserId) throw new Error('instagram_iguserid_missing');
    const appSecret = meta.appSecret ?? process.env.META_APP_SECRET;
    return {
      creds: {
        accessToken: t.token,
        igUserId: meta.igUserId,
        pageId: meta.pageId,
        appsecretProof: appSecret ? computeAppsecretProof(t.token, appSecret) : undefined,
      },
    };
  }
}
