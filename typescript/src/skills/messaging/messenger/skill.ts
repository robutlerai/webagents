/**
 * MessengerSkill — Facebook Messenger Page tokens via Meta Graph.
 *
 * Token resolution: ResolvedToken.token = Page access token. Metadata MUST
 * carry `pageId`; an optional `appSecret` enables `appsecret_proof`.
 *
 * Reference: https://developers.facebook.com/docs/messenger-platform
 */
import { prompt, tool } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  computeAppsecretProof,
  MessagingSkill,
  metaGraph,
  type MessagingSkillOptions,
} from '../shared';

const PROVIDER = 'messenger';
const SEND_TOOL = 'messenger_send_text';

interface MessengerMetadata {
  pageId?: string;
  appSecret?: string;
}

export class MessengerSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('messenger', opts);
  }

  @tool({
    name: SEND_TOOL,
    description:
      'Reply to a Messenger conversation within the 24h customer-service window ' +
      '(RESPONSE messaging_type).',
    parameters: {
      type: 'object',
      properties: {
        recipient_psid: {
          type: 'string',
          description: 'Page-scoped Sender ID. Optional when bridged.',
        },
        text: { type: 'string' },
      },
      required: ['text'],
    },
  })
  async sendText(args: { recipient_psid?: string; text: string }, ctx?: Context) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const psid = args.recipient_psid ?? this.bridgeRecipient(ctx);
    if (!psid) return this.invalidInput('recipient_psid required');
    return this.send(psid, args.text, false);
  }

  @tool({
    name: 'messenger_send_image',
    description:
      'Send an image attachment in a Messenger conversation. Pass either ' +
      '`content_id` (Robutler UUID for internally-hosted media) or ' +
      '`image_url` (absolute https URL Meta can fetch). Meta downloads the ' +
      'URL itself — relative `/api/content/...` paths are not accepted. ' +
      'Subject to the same 24h customer-service window as text replies.',
    parameters: {
      type: 'object',
      properties: {
        recipient_psid: { type: 'string', description: 'Optional when bridged.' },
        content_id: { type: 'string' },
        image_url: { type: 'string' },
        caption: {
          type: 'string',
          description:
            "Optional separate text message sent immediately before the image (Messenger's image attachment has no native caption field).",
        },
      },
      required: [],
    },
  })
  async sendImage(
    args: {
      recipient_psid?: string;
      content_id?: string;
      image_url?: string;
      caption?: string;
    },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    return this.sendAttachment({ ...args, kind: 'image' }, ctx);
  }

  @tool({
    name: 'messenger_send_document',
    description:
      'Send a file attachment (pdf, document, etc.) in a Messenger ' +
      'conversation. Pass either `content_id` or an absolute https ' +
      '`document_url`. Subject to the same 24h customer-service window.',
    parameters: {
      type: 'object',
      properties: {
        recipient_psid: { type: 'string' },
        content_id: { type: 'string' },
        document_url: { type: 'string' },
        caption: { type: 'string' },
      },
      required: [],
    },
  })
  async sendDocument(
    args: {
      recipient_psid?: string;
      content_id?: string;
      document_url?: string;
      caption?: string;
    },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    return this.sendAttachment(
      {
        recipient_psid: args.recipient_psid,
        content_id: args.content_id,
        image_url: args.document_url,
        caption: args.caption,
        kind: 'file',
      },
      ctx,
    );
  }

  @tool({
    name: 'messenger_send_with_human_agent_tag',
    description:
      'Send a Messenger reply outside the 24h window using the HUMAN_AGENT message tag ' +
      '(allowed up to 7 days, requires the page to have the Human Agent permission approved).',
    parameters: {
      type: 'object',
      properties: {
        recipient_psid: { type: 'string' },
        text: { type: 'string' },
      },
      required: ['recipient_psid', 'text'],
    },
  })
  async sendWithHumanAgentTag(args: { recipient_psid: string; text: string }) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    return this.send(args.recipient_psid, args.text, true);
  }

  @tool({
    name: 'messenger_get_page_posts',
    description: "List the connected Page's recent posts.",
    parameters: {
      type: 'object',
      properties: { limit: { type: 'number', default: 20 } },
      required: [],
    },
  })
  async getPagePosts(args: { limit?: number }) {
    if (!this.capabilityEnabled('read_page_posts')) return this.capabilityDisabled('read_page_posts');
    const limit = Math.min(Math.max(args.limit ?? 20, 1), 100);
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'get_posts',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const { creds } = await this.creds();
        return metaGraph(
          `/${creds.pageId}/posts?fields=id,message,created_time,permalink_url&limit=${limit}`,
          { accessToken: creds.accessToken, appsecretProof: creds.appsecretProof },
        );
      },
    );
  }

  @tool({
    name: 'messenger_create_post',
    description:
      'Publish a post to the connected Facebook Page. Hosts SHOULD wrap this through ' +
      'an approval gate when the integration requires post approval.',
    parameters: {
      type: 'object',
      properties: { message: { type: 'string' }, link: { type: 'string' } },
      required: ['message'],
    },
  })
  async createPost(args: { message: string; link?: string }) {
    if (!this.capabilityEnabled('publish_page_posts')) return this.capabilityDisabled('publish_page_posts');
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'create_post',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const { creds } = await this.creds();
        const body: Record<string, unknown> = {
          message: args.message,
          ...(args.link ? { link: args.link } : {}),
        };
        const res = await metaGraph<{ id: string }>(`/${creds.pageId}/feed`, {
          accessToken: creds.accessToken,
          appsecretProof: creds.appsecretProof,
          method: 'POST',
          body,
        });
        return { externalMessageId: res.id };
      },
    );
  }

  @prompt({ name: 'messenger-bridge-awareness' })
  async bridgePrompt(ctx?: Context): Promise<string | null> {
    const b = bridgeMatches(ctx, PROVIDER);
    if (!b) return null;
    return buildBridgeAwarenessPrompt({
      provider: PROVIDER,
      contactDisplayName: b.contactDisplayName,
      sendToolName: SEND_TOOL,
    });
  }

  private async sendAttachment(
    args: {
      recipient_psid?: string;
      content_id?: string;
      image_url?: string;
      caption?: string;
      kind: 'image' | 'file';
    },
    ctx: Context | undefined,
  ) {
    const psid = args.recipient_psid ?? this.bridgeRecipient(ctx);
    if (!psid) return this.invalidInput('recipient_psid required');
    const resolved = await this.resolveOutboundMedia({
      contentId: args.content_id,
      url: args.image_url,
    });
    if ('error' in resolved) return resolved.error;
    if (!/^https?:\/\//i.test(resolved.media.url)) {
      return this.invalidInput(
        'Messenger requires a publicly-reachable https URL. Internally-hosted content must surface an absolute URL the Meta Graph can fetch.',
        'messenger_requires_public_url',
      );
    }
    if (args.caption && args.caption.trim().length > 0) {
      const ackResult = await this.send(psid, args.caption, false);
      if (!ackResult.ok) return ackResult;
    }
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: args.kind === 'image' ? 'send_image' : 'send_document',
        agentId: this.agentId,
        integrationId: this.integrationId,
        windowCheck: { contactExternalId: psid },
      },
      async () => {
        const { creds } = await this.creds();
        const body: Record<string, unknown> = {
          recipient: { id: psid },
          messaging_type: 'RESPONSE',
          message: {
            attachment: {
              type: args.kind,
              payload: { url: resolved.media.url, is_reusable: false },
            },
          },
        };
        const res = await metaGraph<{ message_id?: string }>(`/${creds.pageId}/messages`, {
          accessToken: creds.accessToken,
          method: 'POST',
          body,
          appsecretProof: creds.appsecretProof,
        });
        return { externalMessageId: res.message_id };
      },
    );
  }

  private async send(psid: string, text: string, humanAgentTag: boolean) {
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: humanAgentTag ? 'send_human_agent' : 'send_text',
        agentId: this.agentId,
        integrationId: this.integrationId,
        ...(humanAgentTag ? {} : { windowCheck: { contactExternalId: psid } }),
      },
      async () => {
        const { creds } = await this.creds();
        const body: Record<string, unknown> = {
          recipient: { id: psid },
          message: { text },
        };
        if (humanAgentTag) {
          body.messaging_type = 'MESSAGE_TAG';
          body.tag = 'HUMAN_AGENT';
        } else {
          body.messaging_type = 'RESPONSE';
        }
        const res = await metaGraph<{ message_id?: string; recipient_id?: string }>(
          `/${creds.pageId}/messages`,
          {
            accessToken: creds.accessToken,
            method: 'POST',
            body,
            appsecretProof: creds.appsecretProof,
          },
        );
        return { externalMessageId: res.message_id };
      },
    );
  }

  private async creds() {
    const t = await this.resolveToken();
    const meta = (t.metadata ?? {}) as MessengerMetadata;
    if (!meta.pageId) throw new Error('messenger_pageid_missing');
    const appSecret = meta.appSecret ?? process.env.META_APP_SECRET;
    return {
      creds: {
        accessToken: t.token,
        pageId: meta.pageId,
        appsecretProof: appSecret ? computeAppsecretProof(t.token, appSecret) : undefined,
      },
    };
  }
}
