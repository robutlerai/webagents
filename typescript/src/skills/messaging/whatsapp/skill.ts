/**
 * WhatsAppSkill — Cloud API messaging via Meta Graph.
 *
 * Token resolution: ResolvedToken.token = WhatsApp permanent system user
 * token (or a long-lived user token from Embedded Signup). Metadata MUST
 * carry `phoneNumberId` and `wabaId`; an optional `appSecret` enables
 * `appsecret_proof` signing.
 *
 * The 24h customer-service window is enforced by the host (chat metadata
 * carries `lastUserMessageAt`). Skills surface `whatsapp_send_template`
 * so the agent can step outside the window when the host signals
 * `outside_window`.
 *
 * Reference: https://developers.facebook.com/docs/whatsapp/cloud-api
 */
import { prompt, tool } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  computeAppsecretProof,
  MessagingSkill,
  metaGraph,
  metaGraphUrl,
  type MessagingSkillOptions,
} from '../shared';

const PROVIDER = 'whatsapp';
const SEND_TOOL = 'whatsapp_send_text';

interface WhatsAppMetadata {
  phoneNumberId?: string;
  wabaId?: string;
  appSecret?: string;
}

export class WhatsAppSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('whatsapp', opts);
  }

  @tool({
    name: SEND_TOOL,
    description:
      'Send a WhatsApp text message in reply to an open conversation. Only allowed inside ' +
      'the 24h customer-service window — use whatsapp_send_template otherwise.',
    parameters: {
      type: 'object',
      properties: {
        to: { type: 'string', description: 'Recipient WA id (E.164, no plus). Optional when bridged.' },
        text: { type: 'string' },
      },
      required: ['text'],
    },
  })
  async sendText(args: { to?: string; text: string }, ctx?: Context) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const to = args.to ?? this.bridgeRecipient(ctx);
    if (!to) return this.invalidInput('Recipient required');
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'send_text',
        agentId: this.agentId,
        integrationId: this.integrationId,
        windowCheck: { contactExternalId: to },
      },
      async () => {
        const { creds } = await this.creds();
        const res = await metaGraph<{ messages?: Array<{ id: string }> }>(
          `/${creds.phoneNumberId}/messages`,
          {
            accessToken: creds.accessToken,
            appsecretProof: creds.appsecretProof,
            method: 'POST',
            body: {
              messaging_product: 'whatsapp',
              recipient_type: 'individual',
              to,
              type: 'text',
              text: { body: args.text },
            },
          },
        );
        return { externalMessageId: res.messages?.[0]?.id };
      },
    );
  }

  @tool({
    name: 'whatsapp_send_image',
    description:
      'Send an image message in an open WhatsApp conversation (24h ' +
      'customer-service window). Pass either `content_id` (Robutler UUID) ' +
      'or `image_url` (absolute https URL Meta can fetch). If Meta refuses ' +
      'to fetch the URL, the skill automatically falls back to uploading ' +
      'the bytes via `/${phoneNumberId}/media` and re-sending with the ' +
      'returned media id.',
    parameters: {
      type: 'object',
      properties: {
        to: { type: 'string' },
        content_id: { type: 'string' },
        image_url: { type: 'string' },
        caption: { type: 'string' },
      },
      required: [],
    },
  })
  async sendImage(
    args: {
      to?: string;
      content_id?: string;
      image_url?: string;
      caption?: string;
    },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    return this.sendMedia({ ...args, kind: 'image' }, ctx);
  }

  @tool({
    name: 'whatsapp_send_document',
    description:
      'Send a document (pdf, etc.) in an open WhatsApp conversation. Pass ' +
      'either `content_id` or absolute https `document_url`. Same ' +
      'url-first / multipart-upload fallback as `whatsapp_send_image`.',
    parameters: {
      type: 'object',
      properties: {
        to: { type: 'string' },
        content_id: { type: 'string' },
        document_url: { type: 'string' },
        filename: { type: 'string' },
        caption: { type: 'string' },
      },
      required: [],
    },
  })
  async sendDocument(
    args: {
      to?: string;
      content_id?: string;
      document_url?: string;
      filename?: string;
      caption?: string;
    },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    return this.sendMedia(
      {
        to: args.to,
        content_id: args.content_id,
        image_url: args.document_url,
        caption: args.caption,
        filename: args.filename,
        kind: 'document',
      },
      ctx,
    );
  }

  @tool({
    name: 'whatsapp_send_template',
    description:
      'Send an approved WhatsApp Business template (allowed outside the 24h window). ' +
      'Variables map to body parameters in order.',
    parameters: {
      type: 'object',
      properties: {
        to: { type: 'string' },
        template_name: { type: 'string' },
        language_code: { type: 'string', default: 'en_US' },
        variables: { type: 'array', items: { type: 'string' } },
      },
      required: ['to', 'template_name'],
    },
  })
  async sendTemplate(args: {
    to: string;
    template_name: string;
    language_code?: string;
    variables?: string[];
  }) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'send_template',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const { creds } = await this.creds();
        const components = args.variables?.length
          ? [{ type: 'body', parameters: args.variables.map((t) => ({ type: 'text', text: t })) }]
          : undefined;
        const res = await metaGraph<{ messages?: Array<{ id: string }> }>(
          `/${creds.phoneNumberId}/messages`,
          {
            accessToken: creds.accessToken,
            appsecretProof: creds.appsecretProof,
            method: 'POST',
            body: {
              messaging_product: 'whatsapp',
              to: args.to,
              type: 'template',
              template: {
                name: args.template_name,
                language: { code: args.language_code ?? 'en_US' },
                ...(components ? { components } : {}),
              },
            },
          },
        );
        return { externalMessageId: res.messages?.[0]?.id };
      },
    );
  }

  @tool({
    name: 'whatsapp_list_templates',
    description: "List the WABA's approved message templates with variable counts.",
    parameters: { type: 'object', properties: {}, required: [] },
  })
  async listTemplates() {
    if (!this.capabilityEnabled('manage_templates')) return this.capabilityDisabled('manage_templates');
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'list_templates',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const { creds } = await this.creds();
        return metaGraph<{
          data: Array<{ name: string; language: string; status: string; components?: unknown[] }>;
        }>(`/${creds.wabaId}/message_templates?limit=100`, {
          accessToken: creds.accessToken,
          appsecretProof: creds.appsecretProof,
        });
      },
    );
  }

  @prompt({ name: 'whatsapp-bridge-awareness' })
  async bridgePrompt(ctx?: Context): Promise<string | null> {
    const b = bridgeMatches(ctx, PROVIDER);
    if (!b) return null;
    return buildBridgeAwarenessPrompt({
      provider: PROVIDER,
      contactDisplayName: b.contactDisplayName,
      sendToolName: SEND_TOOL,
    });
  }

  private async sendMedia(
    args: {
      to?: string;
      content_id?: string;
      image_url?: string;
      caption?: string;
      filename?: string;
      kind: 'image' | 'document';
    },
    ctx: Context | undefined,
  ) {
    const to = args.to ?? this.bridgeRecipient(ctx);
    if (!to) return this.invalidInput('Recipient required');
    const resolved = await this.resolveOutboundMedia({
      contentId: args.content_id,
      url: args.image_url,
    });
    if ('error' in resolved) return resolved.error;

    return this.sendMediaWithFallback<{ externalMessageId?: string }>({
      callType: args.kind === 'image' ? 'send_image' : 'send_document',
      media: resolved.media,
      sendByUrl: async (url) =>
        this.wrapApiCall(
          {
            provider: PROVIDER,
            type: args.kind === 'image' ? 'send_image' : 'send_document',
            agentId: this.agentId,
            integrationId: this.integrationId,
            windowCheck: { contactExternalId: to },
          },
          async () => {
            const { creds } = await this.creds();
            const payload =
              args.kind === 'image'
                ? { image: { link: url, ...(args.caption ? { caption: args.caption } : {}) } }
                : {
                    document: {
                      link: url,
                      ...(args.filename ? { filename: args.filename } : {}),
                      ...(args.caption ? { caption: args.caption } : {}),
                    },
                  };
            const res = await metaGraph<{ messages?: Array<{ id: string }> }>(
              `/${creds.phoneNumberId}/messages`,
              {
                accessToken: creds.accessToken,
                appsecretProof: creds.appsecretProof,
                method: 'POST',
                body: {
                  messaging_product: 'whatsapp',
                  recipient_type: 'individual',
                  to,
                  type: args.kind,
                  ...payload,
                },
              },
            );
            return { externalMessageId: res.messages?.[0]?.id };
          },
        ),
      sendByBytes: async (bytes) =>
        this.wrapApiCall(
          {
            provider: PROVIDER,
            type: args.kind === 'image' ? 'send_image_bytes' : 'send_document_bytes',
            agentId: this.agentId,
            integrationId: this.integrationId,
            windowCheck: { contactExternalId: to },
          },
          async () => {
            const { creds } = await this.creds();
            const mediaId = await this.uploadMedia({
              creds,
              bytes,
              filename: args.filename ?? bytes.filename,
            });
            const payload =
              args.kind === 'image'
                ? { image: { id: mediaId, ...(args.caption ? { caption: args.caption } : {}) } }
                : {
                    document: {
                      id: mediaId,
                      ...(args.filename ? { filename: args.filename } : {}),
                      ...(args.caption ? { caption: args.caption } : {}),
                    },
                  };
            const res = await metaGraph<{ messages?: Array<{ id: string }> }>(
              `/${creds.phoneNumberId}/messages`,
              {
                accessToken: creds.accessToken,
                appsecretProof: creds.appsecretProof,
                method: 'POST',
                body: {
                  messaging_product: 'whatsapp',
                  recipient_type: 'individual',
                  to,
                  type: args.kind,
                  ...payload,
                },
              },
            );
            return { externalMessageId: res.messages?.[0]?.id };
          },
        ),
    });
  }

  private async uploadMedia(input: {
    creds: { accessToken: string; phoneNumberId: string; appsecretProof?: string };
    bytes: { buffer: Uint8Array; contentType: string; filename?: string };
    filename?: string;
  }): Promise<string> {
    const { creds, bytes, filename } = input;
    const url = new URL(metaGraphUrl(`/${creds.phoneNumberId}/media`));
    url.searchParams.set('access_token', creds.accessToken);
    if (creds.appsecretProof) url.searchParams.set('appsecret_proof', creds.appsecretProof);
    const form = new FormData();
    form.set('messaging_product', 'whatsapp');
    form.set('type', bytes.contentType);
    form.set(
      'file',
      new Blob([bytes.buffer as BlobPart], { type: bytes.contentType }),
      filename ?? `upload.${guessExt(bytes.contentType)}`,
    );
    const r = await fetch(url.toString(), { method: 'POST', body: form });
    const text = await r.text();
    if (!r.ok) {
      const e = new Error(text.slice(0, 300)) as Error & { status?: number };
      e.status = r.status;
      throw e;
    }
    const parsed = JSON.parse(text) as { id?: string };
    if (!parsed.id) throw new Error('whatsapp_media_upload_no_id');
    return parsed.id;
  }

  private async creds() {
    const t = await this.resolveToken();
    const meta = (t.metadata ?? {}) as WhatsAppMetadata;
    if (!meta.phoneNumberId || !meta.wabaId) {
      throw new Error('whatsapp_metadata_missing');
    }
    const appSecret = meta.appSecret ?? process.env.META_APP_SECRET;
    return {
      creds: {
        accessToken: t.token,
        phoneNumberId: meta.phoneNumberId,
        wabaId: meta.wabaId,
        appsecretProof: appSecret ? computeAppsecretProof(t.token, appSecret) : undefined,
      },
    };
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
