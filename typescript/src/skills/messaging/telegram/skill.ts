/**
 * TelegramSkill — bot-token messaging via the Bot API.
 *
 * Hosts wire `getToken` to their credential store; standalone hosts get the
 * token from `TELEGRAM_BOT_TOKEN`. The `setWebhook` @http callback is for
 * non-portal hosts that want a per-agent webhook URL; the portal exposes
 * `/api/webhooks/telegram` as a single-tenant endpoint instead.
 *
 * Bot API reference: https://core.telegram.org/bots/api (Mar 2026)
 */
import { http, prompt, tool } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  MessagingSkill,
  type MessagingSkillOptions,
} from '../shared';

const PROVIDER = 'telegram';
const SEND_TOOL = 'telegram_send_text';

export class TelegramSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('telegram', opts);
  }

  // -------------------------------------------------------------------------
  // Tools
  // -------------------------------------------------------------------------

  @tool({
    name: SEND_TOOL,
    description:
      'Send a text message via the connected Telegram bot. ' +
      'THIS IS THE ONLY WAY TO REPLY TO A TELEGRAM CONTACT — a plain ' +
      'assistant message stays in the portal and never reaches the user. ' +
      'Use `reply_to_message_id` to thread the reply against an existing ' +
      'message. With `parse_mode=MarkdownV2` you must escape _, *, [, ], ' +
      "(, ), ~, `, >, #, +, -, =, |, {, }, ., ! per Telegram's formatting rules.",
    parameters: {
      type: 'object',
      properties: {
        chat_id: {
          type: 'string',
          description:
            'Telegram chat id. Optional when responding to a bridged message — defaults to the bridged contact.',
        },
        text: { type: 'string' },
        parse_mode: { type: 'string', enum: ['HTML', 'MarkdownV2'] },
        reply_to_message_id: { type: 'number' },
        disable_web_page_preview: { type: 'boolean' },
      },
      required: ['text'],
    },
  })
  async sendText(
    args: {
      chat_id?: string;
      text: string;
      parse_mode?: 'HTML' | 'MarkdownV2';
      reply_to_message_id?: number;
      disable_web_page_preview?: boolean;
    },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const chatId = args.chat_id ?? this.bridgeRecipient(ctx);
    if (!chatId) return this.invalidInput('chat_id required');
    return this.tgCall('sendMessage', 'send_text', {
      chat_id: chatId,
      text: args.text,
      parse_mode: args.parse_mode,
      ...(typeof args.reply_to_message_id === 'number'
        ? { reply_parameters: { message_id: args.reply_to_message_id } }
        : {}),
      ...(args.disable_web_page_preview === true
        ? { link_preview_options: { is_disabled: true } }
        : {}),
    });
  }

  @tool({
    name: 'telegram_send_photo',
    description:
      'Send a photo via the connected Telegram bot. Use this (not a plain ' +
      'assistant message) when replying with an image to a Telegram contact. ' +
      'Pass `content_id` (the UUID returned by Robutler tools like `present` or ' +
      'media-generation) for any internally-hosted media — the skill resolves ' +
      'it to a signed URL Telegram can fetch, with an automatic upload-bytes ' +
      'fallback. Use `photo_url` only for fully-qualified external https URLs.',
    parameters: {
      type: 'object',
      properties: {
        chat_id: { type: 'string' },
        content_id: {
          type: 'string',
          description:
            'Robutler content UUID (preferred for any internally-hosted image). The skill resolves it server-side to a Telegram-fetchable URL or uploaded bytes.',
        },
        photo_url: {
          type: 'string',
          description:
            'Fully-qualified external https URL. Do NOT pass relative `/api/content/...` paths — use `content_id` instead.',
        },
        caption: { type: 'string' },
      },
      required: [],
    },
  })
  async sendPhoto(
    args: { chat_id?: string; content_id?: string; photo_url?: string; caption?: string },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const chatId = args.chat_id ?? this.bridgeRecipient(ctx);
    if (!chatId) return this.invalidInput('chat_id required');
    const resolved = await this.resolveOutboundMedia({ contentId: args.content_id, url: args.photo_url });
    if ('error' in resolved) return resolved.error;
    const fields = stripUndefined({ chat_id: chatId, caption: args.caption });
    return this.sendMediaWithFallback({
      callType: 'send_photo',
      media: resolved.media,
      sendByUrl: (url) => this.tgCall('sendPhoto', 'send_photo', { ...fields, photo: url }),
      sendByBytes: (bytes) =>
        this.tgUpload('sendPhoto', 'send_photo', fields, {
          field: 'photo',
          buffer: bytes.buffer,
          contentType: bytes.contentType,
          filename: bytes.filename ?? 'photo',
        }),
    });
  }

  @tool({
    name: 'telegram_send_document',
    description:
      'Send a document via the connected Telegram bot. Use this (not a plain ' +
      'assistant message) when sending a file to a Telegram contact. Pass ' +
      '`content_id` (Robutler UUID) for internally-hosted files — the skill ' +
      'resolves it to a signed URL with an automatic upload-bytes fallback. ' +
      'Use `document_url` only for fully-qualified external https URLs.',
    parameters: {
      type: 'object',
      properties: {
        chat_id: { type: 'string' },
        content_id: {
          type: 'string',
          description:
            'Robutler content UUID (preferred for any internally-hosted file).',
        },
        document_url: {
          type: 'string',
          description:
            'Fully-qualified external https URL. Do NOT pass relative `/api/content/...` paths — use `content_id` instead.',
        },
        caption: { type: 'string' },
      },
      required: [],
    },
  })
  async sendDocument(
    args: { chat_id?: string; content_id?: string; document_url?: string; caption?: string },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const chatId = args.chat_id ?? this.bridgeRecipient(ctx);
    if (!chatId) return this.invalidInput('chat_id required');
    const resolved = await this.resolveOutboundMedia({ contentId: args.content_id, url: args.document_url });
    if ('error' in resolved) return resolved.error;
    const fields = stripUndefined({ chat_id: chatId, caption: args.caption });
    return this.sendMediaWithFallback({
      callType: 'send_document',
      media: resolved.media,
      sendByUrl: (url) => this.tgCall('sendDocument', 'send_document', { ...fields, document: url }),
      sendByBytes: (bytes) =>
        this.tgUpload('sendDocument', 'send_document', fields, {
          field: 'document',
          buffer: bytes.buffer,
          contentType: bytes.contentType,
          filename: bytes.filename ?? 'file',
        }),
    });
  }

  @tool({
    name: 'telegram_send_typing',
    description: 'Show the typing indicator in the given Telegram chat for ~5 seconds.',
    parameters: {
      type: 'object',
      properties: { chat_id: { type: 'string' } },
      required: [],
    },
  })
  async sendTyping(args: { chat_id?: string }, ctx?: Context) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const chatId = args.chat_id ?? this.bridgeRecipient(ctx);
    if (!chatId) return this.invalidInput('chat_id required');
    return this.tgCall('sendChatAction', 'send_typing', { chat_id: chatId, action: 'typing' });
  }

  // -------------------------------------------------------------------------
  // Prompts (bridge-aware)
  // -------------------------------------------------------------------------

  @prompt({
    name: 'telegram-bridge-awareness',
    description: 'Adds a bridge-context note when the agent is replying to a Telegram contact.',
  })
  async bridgePrompt(ctx?: Context): Promise<string | null> {
    const b = bridgeMatches(ctx, PROVIDER);
    if (!b) return null;
    return buildBridgeAwarenessPrompt({
      provider: PROVIDER,
      contactDisplayName: b.contactDisplayName,
      sendToolName: SEND_TOOL,
    });
  }

  // -------------------------------------------------------------------------
  // HTTP — per-agent webhook setup helper for non-portal hosts.
  // -------------------------------------------------------------------------

  @http({
    path: '/messaging/telegram/set-webhook',
    method: 'POST',
    auth: 'session',
    description:
      'Register the Telegram webhook to point at this agent. Standalone hosts ' +
      'use this to wire BotFather → their daemon without hand-running curl.',
  })
  async setWebhook(_req: Request): Promise<Response> {
    try {
      const t = await this.resolveToken();
      const url = `${this.httpBaseUrl}/messaging/telegram/inbound`;
      const r = await fetch(`https://api.telegram.org/bot${t.token}/setWebhook`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url,
          allowed_updates: ['message', 'edited_message', 'callback_query'],
          drop_pending_updates: true,
        }),
      });
      const j = (await r.json()) as { ok: boolean; description?: string };
      return new Response(JSON.stringify({ ok: j.ok, description: j.description, webhookUrl: url }), {
        status: j.ok ? 200 : 502,
        headers: { 'Content-Type': 'application/json' },
      });
    } catch (e) {
      return new Response(JSON.stringify({ ok: false, error: (e as Error).message }), { status: 500 });
    }
  }

  // -------------------------------------------------------------------------
  // Internal — Telegram-specific HTTP wrappers
  // -------------------------------------------------------------------------

  private async tgCall(path: string, callType: string, body: Record<string, unknown>) {
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: callType,
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        const r = await fetch(`https://api.telegram.org/bot${t.token}/${path}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const j = (await r.json()) as {
          ok: boolean;
          result?: unknown;
          description?: string;
          error_code?: number;
        };
        if (!r.ok || !j.ok) {
          const err = new Error(j.description ?? `telegram ${path} ${r.status}`) as Error & {
            status?: number;
            code?: number;
          };
          err.status = r.status;
          err.code = j.error_code;
          throw err;
        }
        return j.result;
      },
    );
  }

  /**
   * Multipart variant of {@link tgCall}. Used by the bytes-fallback path
   * when Telegram refuses to fetch the URL we handed it. Browser/Node
   * `FormData` is available in all modern runtimes (Node ≥ 18, edge,
   * Bun), no extra dependency.
   */
  private async tgUpload(
    path: string,
    callType: string,
    fields: Record<string, unknown>,
    file: { field: string; buffer: Uint8Array; contentType: string; filename: string },
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
        const form = new FormData();
        for (const [k, v] of Object.entries(fields)) {
          if (v == null) continue;
          form.append(k, typeof v === 'string' ? v : String(v));
        }
        // Cast: TS narrows `Uint8Array<ArrayBufferLike>` (which includes
        // `SharedArrayBuffer`) but `Blob` only accepts `ArrayBuffer`-backed
        // views. Our resolver always produces a regular Uint8Array so
        // the runtime check is unnecessary.
        const blob = new Blob([file.buffer as BlobPart], { type: file.contentType });
        form.append(file.field, blob, file.filename);
        const r = await fetch(`https://api.telegram.org/bot${t.token}/${path}`, {
          method: 'POST',
          body: form,
        });
        const j = (await r.json()) as {
          ok: boolean;
          result?: unknown;
          description?: string;
          error_code?: number;
        };
        if (!r.ok || !j.ok) {
          const err = new Error(j.description ?? `telegram ${path} ${r.status}`) as Error & {
            status?: number;
            code?: number;
          };
          err.status = r.status;
          err.code = j.error_code;
          throw err;
        }
        return j.result;
      },
    );
  }
}

function stripUndefined(o: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(o)) if (v !== undefined) out[k] = v;
  return out;
}
