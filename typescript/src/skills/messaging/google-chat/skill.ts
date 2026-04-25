/**
 * GoogleChatSkill - Google Workspace Chat App.
 *
 * Auth model:
 *   - Workspace bot (separate provider entry from `google` so personal Google
 *     identity rows are unaffected). OAuth2 with `chat.bot` +
 *     `chat.spaces.readonly`. Workspace domain is captured at OAuth callback
 *     time from the `hd` claim and stored on connected_account.metadata.
 *
 * Tools:
 *   - `google_chat_send_message` — text message in a space.
 *   - `google_chat_send_card` — Card v2 message (rich UI).
 *   - `google_chat_list_spaces` — spaces the bot is a member of.
 *
 * HTTP endpoint:
 *   - `POST /messaging/google-chat/event` — JWT-verified Chat App event
 *     (MESSAGE / ADDED_TO_SPACE / REMOVED_FROM_SPACE / CARD_CLICKED).
 *
 * References:
 *   https://developers.google.com/chat/api/guides/v1/messages/create
 *   https://developers.google.com/chat/api/guides/v1/spaces/list
 *   https://developers.google.com/chat/how-tos/bots-develop#verify_request
 */
import { http, prompt, tool } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  MessagingSkill,
  verifyGoogleChatJwt,
  type MessagingSkillOptions,
} from '../shared';

const PROVIDER = 'google-chat';
const SEND_TOOL = 'google_chat_send_message';
const SEND_CARD_TOOL = 'google_chat_send_card';
const LIST_SPACES_TOOL = 'google_chat_list_spaces';

const CHAT_API = 'https://chat.googleapis.com/v1';

interface GoogleChatMetadata {
  workspaceDomain?: string;
  /** Project number / id used as the JWT `aud` claim by Google Chat. */
  expectedAudience?: string;
}

export class GoogleChatSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('google-chat', opts);
  }

  // ---- Outbound tools --------------------------------------------------

  @tool({
    name: SEND_TOOL,
    description:
      'Send a plain-text message into a Google Chat space. THIS IS THE ONLY ' +
      'WAY TO REPLY TO A BRIDGED GOOGLE CHAT MESSAGE.',
    parameters: {
      type: 'object',
      properties: {
        space: {
          type: 'string',
          description: 'Space resource name (e.g. "spaces/AAAAxxx").',
        },
        thread: {
          type: 'string',
          description: 'Thread resource name (e.g. "spaces/AAAAxxx/threads/yyy") to reply within.',
        },
        text: { type: 'string' },
      },
      required: ['text'],
    },
  })
  async sendMessage(
    args: { space?: string; thread?: string; text: string },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const space = args.space ?? this.bridgeRecipient(ctx);
    if (!space) return this.invalidInput('space required');
    if (!args.text?.trim()) return this.invalidInput('text required');
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'send_message',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        const body: Record<string, unknown> = { text: args.text };
        if (args.thread) body.thread = { name: args.thread };
        const r = await chatFetch<{ name?: string }>(
          `${CHAT_API}/${space}/messages`,
          t.token,
          'POST',
          body,
        );
        return { externalMessageId: r.name };
      },
    );
  }

  @tool({
    name: SEND_CARD_TOOL,
    description:
      'Send a Card v2 message into a Google Chat space. Supply the cardsV2 array per the Chat REST contract.',
    parameters: {
      type: 'object',
      properties: {
        space: { type: 'string' },
        thread: { type: 'string' },
        cardsV2: { type: 'array', items: { type: 'object' } },
        text: { type: 'string', description: 'Optional fallback text shown in notifications.' },
      },
      required: ['space', 'cardsV2'],
    },
  })
  async sendCard(args: { space: string; thread?: string; cardsV2: unknown[]; text?: string }) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    if (!args.space) return this.invalidInput('space required');
    if (!Array.isArray(args.cardsV2) || args.cardsV2.length === 0) {
      return this.invalidInput('cardsV2 required');
    }
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'send_card',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        const body: Record<string, unknown> = { cardsV2: args.cardsV2 };
        if (args.text) body.text = args.text;
        if (args.thread) body.thread = { name: args.thread };
        const r = await chatFetch<{ name?: string }>(
          `${CHAT_API}/${args.space}/messages`,
          t.token,
          'POST',
          body,
        );
        return { externalMessageId: r.name };
      },
    );
  }

  @tool({
    name: LIST_SPACES_TOOL,
    description: 'List Google Chat spaces (rooms + DMs) the bot is a member of.',
    parameters: {
      type: 'object',
      properties: {
        pageSize: { type: 'integer' },
        pageToken: { type: 'string' },
      },
    },
  })
  async listSpaces(args: { pageSize?: number; pageToken?: string }) {
    if (!this.capabilityEnabled('list_spaces')) return this.capabilityDisabled('list_spaces');
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'list_spaces',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        const params = new URLSearchParams();
        if (args.pageSize) params.set('pageSize', String(args.pageSize));
        if (args.pageToken) params.set('pageToken', args.pageToken);
        const url = `${CHAT_API}/spaces${params.size ? `?${params.toString()}` : ''}`;
        return await chatFetch<{ spaces?: unknown[]; nextPageToken?: string }>(url, t.token, 'GET');
      },
    );
  }

  // ---- Webhook (event) endpoint ----------------------------------------

  @http({
    path: '/messaging/google-chat/event',
    method: 'POST',
    auth: 'public',
    description:
      'Google Chat App event endpoint (JWT-verified). Logs the event for the host bridge.',
  })
  async event(req: Request): Promise<Response> {
    const t = await this.resolveToken().catch(() => null);
    const meta = (t?.metadata ?? {}) as GoogleChatMetadata;
    const expectedAudience =
      meta.expectedAudience ??
      process.env.GOOGLE_CHAT_PROJECT_NUMBER ??
      process.env.GOOGLE_CHAT_AUDIENCE;
    if (!expectedAudience) return new Response('audience_unconfigured', { status: 503 });

    const auth = req.headers.get('authorization') ?? '';
    if (!auth.toLowerCase().startsWith('bearer ')) {
      return new Response('forbidden', { status: 403 });
    }
    const jwt = auth.slice('bearer '.length).trim();
    const ok = await verifyGoogleChatJwt({ jwt, expectedAudience });
    if (!ok) return new Response('forbidden', { status: 403 });

    const rawBody = await req.text();
    console.log(
      JSON.stringify({
        kind: 'google-chat.event',
        agentId: this.agentId,
        integrationId: this.integrationId,
        body: rawBody.slice(0, 4000),
      }),
    );
    return new Response('ok', { status: 200 });
  }

  @prompt({ name: 'google-chat-bridge-awareness' })
  async bridgePrompt(ctx?: Context): Promise<string | null> {
    const b = bridgeMatches(ctx, PROVIDER);
    if (!b) return null;
    return buildBridgeAwarenessPrompt({
      provider: PROVIDER,
      contactDisplayName: b.contactDisplayName,
      sendToolName: SEND_TOOL,
    });
  }
}

async function chatFetch<T>(
  url: string,
  token: string,
  method: 'GET' | 'POST',
  body?: unknown,
): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await r.text();
  if (!r.ok) {
    const err = new Error(text.slice(0, 300) || `google_chat ${r.status}`) as Error & {
      status?: number;
    };
    err.status = r.status;
    throw err;
  }
  return text ? (JSON.parse(text) as T) : ({} as T);
}
