/**
 * XSkill - Twitter/X DM bridge + tweet publishing.
 *
 * Two capability surfaces in one skill module:
 *   - DMs (require `dm.write`): inbound webhooks via Account Activity API,
 *     outbound `send_dm` / `send_dm_in_conversation` via the Twitter API v2.
 *   - Tweet publishing (require `tweet.write`, approval-gated through
 *     `PostApprovalGate.requestApproval`): `post_tweet`, `post_thread`,
 *     `delete_tweet`.
 *
 * Auth model:
 *   - User access tokens (OAuth 2.0 PKCE, `dm.read`/`dm.write`/`tweet.write`)
 *     for outbound API calls. Resolved via `MessagingSkill.resolveToken`.
 *   - App-level OAuth 1.0a consumer secret for webhook signature checks
 *     (`X-Twitter-Webhooks-Signature`). Surfaced through `metadata.consumerSecret`
 *     on the connected_account row by the portal connect callback. Hosts
 *     running standalone supply the same value via `X_CONSUMER_SECRET` env.
 *
 * Webhook endpoints are *only* registered when DM scopes are granted; pure
 * publish-only rows skip the inbound surface entirely.
 *
 * References:
 *   - DM: https://developer.twitter.com/en/docs/twitter-api/direct-messages/manage/api-reference
 *   - Tweets: https://developer.twitter.com/en/docs/twitter-api/tweets/manage-tweets/api-reference/post-tweets
 *   - Account Activity: https://developer.twitter.com/en/docs/twitter-api/enterprise/account-activity-api
 */
import { http, prompt, tool } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  buildXCrcResponse,
  MessagingSkill,
  verifyXWebhookSignature,
  type MessagingSkillOptions,
} from '../shared';

const PROVIDER = 'x';
const DM_TOOL = 'x_send_dm';
const POST_TWEET = 'x_post_tweet';
const POST_THREAD = 'x_post_thread';
const DELETE_TWEET = 'x_delete_tweet';

const X_API = 'https://api.twitter.com';

interface XMetadata {
  consumerSecret?: string;
  /** App-level webhook id from /2/account_activity/webhooks. */
  webhookId?: string;
}

export class XSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('x', opts);
  }

  // ---- DM tools --------------------------------------------------------

  @tool({
    name: DM_TOOL,
    description:
      'Send an X DM to a participant. THIS IS THE ONLY WAY TO REPLY TO A ' +
      'BRIDGED X CONTACT - a plain assistant message stays in the portal.',
    parameters: {
      type: 'object',
      properties: {
        participant_id: { type: 'string', description: 'X user id (numeric).' },
        text: { type: 'string' },
      },
      required: ['text'],
    },
  })
  async sendDm(args: { participant_id?: string; text: string }, ctx?: Context) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const recipient = args.participant_id ?? this.bridgeRecipient(ctx);
    if (!recipient) return this.invalidInput('participant_id required');
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'send_dm',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        const r = await xFetch<{ data?: { dm_event_id?: string } }>(
          `${X_API}/2/dm_conversations/with/${encodeURIComponent(recipient)}/messages`,
          t.token,
          { text: args.text },
        );
        return { externalMessageId: r.data?.dm_event_id };
      },
    );
  }

  @tool({
    name: 'x_send_dm_in_conversation',
    description: 'Send an X DM into an existing dm_conversation_id.',
    parameters: {
      type: 'object',
      properties: {
        dm_conversation_id: { type: 'string' },
        text: { type: 'string' },
      },
      required: ['dm_conversation_id', 'text'],
    },
  })
  async sendDmInConversation(args: { dm_conversation_id: string; text: string }) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    if (!args.dm_conversation_id) return this.invalidInput('dm_conversation_id required');
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'send_dm_in_conversation',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        const r = await xFetch<{ data?: { dm_event_id?: string } }>(
          `${X_API}/2/dm_conversations/${encodeURIComponent(args.dm_conversation_id)}/messages`,
          t.token,
          { text: args.text },
        );
        return { externalMessageId: r.data?.dm_event_id };
      },
    );
  }

  // ---- Tweet publish tools (approval-gated) ----------------------------

  @tool({
    name: POST_TWEET,
    description:
      'Publish a single tweet. Subject to the host approval gate when configured.',
    parameters: {
      type: 'object',
      properties: {
        text: { type: 'string' },
        reply_to_tweet_id: { type: 'string' },
        media_ids: { type: 'array', items: { type: 'string' } },
      },
      required: ['text'],
    },
  })
  async postTweet(args: { text: string; reply_to_tweet_id?: string; media_ids?: string[] }) {
    if (!this.capabilityEnabled('publish_posts')) return this.capabilityDisabled('publish_posts');
    if (!args.text?.trim()) return this.invalidInput('text required');
    const gate = await this.maybeRequestApproval(POST_TWEET, args as Record<string, unknown>);
    if (gate) return { ok: true as const, data: gate };

    const idempotencyKey = this.integrationId
      ? `${this.integrationId}:tweet:${hashShort(args.text)}`
      : undefined;
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'post_tweet',
        agentId: this.agentId,
        integrationId: this.integrationId,
        idempotencyKey,
      },
      async () => {
        const t = await this.resolveToken();
        const id = await postOneTweet(t.token, args);
        return { externalMessageId: id };
      },
    );
  }

  @tool({
    name: POST_THREAD,
    description:
      'Publish a chained thread of tweets. Each tweet replies to the previous one. ' +
      'Subject to the host approval gate when configured.',
    parameters: {
      type: 'object',
      properties: {
        tweets: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              text: { type: 'string' },
              media_ids: { type: 'array', items: { type: 'string' } },
            },
            required: ['text'],
          },
        },
      },
      required: ['tweets'],
    },
  })
  async postThread(args: { tweets: Array<{ text: string; media_ids?: string[] }> }) {
    if (!this.capabilityEnabled('publish_posts')) return this.capabilityDisabled('publish_posts');
    if (!args.tweets?.length) return this.invalidInput('tweets required');
    const gate = await this.maybeRequestApproval(POST_THREAD, args as Record<string, unknown>);
    if (gate) return { ok: true as const, data: gate };

    const idempotencyKey = this.integrationId
      ? `${this.integrationId}:thread:${hashShort(args.tweets.map((t) => t.text).join('|'))}`
      : undefined;
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'post_thread',
        agentId: this.agentId,
        integrationId: this.integrationId,
        idempotencyKey,
      },
      async () => {
        const t = await this.resolveToken();
        const ids: string[] = [];
        let prev: string | undefined;
        for (const tw of args.tweets) {
          const id = await postOneTweet(t.token, {
            text: tw.text,
            media_ids: tw.media_ids,
            reply_to_tweet_id: prev,
          });
          if (id) ids.push(id);
          prev = id;
        }
        return { tweetIds: ids, externalMessageId: ids[0] };
      },
    );
  }

  @tool({
    name: DELETE_TWEET,
    description: 'Delete one of the connected user\'s tweets by id.',
    parameters: {
      type: 'object',
      properties: { tweet_id: { type: 'string' } },
      required: ['tweet_id'],
    },
  })
  async deleteTweet(args: { tweet_id: string }) {
    if (!this.capabilityEnabled('publish_posts')) return this.capabilityDisabled('publish_posts');
    if (!args.tweet_id) return this.invalidInput('tweet_id required');
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'delete_tweet',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        const r = await fetch(`${X_API}/2/tweets/${encodeURIComponent(args.tweet_id)}`, {
          method: 'DELETE',
          headers: { Authorization: `Bearer ${t.token}` },
        });
        if (!r.ok) {
          const text = await r.text();
          const e = new Error(text.slice(0, 300)) as Error & { status?: number };
          e.status = r.status;
          throw e;
        }
        return { deleted: true };
      },
    );
  }

  // ---- Webhook endpoints (DM only) --------------------------------------

  /**
   * X CRC challenge. X pings the webhook URL with `?crc_token=<token>` and
   * expects `{ "response_token": "sha256=<base64-hmac>" }` so it can
   * confirm the endpoint is owned by the app's consumer secret holder.
   */
  @http({
    path: '/messaging/x/webhook',
    method: 'GET',
    auth: 'public',
    description: 'X Account Activity CRC challenge handler.',
  })
  async crc(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const crcToken = url.searchParams.get('crc_token');
    if (!crcToken) return new Response('crc_token required', { status: 400 });
    const t = await this.resolveToken();
    const meta = (t.metadata ?? {}) as XMetadata;
    const consumerSecret = meta.consumerSecret ?? process.env.X_CONSUMER_SECRET;
    if (!consumerSecret) return new Response('consumer_secret_unavailable', { status: 500 });
    return Response.json({ response_token: buildXCrcResponse(consumerSecret, crcToken) });
  }

  /**
   * X Account Activity webhook. POSTs DM events; signature verified via
   * `X-Twitter-Webhooks-Signature`. Hosts implement the actual bridge
   * persistence; the SDK skill emits a structured log line so portal hosts
   * tail it.
   */
  @http({
    path: '/messaging/x/webhook',
    method: 'POST',
    auth: 'signature',
    description: 'X Account Activity webhook (DM events).',
  })
  async webhook(req: Request): Promise<Response> {
    const t = await this.resolveToken();
    const meta = (t.metadata ?? {}) as XMetadata;
    const consumerSecret = meta.consumerSecret ?? process.env.X_CONSUMER_SECRET;
    if (!consumerSecret) return new Response('consumer_secret_unavailable', { status: 500 });
    const rawBody = await req.text();
    const ok = verifyXWebhookSignature({
      consumerSecret,
      signatureHeader: req.headers.get('x-twitter-webhooks-signature') ?? '',
      rawBody,
    });
    if (!ok) return new Response('forbidden', { status: 403 });
    console.log(
      JSON.stringify({
        kind: 'x.account-activity',
        agentId: this.agentId,
        integrationId: this.integrationId,
        body: rawBody.slice(0, 4000),
      }),
    );
    return new Response('ok', { status: 200 });
  }

  @prompt({ name: 'x-bridge-awareness' })
  async bridgePrompt(ctx?: Context): Promise<string | null> {
    const b = bridgeMatches(ctx, PROVIDER);
    if (!b) return null;
    return buildBridgeAwarenessPrompt({
      provider: PROVIDER,
      contactDisplayName: b.contactDisplayName,
      sendToolName: DM_TOOL,
    });
  }
}

async function xFetch<T>(url: string, token: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  const text = await r.text();
  if (!r.ok) {
    const err = new Error(text.slice(0, 300) || `x ${r.status}`) as Error & { status?: number };
    err.status = r.status;
    throw err;
  }
  return text ? (JSON.parse(text) as T) : ({} as T);
}

async function postOneTweet(
  token: string,
  args: { text: string; reply_to_tweet_id?: string; media_ids?: string[] },
): Promise<string | undefined> {
  const body: Record<string, unknown> = { text: args.text };
  if (args.reply_to_tweet_id) {
    body.reply = { in_reply_to_tweet_id: args.reply_to_tweet_id };
  }
  if (args.media_ids?.length) {
    body.media = { media_ids: args.media_ids };
  }
  const r = await xFetch<{ data?: { id?: string } }>(`${X_API}/2/tweets`, token, body);
  return r.data?.id;
}

function hashShort(s: string): string {
  let h = 0;
  for (let i = 0; i < s.length; i += 1) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h).toString(36);
}
