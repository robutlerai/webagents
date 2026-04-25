/**
 * TwilioSkill — SMS / MMS via Twilio Messages API.
 *
 * Token resolution: ResolvedToken.token = AUTH_TOKEN; metadata carries
 * `accountSid` (or `subaccountSid`) and a default `fromNumber` /
 * `messagingServiceSid`. When the host has registered an A2P 10DLC
 * campaign for the sender (US 10DLC compliance), it should set
 * `metadata.a2pCampaignRegistered = true`; otherwise this skill refuses
 * +1 sends to keep operators out of carrier-blocked territory.
 *
 * Status callbacks (delivery / failed) hit the `/status` @http endpoint;
 * the portal wires that to its bridged-message storage to surface delivery
 * states in the chat UI.
 *
 * Reference: https://www.twilio.com/docs/messaging/api
 */
import { http, prompt, tool } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  MessagingSkill,
  verifyTwilioSignature,
  type MessagingSkillOptions,
} from '../shared';

const PROVIDER = 'twilio';
const SEND_TOOL = 'twilio_send_sms';

interface TwilioMetadata {
  accountSid?: string;
  subaccountSid?: string;
  fromNumber?: string;
  messagingServiceSid?: string;
  a2pCampaignRegistered?: boolean;
}

export class TwilioSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('twilio', opts);
  }

  @tool({
    name: SEND_TOOL,
    description:
      'Send an SMS from the connected Twilio number. THIS IS THE ONLY WAY ' +
      'TO REPLY TO A BRIDGED SMS CONTACT — a plain assistant message stays ' +
      'in the portal and never reaches the recipient.',
    parameters: {
      type: 'object',
      properties: {
        to: { type: 'string', description: 'E.164. Optional when responding to a bridged contact.' },
        body: { type: 'string' },
        from: { type: 'string' },
      },
      required: ['body'],
    },
  })
  async sendSms(args: { to?: string; body: string; from?: string }, ctx?: Context) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    return this.send(args, ctx, false);
  }

  @tool({
    name: 'twilio_send_mms',
    description:
      'Send an MMS (image / media) from the connected Twilio number. ' +
      'Pass either `content_id` (the Robutler UUID of an internally-hosted ' +
      'attachment) or `media_url` (a fully-qualified https URL Twilio can ' +
      'fetch). Twilio downloads the URL itself — relative `/api/content/...` ' +
      'paths are not accepted.',
    parameters: {
      type: 'object',
      properties: {
        to: { type: 'string' },
        body: { type: 'string' },
        content_id: {
          type: 'string',
          description: 'Robutler content UUID for internally-hosted media.',
        },
        media_url: {
          type: 'string',
          description: 'Absolute https URL to media — must be reachable by Twilio.',
        },
        from: { type: 'string' },
      },
      required: [],
    },
  })
  async sendMms(
    args: {
      to?: string;
      body?: string;
      content_id?: string;
      media_url?: string;
      from?: string;
    },
    ctx?: Context,
  ) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const resolved = await this.resolveOutboundMedia({
      contentId: args.content_id,
      url: args.media_url,
    });
    if ('error' in resolved) return resolved.error;
    if (!/^https?:\/\//i.test(resolved.media.url)) {
      return this.invalidInput(
        'Twilio MMS requires a publicly-reachable https URL. The resolved content does not expose one — host must surface an absolute URL for outbound media.',
        'twilio_requires_public_url',
      );
    }
    return this.send(
      { to: args.to, body: args.body ?? '', from: args.from },
      ctx,
      true,
      resolved.media.url,
    );
  }

  @prompt({ name: 'twilio-bridge-awareness' })
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
   * Twilio status callback (delivered / failed / undelivered). Hosts wire
   * this URL on each Messaging Service or per-number setting; the portal
   * dispatcher surfaces it as `/api/agents/{id}/messaging/twilio/status`.
   */
  @http({
    path: '/messaging/twilio/status',
    method: 'POST',
    auth: 'signature',
    description: 'Twilio status callback (delivered/failed) — verifies X-Twilio-Signature.',
  })
  async statusCallback(req: Request): Promise<Response> {
    const t = await this.resolveToken();
    const rawBody = await req.text();
    const params = Object.fromEntries(new URLSearchParams(rawBody)) as Record<string, string>;
    const url = req.headers.get('X-Forwarded-Url') ?? req.url;
    const signature = req.headers.get('X-Twilio-Signature') ?? '';
    const ok = verifyTwilioSignature({
      authToken: t.token,
      url,
      params,
      signature,
    });
    if (!ok) return new Response('forbidden', { status: 403 });
    // No host-level persistence in the skill; emit a structured log so the
    // portal's status mirror picks it up (it tails skill stdout in dev).
    console.log(
      `[twilio-status] sid=${params.MessageSid} status=${params.MessageStatus} to=${params.To}`,
    );
    return new Response('ok', { status: 200 });
  }

  // -------------------------------------------------------------------------

  private async send(
    args: { to?: string; body: string; from?: string },
    ctx: Context | undefined,
    mms: boolean,
    mediaUrl?: string,
  ) {
    const to = args.to ?? this.bridgeRecipient(ctx);
    if (!to) return this.invalidInput('Recipient required');
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: mms ? 'send_mms' : 'send_sms',
        agentId: this.agentId,
        integrationId: this.integrationId,
        idempotencyKey: this.integrationId
          ? `${this.integrationId}:${to}:${args.body.slice(0, 64)}`
          : undefined,
      },
      async () => {
        const t = await this.resolveToken();
        const meta = (t.metadata ?? {}) as TwilioMetadata;
        if (to.startsWith('+1') && !meta.a2pCampaignRegistered) {
          throw new Error('a2p_campaign_not_registered');
        }
        const sid = meta.subaccountSid ?? meta.accountSid;
        if (!sid) throw new Error('twilio_account_sid_missing');
        const from = args.from ?? meta.fromNumber;
        if (!from && !meta.messagingServiceSid) {
          throw new Error('twilio_from_or_service_required');
        }
        const auth = 'Basic ' + Buffer.from(`${sid}:${t.token}`).toString('base64');
        const body = new URLSearchParams({ To: to, Body: args.body });
        if (from) body.set('From', from);
        else if (meta.messagingServiceSid) body.set('MessagingServiceSid', meta.messagingServiceSid);
        if (mediaUrl) body.set('MediaUrl', mediaUrl);
        const r = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${sid}/Messages.json`, {
          method: 'POST',
          headers: { Authorization: auth, 'Content-Type': 'application/x-www-form-urlencoded' },
          body: body.toString(),
        });
        const j = (await r.json()) as { sid?: string; message?: string; code?: number };
        if (!r.ok) {
          const e = new Error(j.message ?? `twilio ${r.status}`) as Error & {
            status?: number;
            code?: number;
          };
          e.status = r.status;
          e.code = j.code;
          throw e;
        }
        return { externalMessageId: j.sid };
      },
    );
  }
}
