/**
 * SendGridSkill - transactional email send + Inbound Parse webhook.
 *
 * Token resolution: ResolvedToken.token = SendGrid API key (`SG.…`). Metadata
 * MUST carry `fromEmail` (a verified sender). For Inbound Parse, an
 * additional `inboundDomain` (the MX-routed domain) and `inboundParseToken`
 * (per-account opaque secret embedded in the webhook URL) are stored on the
 * connected_accounts row by the portal's connect form.
 *
 * Outbound is two tools: a raw `send_email` (subject + html/text) and a
 * `send_template` that fills a SendGrid dynamic template by id. Both go to
 * `POST https://api.sendgrid.com/v3/mail/send`.
 *
 * Inbound is a single `@http POST /messaging/sendgrid/inbound-parse/:token`
 * endpoint. The route validates the token against the resolved
 * connected_accounts row, parses the multipart form-data body, and emits
 * a structured payload the host's bridge framework picks up.
 *
 * Reference: https://docs.sendgrid.com/api-reference/mail-send/mail-send
 */
import { http, prompt, tool } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  MessagingSkill,
  type MessagingSkillOptions,
} from '../shared';

const PROVIDER = 'sendgrid';
const SEND_TOOL = 'sendgrid_send_email';
const TEMPLATE_TOOL = 'sendgrid_send_template';

const SENDGRID_API = 'https://api.sendgrid.com';

interface SendGridMetadata {
  fromEmail?: string;
  fromName?: string;
  inboundDomain?: string;
  inboundParseToken?: string;
}

export class SendGridSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('sendgrid', opts);
  }

  @tool({
    name: SEND_TOOL,
    description:
      'Send a transactional email via SendGrid v3 mail/send. Uses the verified ' +
      "sender configured in the integration's metadata when `from` is omitted.",
    parameters: {
      type: 'object',
      properties: {
        to: { type: 'string', description: 'Recipient email address.' },
        subject: { type: 'string' },
        html: { type: 'string' },
        text: { type: 'string' },
        from: { type: 'string', description: 'Override the verified sender.' },
        replyTo: { type: 'string' },
        cc: { type: 'array', items: { type: 'string' } },
        bcc: { type: 'array', items: { type: 'string' } },
      },
      required: ['to', 'subject'],
    },
  })
  async sendEmail(args: {
    to: string;
    subject: string;
    html?: string;
    text?: string;
    from?: string;
    replyTo?: string;
    cc?: string[];
    bcc?: string[];
  }, ctx?: Context) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const to = args.to ?? this.bridgeRecipient(ctx);
    if (!to) return this.invalidInput('to required');
    if (!args.html && !args.text) return this.invalidInput('html or text required');
    const idempotencyKey = this.integrationId
      ? `${this.integrationId}:${to}:${args.subject.slice(0, 64)}`
      : undefined;
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'send_email',
        agentId: this.agentId,
        integrationId: this.integrationId,
        idempotencyKey,
      },
      async () => {
        const t = await this.resolveToken();
        const meta = (t.metadata ?? {}) as SendGridMetadata;
        const from = args.from ?? meta.fromEmail;
        if (!from) throw new Error('sendgrid_from_email_missing');
        const body: Record<string, unknown> = {
          personalizations: [
            {
              to: [{ email: to }],
              ...(args.cc?.length ? { cc: args.cc.map((e) => ({ email: e })) } : {}),
              ...(args.bcc?.length ? { bcc: args.bcc.map((e) => ({ email: e })) } : {}),
              subject: args.subject,
            },
          ],
          from: { email: from, ...(meta.fromName ? { name: meta.fromName } : {}) },
          ...(args.replyTo ? { reply_to: { email: args.replyTo } } : {}),
          content: [
            ...(args.text ? [{ type: 'text/plain', value: args.text }] : []),
            ...(args.html ? [{ type: 'text/html', value: args.html }] : []),
          ],
        };
        return sendgridSend(t.token, body, idempotencyKey);
      },
    );
  }

  @tool({
    name: TEMPLATE_TOOL,
    description: 'Send a SendGrid dynamic-template email by template id.',
    parameters: {
      type: 'object',
      properties: {
        to: { type: 'string' },
        templateId: { type: 'string', description: 'SendGrid dynamic template id (d-…).' },
        dynamicTemplateData: { type: 'object', additionalProperties: true },
        from: { type: 'string' },
      },
      required: ['to', 'templateId'],
    },
  })
  async sendTemplate(args: {
    to: string;
    templateId: string;
    dynamicTemplateData?: Record<string, unknown>;
    from?: string;
  }, ctx?: Context) {
    if (!this.capabilityEnabled('send_messages')) return this.capabilityDisabled('send_messages');
    const to = args.to ?? this.bridgeRecipient(ctx);
    if (!to) return this.invalidInput('to required');
    const idempotencyKey = this.integrationId
      ? `${this.integrationId}:${to}:${args.templateId}`
      : undefined;
    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'send_template',
        agentId: this.agentId,
        integrationId: this.integrationId,
        idempotencyKey,
      },
      async () => {
        const t = await this.resolveToken();
        const meta = (t.metadata ?? {}) as SendGridMetadata;
        const from = args.from ?? meta.fromEmail;
        if (!from) throw new Error('sendgrid_from_email_missing');
        const body: Record<string, unknown> = {
          personalizations: [
            {
              to: [{ email: to }],
              ...(args.dynamicTemplateData
                ? { dynamic_template_data: args.dynamicTemplateData }
                : {}),
            },
          ],
          from: { email: from, ...(meta.fromName ? { name: meta.fromName } : {}) },
          template_id: args.templateId,
        };
        return sendgridSend(t.token, body, idempotencyKey);
      },
    );
  }

  /**
   * SendGrid Inbound Parse webhook. URL pattern:
   *   POST /messaging/sendgrid/inbound-parse/:token
   *
   * The `:token` is `metadata.inboundParseToken` set on the connected
   * account at connect time (rotated on disconnect). Verifying it
   * provides a per-account secret without relying on SendGrid's basic
   * auth field, which they advise against using in production.
   *
   * Body is multipart/form-data with `from`, `to`, `subject`, `text`,
   * `html`, and (when configured) `attachments`. The skill emits the
   * structured event into the host bridge through a `console.log`
   * structured fence; portal hosts mount a thin route on top that
   * dispatches into `processSendGridInboundParse`.
   */
  @http({
    path: '/messaging/sendgrid/inbound-parse/:token',
    method: 'POST',
    auth: 'public',
    description: 'SendGrid Inbound Parse webhook (multipart/form-data).',
  })
  async inboundParse(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const token = url.pathname.split('/').pop() ?? '';
    if (!token) return new Response('forbidden', { status: 403 });
    const t = await this.resolveToken();
    const meta = (t.metadata ?? {}) as SendGridMetadata;
    if (!meta.inboundParseToken || meta.inboundParseToken !== token) {
      return new Response('forbidden', { status: 403 });
    }
    const ct = req.headers.get('content-type') ?? '';
    if (!ct.startsWith('multipart/form-data')) {
      return new Response('expected multipart/form-data', { status: 400 });
    }
    const form = await req.formData();
    const get = (k: string) => {
      const v = form.get(k);
      return typeof v === 'string' ? v : undefined;
    };
    console.log(
      JSON.stringify({
        kind: 'sendgrid.inbound-parse',
        from: get('from'),
        to: get('to'),
        subject: get('subject'),
        text: get('text'),
        html: get('html'),
        agentId: this.agentId,
        integrationId: this.integrationId,
      }),
    );
    return new Response('ok', { status: 200 });
  }

  @prompt({ name: 'sendgrid-bridge-awareness' })
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

async function sendgridSend(
  apiKey: string,
  body: Record<string, unknown>,
  idempotencyKey?: string,
): Promise<{ externalMessageId?: string }> {
  // SendGrid v3 /mail/send does NOT honor an Idempotency-Key request header
  // (the API is not idempotent-safe). The recommended pattern is to attach
  // the key as a custom_arg and dedupe on the inbound event webhook. We
  // forward the key both as a custom_arg and as a header so:
  //   - hosts that already key off the Event Webhook can dedupe deliveries
  //     by `custom_args.idempotency_key`;
  //   - if SendGrid ever adds idempotency support, it lights up automatically
  //     without a code change.
  // Ref: https://stackoverflow.com/q/76385011 (linking to SendGrid support).
  const finalBody: Record<string, unknown> = idempotencyKey
    ? {
        ...body,
        custom_args: {
          ...((body.custom_args as Record<string, unknown> | undefined) ?? {}),
          idempotency_key: idempotencyKey,
        },
      }
    : body;
  const headers: Record<string, string> = {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  };
  if (idempotencyKey) headers['Idempotency-Key'] = idempotencyKey;
  const r = await fetch(`${SENDGRID_API}/v3/mail/send`, {
    method: 'POST',
    headers,
    body: JSON.stringify(finalBody),
  });
  if (!r.ok) {
    const text = await r.text();
    const err = new Error(text.slice(0, 300) || `sendgrid ${r.status}`) as Error & {
      status?: number;
    };
    err.status = r.status;
    throw err;
  }
  // SendGrid returns the message id in the `X-Message-Id` response header on
  // 202; the body is empty.
  const externalMessageId = r.headers.get('x-message-id') ?? undefined;
  return { externalMessageId };
}
