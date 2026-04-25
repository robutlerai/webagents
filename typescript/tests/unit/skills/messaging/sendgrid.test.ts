/**
 * SendGridSkill — capability gating, mail/send shape, idempotency header,
 * Inbound Parse token validation.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { SendGridSkill } from '../../../../src/skills/messaging/sendgrid';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof SendGridSkill>[0]> = {}) {
  return new SendGridSkill({
    agentId: 'agent-1',
    integrationId: 'integ-1',
    enabledCapabilities: ['send_messages'],
    getToken: async () => ({
      token: 'SG.test-key',
      metadata: {
        fromEmail: 'noreply@example.com',
        fromName: 'Example',
        inboundParseToken: 'parse-token-123',
      },
    }),
    ...opts,
  });
}

describe('SendGridSkill.sendEmail', () => {
  it('refuses without send_messages capability', async () => {
    const skill = makeSkill({ enabledCapabilities: [] as string[] });
    // Empty enabledCapabilities means "all enabled" by base contract; force
    // an unrelated capability so send_messages is gated off.
    const skill2 = makeSkill({ enabledCapabilities: ['unrelated'] });
    const r = await skill2.sendEmail({ to: 'x@y.com', subject: 'hi', text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', code: 'capability_disabled' });
    void skill;
  });

  it('rejects missing to', async () => {
    const skill = makeSkill();
    const r = await skill.sendEmail({ to: '', subject: 'hi', text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', message: 'to required' });
  });

  it('rejects empty body (no html or text)', async () => {
    const skill = makeSkill();
    const r = await skill.sendEmail({ to: 'x@y.com', subject: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', message: 'html or text required' });
  });

  it('POSTs to /v3/mail/send with personalizations + idempotency custom_arg', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(null, {
        status: 202,
        headers: { 'x-message-id': 'msg-abc' },
      }),
    );
    const skill = makeSkill();
    const r = (await skill.sendEmail({
      to: 'x@y.com',
      subject: 'hi',
      text: 'body',
    })) as { ok: true; data: { externalMessageId?: string }; providerMessageId?: string };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('msg-abc');
    expect(r.providerMessageId).toBe('msg-abc');
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://api.sendgrid.com/v3/mail/send');
    const headers = init.headers as Record<string, string>;
    expect(headers.Authorization).toBe('Bearer SG.test-key');
    // Idempotency-Key is sent for forward compatibility (currently a no-op
    // on SendGrid's side per their API contract).
    expect(headers['Idempotency-Key']).toBe('integ-1:x@y.com:hi');
    const body = JSON.parse(init.body as string) as {
      from: { email: string };
      custom_args?: { idempotency_key?: string };
    };
    expect(body.from.email).toBe('noreply@example.com');
    // The same key is mirrored into custom_args so hosts can dedupe via
    // SendGrid's Event Webhook (which echoes custom_args back).
    expect(body.custom_args?.idempotency_key).toBe('integ-1:x@y.com:hi');
  });
});

describe('SendGridSkill.sendTemplate', () => {
  it('POSTs with template_id and dynamic_template_data', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(null, { status: 202, headers: { 'x-message-id': 'msg-xyz' } }),
    );
    const skill = makeSkill();
    const r = (await skill.sendTemplate({
      to: 'x@y.com',
      templateId: 'd-abc',
      dynamicTemplateData: { name: 'Alice' },
    })) as { ok: true; data: { externalMessageId?: string } };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('msg-xyz');
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      template_id?: string;
      personalizations: Array<{ dynamic_template_data?: Record<string, unknown> }>;
    };
    expect(body.template_id).toBe('d-abc');
    expect(body.personalizations[0].dynamic_template_data).toEqual({ name: 'Alice' });
  });
});

describe('SendGridSkill Inbound Parse', () => {
  it('rejects mismatched token', async () => {
    const skill = makeSkill();
    const req = new Request(
      'https://example.com/messaging/sendgrid/inbound-parse/wrong-token',
      {
        method: 'POST',
        headers: { 'content-type': 'multipart/form-data; boundary=xx' },
        body: '',
      },
    );
    const res = await skill.inboundParse(req);
    expect(res.status).toBe(403);
  });

  it('accepts matching token + multipart form', async () => {
    const skill = makeSkill();
    const form = new FormData();
    form.append('from', 'sender@x.com');
    form.append('to', 'agent@example.com');
    form.append('subject', 'hi');
    form.append('text', 'hello');
    const req = new Request(
      'https://example.com/messaging/sendgrid/inbound-parse/parse-token-123',
      { method: 'POST', body: form },
    );
    const res = await skill.inboundParse(req);
    expect(res.status).toBe(200);
  });
});
