/**
 * TwilioSkill — capability gating, A2P guard, signature verification on
 * the @http status callback.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TwilioSkill } from '../../../../src/skills/messaging/twilio';

const TOKEN = 'twilio-auth-token';
const SID = 'AC1234567890';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof TwilioSkill>[0]> = {}) {
  return new TwilioSkill({
    agentId: 'agent-1',
    enabledCapabilities: ['send_messages'],
    getToken: async () => ({
      token: TOKEN,
      metadata: { accountSid: SID, fromNumber: '+15555550100', a2pCampaignRegistered: true },
    }),
    ...opts,
  });
}

describe('TwilioSkill', () => {
  it('refuses +1 sends when A2P campaign not registered', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('should not be called'));
    const skill = new TwilioSkill({
      agentId: 'agent-1',
      enabledCapabilities: ['send_messages'],
      getToken: async () => ({ token: TOKEN, metadata: { accountSid: SID, fromNumber: '+1555' } }),
    });
    const r = await skill.sendSms({ to: '+15555551234', body: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'provider_api_error', message: 'a2p_campaign_not_registered' });
  });

  it('builds a Messages.json POST with Basic auth + form body', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response(JSON.stringify({ sid: 'SM123' }), { status: 200 }));
    const skill = makeSkill();
    const r = (await skill.sendSms({ to: '+15555551234', body: 'hi' })) as {
      ok: true;
      data: { externalMessageId?: string };
      providerMessageId?: string;
    };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('SM123');
    expect(r.providerMessageId).toBe('SM123');
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`https://api.twilio.com/2010-04-01/Accounts/${SID}/Messages.json`);
    expect((init.headers as Record<string, string>).Authorization).toMatch(/^Basic /);
    const body = init.body as string;
    expect(body).toMatch(/To=%2B15555551234/);
    expect(body).toMatch(/Body=hi/);
    expect(body).toMatch(/From=%2B15555550100/);
  });

  it('sendMms accepts content_id and resolves to a public URL', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response(JSON.stringify({ sid: 'MM1' }), { status: 200 }));
    const skill = new TwilioSkill({
      agentId: 'agent-1',
      enabledCapabilities: ['send_messages'],
      getToken: async () => ({
        token: TOKEN,
        metadata: { accountSid: SID, fromNumber: '+15555550100', a2pCampaignRegistered: true },
      }),
      resolveMediaForOutbound: async (id) => ({
        url: `https://cdn.example.com/content/${id}.png`,
      }),
    });
    const r = (await skill.sendMms({
      to: '+15555551234',
      body: 'see this',
      content_id: '11111111-2222-3333-4444-555555555555',
    })) as { ok: true; data: { externalMessageId?: string } };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('MM1');
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = init.body as string;
    expect(body).toMatch(/MediaUrl=https%3A%2F%2Fcdn\.example\.com%2Fcontent%2F/);
  });

  it('sendMms rejects content_id when host cannot resolve to a public URL', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('should not be called'));
    const skill = new TwilioSkill({
      agentId: 'agent-1',
      enabledCapabilities: ['send_messages'],
      getToken: async () => ({
        token: TOKEN,
        metadata: { accountSid: SID, fromNumber: '+15555550100', a2pCampaignRegistered: true },
      }),
      resolveMediaForOutbound: async () => ({ url: '/api/content/abc.png' }),
    });
    const r = await skill.sendMms({
      to: '+15555551234',
      content_id: '11111111-2222-3333-4444-555555555555',
    });
    expect(r).toMatchObject({ ok: false, code: 'twilio_requires_public_url' });
  });

  it('sendMms rejects when neither content_id nor media_url is provided', async () => {
    const skill = makeSkill();
    const r = await skill.sendMms({ to: '+15555551234', body: 'hi' });
    expect(r).toMatchObject({ ok: false, code: 'invalid_media_reference' });
  });

  it('rejects status callback with bad signature', async () => {
    const skill = makeSkill();
    const req = new Request('https://example.com/x', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'X-Twilio-Signature': 'wrong',
      },
      body: 'MessageSid=SM1&MessageStatus=delivered',
    });
    const res = await skill.statusCallback(req);
    expect(res.status).toBe(403);
  });
});
