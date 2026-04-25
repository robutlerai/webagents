/**
 * WhatsAppSkill — capability gating, recipient resolution from bridge,
 * Graph API wire shape.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { WhatsAppSkill } from '../../../../src/skills/messaging/whatsapp';

const TOKEN = 'whatsapp-system-user-token';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof WhatsAppSkill>[0]> = {}) {
  return new WhatsAppSkill({
    agentId: 'agent-1',
    enabledCapabilities: ['send_messages'],
    getToken: async () => ({
      token: TOKEN,
      metadata: { phoneNumberId: 'PN1', wabaId: 'WABA1' },
    }),
    ...opts,
  });
}

describe('WhatsAppSkill', () => {
  it('refuses when capability is disabled', async () => {
    const skill = makeSkill({ enabledCapabilities: ['receive_messages'] });
    const r = await skill.sendText({ to: '15551234567', text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', code: 'capability_disabled' });
  });

  it('returns invalid_input when no bridge', async () => {
    const skill = makeSkill();
    const r = await skill.sendText({ text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', message: 'Recipient required' });
  });

  it('POSTs /<phoneNumberId>/messages with correct body', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response(JSON.stringify({ messages: [{ id: 'wamid.AAA' }] }), { status: 200 }));
    const skill = makeSkill();
    const r = (await skill.sendText({ to: '15551234567', text: 'hi' })) as {
      ok: true;
      data: { externalMessageId?: string };
      providerMessageId?: string;
    };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('wamid.AAA');
    expect(r.providerMessageId).toBe('wamid.AAA');
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toContain('/PN1/messages');
    const body = JSON.parse(init.body as string) as { to: string; type: string };
    expect(body.to).toBe('15551234567');
    expect(body.type).toBe('text');
  });

  it('sendImage posts image.link first', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(
        new Response(JSON.stringify({ messages: [{ id: 'wamid.IMG' }] }), { status: 200 }),
      );
    const skill = makeSkill();
    const r = (await skill.sendImage({
      to: '15551234567',
      image_url: 'https://cdn.example.com/x.png',
    })) as { ok: true; data: { externalMessageId?: string } };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('wamid.IMG');
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      type: string;
      image: { link: string };
    };
    expect(body.type).toBe('image');
    expect(body.image.link).toBe('https://cdn.example.com/x.png');
  });

  it('sendImage falls back to /media upload when URL fetch fails', async () => {
    const responses: Response[] = [
      new Response(
        JSON.stringify({
          error: { message: 'failed to get HTTP URL content for media' },
        }),
        { status: 400 },
      ),
      new Response(JSON.stringify({ id: 'media-id-1' }), { status: 200 }),
      new Response(JSON.stringify({ messages: [{ id: 'wamid.IMG2' }] }), { status: 200 }),
    ];
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockImplementation(async () => responses.shift()!);
    const skill = new WhatsAppSkill({
      agentId: 'agent-1',
      enabledCapabilities: ['send_messages'],
      getToken: async () => ({
        token: TOKEN,
        metadata: { phoneNumberId: 'PN1', wabaId: 'WABA1' },
      }),
      resolveMediaForOutbound: async () => ({
        url: 'https://cdn.example.com/x.png',
        fetchBytes: async () => ({
          buffer: new Uint8Array([1, 2, 3]),
          contentType: 'image/png',
          filename: 'x.png',
        }),
      }),
    });
    const r = (await skill.sendImage({
      to: '15551234567',
      content_id: '11111111-2222-3333-4444-555555555555',
    })) as { ok: true; data: { externalMessageId?: string } };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('wamid.IMG2');
    expect(fetchSpy).toHaveBeenCalledTimes(3);
    const uploadUrl = String(fetchSpy.mock.calls[1][0]);
    expect(uploadUrl).toContain('/PN1/media');
    const finalBody = JSON.parse(
      (fetchSpy.mock.calls[2][1] as RequestInit).body as string,
    ) as { image: { id: string } };
    expect(finalBody.image.id).toBe('media-id-1');
  });

  it('sendDocument posts document.link with filename', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(
        new Response(JSON.stringify({ messages: [{ id: 'wamid.DOC' }] }), { status: 200 }),
      );
    const skill = makeSkill();
    const r = await skill.sendDocument({
      to: '15551234567',
      document_url: 'https://cdn.example.com/x.pdf',
      filename: 'invoice.pdf',
    });
    expect(r.ok).toBe(true);
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      type: string;
      document: { link: string; filename?: string };
    };
    expect(body.type).toBe('document');
    expect(body.document.link).toBe('https://cdn.example.com/x.pdf');
    expect(body.document.filename).toBe('invoice.pdf');
  });
});
