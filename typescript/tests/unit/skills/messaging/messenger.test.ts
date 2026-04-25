/**
 * MessengerSkill — capability gating + Page-level send wire shape.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MessengerSkill } from '../../../../src/skills/messaging/messenger';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof MessengerSkill>[0]> = {}) {
  return new MessengerSkill({
    agentId: 'agent-1',
    enabledCapabilities: ['send_messages'],
    getToken: async () => ({ token: 'page-token', metadata: { pageId: 'PG1' } }),
    ...opts,
  });
}

describe('MessengerSkill', () => {
  it('refuses without recipient when not bridged', async () => {
    const skill = makeSkill();
    const r = await skill.sendText({ text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input' });
  });

  it('POSTs /<pageId>/messages with RESPONSE messaging_type', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response(JSON.stringify({ message_id: 'mid.123' }), { status: 200 }));
    const skill = makeSkill();
    const r = (await skill.sendText({ recipient_psid: 'PSID1', text: 'hi' })) as {
      ok: true;
      data: { externalMessageId?: string };
      providerMessageId?: string;
    };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('mid.123');
    expect(r.providerMessageId).toBe('mid.123');
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toContain('/PG1/messages');
    const body = JSON.parse(init.body as string) as { messaging_type: string };
    expect(body.messaging_type).toBe('RESPONSE');
  });

  it('sendImage posts an image attachment with absolute https url', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response(JSON.stringify({ message_id: 'mid.img' }), { status: 200 }));
    const skill = makeSkill();
    const r = (await skill.sendImage({
      recipient_psid: 'PSID1',
      image_url: 'https://cdn.example.com/x.png',
    })) as { ok: true; data: { externalMessageId?: string } };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('mid.img');
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      message: { attachment: { type: string; payload: { url: string } } };
    };
    expect(body.message.attachment.type).toBe('image');
    expect(body.message.attachment.payload.url).toBe('https://cdn.example.com/x.png');
  });

  it('sendImage rejects relative content URLs (Meta needs public https)', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('should not be called'));
    const skill = new MessengerSkill({
      agentId: 'agent-1',
      enabledCapabilities: ['send_messages'],
      getToken: async () => ({ token: 'page-token', metadata: { pageId: 'PG1' } }),
      resolveMediaForOutbound: async () => ({ url: '/api/content/abc.png' }),
    });
    const r = await skill.sendImage({
      recipient_psid: 'PSID1',
      content_id: '11111111-2222-3333-4444-555555555555',
    });
    expect(r).toMatchObject({ ok: false, code: 'messenger_requires_public_url' });
  });

  it('sendDocument posts a file attachment', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response(JSON.stringify({ message_id: 'mid.doc' }), { status: 200 }));
    const skill = makeSkill();
    const r = (await skill.sendDocument({
      recipient_psid: 'PSID1',
      document_url: 'https://cdn.example.com/x.pdf',
    })) as { ok: true; data: { externalMessageId?: string } };
    expect(r.ok).toBe(true);
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      message: { attachment: { type: string; payload: { url: string } } };
    };
    expect(body.message.attachment.type).toBe('file');
    expect(body.message.attachment.payload.url).toBe('https://cdn.example.com/x.pdf');
  });

  it('sendImage with caption sends text first then attachment', async () => {
    const responses = [
      new Response(JSON.stringify({ message_id: 'mid.text' }), { status: 200 }),
      new Response(JSON.stringify({ message_id: 'mid.img' }), { status: 200 }),
    ];
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockImplementation(async () => responses.shift()!);
    const skill = makeSkill();
    const r = await skill.sendImage({
      recipient_psid: 'PSID1',
      image_url: 'https://cdn.example.com/x.png',
      caption: 'check this out',
    });
    expect(r.ok).toBe(true);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
    const firstBody = JSON.parse(
      (fetchSpy.mock.calls[0][1] as RequestInit).body as string,
    ) as { message: { text?: string } };
    expect(firstBody.message.text).toBe('check this out');
  });
});
