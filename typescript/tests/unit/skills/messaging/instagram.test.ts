/**
 * InstagramSkill — DM via connected Page; image publish two-step.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { InstagramSkill } from '../../../../src/skills/messaging/instagram';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof InstagramSkill>[0]> = {}) {
  return new InstagramSkill({
    agentId: 'agent-1',
    enabledCapabilities: ['send_messages', 'publish_posts'],
    getToken: async () => ({
      token: 'page-token',
      metadata: { igUserId: 'IG1', pageId: 'PG1' },
    }),
    ...opts,
  });
}

describe('InstagramSkill', () => {
  it('refuses DM without recipient', async () => {
    const skill = makeSkill();
    const r = await skill.sendDm({ text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input' });
  });

  it('publishes image via /media + /media_publish', async () => {
    const containerId = 'CON1';
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes('/media_publish')) {
        return new Response(JSON.stringify({ id: 'MEDIA1' }), { status: 200 });
      }
      if (url.includes('/media')) {
        return new Response(JSON.stringify({ id: containerId }), { status: 200 });
      }
      throw new Error('unexpected ' + url);
    });
    const skill = makeSkill();
    const r = (await skill.publishImage({ image_url: 'https://x/y.jpg', caption: 'hi' })) as {
      ok: true;
      data: { externalMessageId?: string; containerId?: string };
      providerMessageId?: string;
    };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('MEDIA1');
    expect(r.providerMessageId).toBe('MEDIA1');
    expect(r.data.containerId).toBe(containerId);
    expect(fetchSpy.mock.calls.length).toBe(2);
  });

  it('publishImage accepts content_id when host resolves to public URL', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes('/media_publish')) {
        return new Response(JSON.stringify({ id: 'MEDIA9' }), { status: 200 });
      }
      return new Response(JSON.stringify({ id: 'CON9' }), { status: 200 });
    });
    const skill = makeSkill({
      resolveMediaForOutbound: async (id) => ({
        url: `https://cdn.example.com/${id}.png`,
      }),
    });
    const r = (await skill.publishImage({
      content_id: '11111111-2222-3333-4444-555555555555',
      caption: 'hi',
    })) as { ok: true; data: { externalMessageId?: string } };
    expect(r.ok).toBe(true);
    const firstUrl = String(fetchSpy.mock.calls[0][0]);
    expect(firstUrl).toContain('/IG1/media');
  });

  it('sendImage posts an image attachment to the connected Page', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response(JSON.stringify({ message_id: 'mid.ig' }), { status: 200 }));
    const skill = makeSkill();
    const r = (await skill.sendImage({
      recipient_id: 'IGSID1',
      image_url: 'https://cdn.example.com/x.png',
    })) as { ok: true; data: { externalMessageId?: string } };
    expect(r.ok).toBe(true);
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toContain('/PG1/messages');
    const body = JSON.parse(init.body as string) as {
      message: { attachment: { type: string; payload: { url: string } } };
    };
    expect(body.message.attachment.type).toBe('image');
    expect(body.message.attachment.payload.url).toBe('https://cdn.example.com/x.png');
  });

  it('sendImage rejects when content cannot be resolved to a public URL', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('should not be called'));
    const skill = makeSkill({
      resolveMediaForOutbound: async () => ({ url: '/api/content/abc.png' }),
    });
    const r = await skill.sendImage({
      recipient_id: 'IGSID1',
      content_id: '11111111-2222-3333-4444-555555555555',
    });
    expect(r).toMatchObject({ ok: false, code: 'instagram_requires_public_url' });
  });
});
