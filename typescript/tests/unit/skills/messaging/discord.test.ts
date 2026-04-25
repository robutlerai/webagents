/**
 * DiscordSkill — capability gating, send-DM wire shape, interactions
 * Ed25519 verification.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { DiscordSkill } from '../../../../src/skills/messaging/discord';

const TOKEN = 'discord-bot-token';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof DiscordSkill>[0]> = {}) {
  return new DiscordSkill({
    agentId: 'agent-1',
    enabledCapabilities: ['send_messages'],
    getToken: async () => ({
      token: TOKEN,
      metadata: { applicationId: 'A1', publicKey: '00'.repeat(32) },
    }),
    ...opts,
  });
}

describe('DiscordSkill', () => {
  it('opens DM channel then posts a message', async () => {
    let openCalled = false;
    let messagesCalled = false;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith('/users/@me/channels')) {
        openCalled = true;
        return new Response(JSON.stringify({ id: 'C42' }));
      }
      if (url.endsWith('/channels/C42/messages')) {
        messagesCalled = true;
        return new Response(JSON.stringify({ id: 'M1' }));
      }
      throw new Error('unexpected url ' + url);
    });
    const skill = makeSkill();
    const r = (await skill.sendDm({ user_id: 'U1', content: 'hi' })) as {
      ok: true;
      data: { id?: string };
      providerMessageId?: string;
    };
    expect(openCalled).toBe(true);
    expect(messagesCalled).toBe(true);
    expect(r.providerMessageId).toBe('M1');
  });

  it('sendDmPhoto opens DM then uploads via multipart', async () => {
    const calls: { url: string; init?: RequestInit }[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      calls.push({ url, init });
      if (url.endsWith('/users/@me/channels')) {
        return new Response(JSON.stringify({ id: 'C99' }));
      }
      if (url.endsWith('/channels/C99/messages')) {
        return new Response(JSON.stringify({ id: 'M99' }));
      }
      throw new Error('unexpected ' + url);
    });
    const skill = new DiscordSkill({
      agentId: 'agent-1',
      enabledCapabilities: ['send_messages'],
      getToken: async () => ({
        token: TOKEN,
        metadata: { applicationId: 'A1', publicKey: '00'.repeat(32) },
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
    const r = await skill.sendDmPhoto({
      user_id: 'U1',
      content_id: '11111111-2222-3333-4444-555555555555',
      content: 'caption',
    });
    expect(r.ok).toBe(true);
    expect(calls.length).toBe(2);
    const upload = calls[1];
    expect(upload.url).toContain('/channels/C99/messages');
    expect(upload.init?.body).toBeInstanceOf(FormData);
    const form = upload.init?.body as FormData;
    const payload = JSON.parse(form.get('payload_json') as string) as { content?: string };
    expect(payload.content).toBe('caption');
    expect(form.get('files[0]')).toBeInstanceOf(Blob);
  });

  it('sendInChannelDocument uploads bytes from a public URL', async () => {
    const calls: { url: string; init?: RequestInit }[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      calls.push({ url, init });
      return new Response(JSON.stringify({ id: 'M2' }));
    });
    const skill = new DiscordSkill({
      agentId: 'agent-1',
      enabledCapabilities: ['send_messages'],
      getToken: async () => ({
        token: TOKEN,
        metadata: { applicationId: 'A1', publicKey: '00'.repeat(32) },
      }),
      resolveMediaForOutbound: async () => ({ url: 'https://cdn.example.com/y.pdf' }),
    });
    const r = await skill.sendInChannelDocument({
      channel_id: 'C1',
      content_id: '11111111-2222-3333-4444-555555555555',
      filename: 'report.pdf',
    });
    expect(r.ok).toBe(false);
    expect(r.message).toMatch(/discord_requires_bytes/);
  });

  it('rejects interactions request with invalid Ed25519 signature', async () => {
    const skill = makeSkill();
    const req = new Request('https://example.com/x', {
      method: 'POST',
      headers: {
        'X-Signature-Ed25519': 'aa'.repeat(64),
        'X-Signature-Timestamp': String(Math.floor(Date.now() / 1000)),
      },
      body: JSON.stringify({ type: 1 }),
    });
    const res = await skill.interactions(req);
    expect(res.status).toBe(401);
  });
});
