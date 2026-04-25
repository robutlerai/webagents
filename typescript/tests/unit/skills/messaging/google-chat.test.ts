/**
 * GoogleChatSkill — capability gating, message/card POST shape, JWT-verified
 * event endpoint guards.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { GoogleChatSkill } from '../../../../src/skills/messaging/google-chat';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof GoogleChatSkill>[0]> = {}) {
  return new GoogleChatSkill({
    agentId: 'agent-1',
    integrationId: 'integ-1',
    enabledCapabilities: ['send_messages', 'list_spaces'],
    getToken: async () => ({
      token: 'gchat-token',
      metadata: {
        workspaceDomain: 'example.com',
        expectedAudience: 'project-123',
      },
    }),
    ...opts,
  });
}

describe('GoogleChatSkill.sendMessage', () => {
  it('refuses without send_messages capability', async () => {
    const skill = makeSkill({ enabledCapabilities: ['list_spaces'] });
    const r = await skill.sendMessage({ space: 'spaces/abc', text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', code: 'capability_disabled' });
  });

  it('rejects missing space', async () => {
    const skill = makeSkill();
    const r = await skill.sendMessage({ text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', message: 'space required' });
  });

  it('POSTs to <space>/messages with text body', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ name: 'spaces/abc/messages/m-1' })),
    );
    const skill = makeSkill();
    const r = (await skill.sendMessage({ space: 'spaces/abc', text: 'hello' })) as {
      ok: true;
      data: { externalMessageId?: string };
      providerMessageId?: string;
    };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('spaces/abc/messages/m-1');
    expect(r.providerMessageId).toBe('spaces/abc/messages/m-1');
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://chat.googleapis.com/v1/spaces/abc/messages');
    expect(init.method).toBe('POST');
    expect(init.body as string).toContain('"text":"hello"');
  });

  it('forwards thread.name when provided', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ name: 'spaces/abc/messages/m-2' })),
    );
    const skill = makeSkill();
    await skill.sendMessage({ space: 'spaces/abc', text: 'reply', thread: 'spaces/abc/threads/t1' });
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(init.body as string).toContain('"thread":{"name":"spaces/abc/threads/t1"}');
  });
});

describe('GoogleChatSkill.sendCard', () => {
  it('rejects empty cardsV2', async () => {
    const skill = makeSkill();
    const r = await skill.sendCard({ space: 'spaces/abc', cardsV2: [] });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', message: 'cardsV2 required' });
  });
});

describe('GoogleChatSkill event endpoint', () => {
  it('returns 403 without a Bearer authorization header', async () => {
    const skill = makeSkill();
    const req = new Request('https://example.com/messaging/google-chat/event', {
      method: 'POST',
      body: '{}',
    });
    const res = await skill.event(req);
    expect(res.status).toBe(403);
  });

  it('returns 403 for an obviously invalid JWT (no JWKS roundtrip)', async () => {
    const skill = makeSkill();
    const req = new Request('https://example.com/messaging/google-chat/event', {
      method: 'POST',
      headers: { authorization: 'Bearer not.a.jwt' },
      body: '{}',
    });
    const res = await skill.event(req);
    expect(res.status).toBe(403);
  });

  it('returns 503 when expectedAudience is unconfigured', async () => {
    const skill = makeSkill({
      getToken: async () => ({ token: 'gchat-token', metadata: {} }),
    });
    delete process.env.GOOGLE_CHAT_PROJECT_NUMBER;
    delete process.env.GOOGLE_CHAT_AUDIENCE;
    const req = new Request('https://example.com/messaging/google-chat/event', {
      method: 'POST',
      headers: { authorization: 'Bearer not.a.jwt' },
      body: '{}',
    });
    const res = await skill.event(req);
    expect(res.status).toBe(503);
  });
});
