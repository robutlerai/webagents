/**
 * SlackSkill — capability gating, fetch wire shape, signature
 * verification on the @http interactivity endpoint.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { SlackSkill } from '../../../../src/skills/messaging/slack';

const TOKEN = 'xoxb-test';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof SlackSkill>[0]> = {}) {
  return new SlackSkill({
    agentId: 'agent-1',
    enabledCapabilities: ['send_messages', 'publish_posts'],
    getToken: async () => ({
      token: TOKEN,
      metadata: { teamId: 'T1', botUserId: 'U1', signingSecret: 'sshh' },
    }),
    ...opts,
  });
}

describe('SlackSkill', () => {
  it('opens conversation then posts when sending DM', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith('/conversations.open')) {
        return new Response(JSON.stringify({ ok: true, channel: { id: 'D1' } }));
      }
      return new Response(JSON.stringify({ ok: true, ts: '1.0', channel: 'D1' }));
    });
    const skill = makeSkill();
    const r = (await skill.sendDm({ user_id: 'U2', text: 'hi' })) as {
      ok: true;
      data: { externalMessageId?: string };
      providerMessageId?: string;
    };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('D1:1.0');
    expect(r.providerMessageId).toBe('D1:1.0');
    expect(fetchSpy.mock.calls.map((c) => String(c[0]))).toContain('https://slack.com/api/chat.postMessage');
  });

  it('rejects interactivity request with bad signature', async () => {
    const skill = makeSkill();
    const req = new Request('https://example.com/x', {
      method: 'POST',
      headers: {
        'X-Slack-Request-Timestamp': String(Math.floor(Date.now() / 1000)),
        'X-Slack-Signature': 'v0=00',
      },
      body: 'payload=%7B%7D',
    });
    const res = await skill.interactivity(req);
    expect(res.status).toBe(403);
  });

  it('postInChannelPhoto runs the 3-step files.getUploadURLExternal flow', async () => {
    const calls: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      calls.push(url);
      if (url.includes('/files.getUploadURLExternal')) {
        return new Response(
          JSON.stringify({
            ok: true,
            upload_url: 'https://files.slack.com/upload/abc',
            file_id: 'F1',
          }),
        );
      }
      if (url.startsWith('https://files.slack.com/')) {
        return new Response('ok', { status: 200 });
      }
      if (url.endsWith('/files.completeUploadExternal')) {
        return new Response(JSON.stringify({ ok: true }));
      }
      throw new Error('unexpected ' + url);
    });
    const skill = new SlackSkill({
      agentId: 'agent-1',
      enabledCapabilities: ['send_messages', 'publish_posts'],
      getToken: async () => ({
        token: TOKEN,
        metadata: { teamId: 'T1', botUserId: 'U1', signingSecret: 'sshh' },
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
    const r = (await skill.postInChannelPhoto({
      channel: 'C1',
      content_id: '11111111-2222-3333-4444-555555555555',
      initial_comment: 'check',
    })) as { ok: true; data: { externalMessageId?: string } };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('F1');
    expect(calls.some((u) => u.includes('/files.getUploadURLExternal'))).toBe(true);
    expect(calls.some((u) => u.startsWith('https://files.slack.com/'))).toBe(true);
    expect(calls.some((u) => u.endsWith('/files.completeUploadExternal'))).toBe(true);
  });

  it('postInChannelPhoto rejects when host cannot supply bytes', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('should not be called'));
    const skill = new SlackSkill({
      agentId: 'agent-1',
      enabledCapabilities: ['send_messages', 'publish_posts'],
      getToken: async () => ({
        token: TOKEN,
        metadata: { teamId: 'T1', botUserId: 'U1', signingSecret: 'sshh' },
      }),
      resolveMediaForOutbound: async () => ({ url: 'https://cdn.example.com/x.png' }),
    });
    const r = await skill.postInChannelPhoto({
      channel: 'C1',
      content_id: '11111111-2222-3333-4444-555555555555',
    });
    expect(r.ok).toBe(false);
    expect(r.message).toMatch(/slack_requires_bytes/);
  });

  it('disables postInChannel when neither capability is enabled', async () => {
    const skill = makeSkill({ enabledCapabilities: ['receive_messages'] });
    const r = await skill.postInChannel({ channel: 'C1', text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', code: 'capability_disabled' });
  });
});
