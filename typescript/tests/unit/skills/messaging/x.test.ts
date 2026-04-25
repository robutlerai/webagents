/**
 * XSkill — DM + tweet publish flows.
 *
 * Capability gating, JSON request shape, ApiCallResult discrimination,
 * and CRC handler / signature verification at the @http endpoints.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createHmac } from 'node:crypto';
import { XSkill } from '../../../../src/skills/messaging/x';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof XSkill>[0]> = {}) {
  return new XSkill({
    agentId: 'agent-1',
    integrationId: 'integ-1',
    enabledCapabilities: ['send_messages', 'publish_posts'],
    getToken: async () => ({
      token: 'x-token',
      metadata: { consumerSecret: 'secret-abc' },
    }),
    ...opts,
  });
}

describe('XSkill DMs', () => {
  it('refuses send_dm without send_messages capability', async () => {
    const skill = makeSkill({ enabledCapabilities: ['publish_posts'] });
    const r = await skill.sendDm({ participant_id: '123', text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', code: 'capability_disabled' });
  });

  it('rejects missing participant_id', async () => {
    const skill = makeSkill();
    const r = await skill.sendDm({ text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', message: 'participant_id required' });
  });

  it('POSTs to /2/dm_conversations/with/<id>/messages and returns dm_event_id', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ data: { dm_event_id: 'evt-1' } })),
    );
    const skill = makeSkill();
    const r = (await skill.sendDm({ participant_id: '999', text: 'hello' })) as {
      ok: true;
      data: { externalMessageId?: string };
      providerMessageId?: string;
    };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('evt-1');
    expect(r.providerMessageId).toBe('evt-1');
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://api.twitter.com/2/dm_conversations/with/999/messages');
    expect(init.method).toBe('POST');
    expect(init.body as string).toContain('"text":"hello"');
  });

  it('routes send_dm_in_conversation by dm_conversation_id', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ data: { dm_event_id: 'evt-2' } })),
    );
    const skill = makeSkill();
    const r = (await skill.sendDmInConversation({
      dm_conversation_id: 'conv-1',
      text: 'reply',
    })) as { ok: true; data: { externalMessageId?: string } };
    expect(r.ok).toBe(true);
    const [url] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://api.twitter.com/2/dm_conversations/conv-1/messages');
  });
});

describe('XSkill tweet publish', () => {
  it('refuses post_tweet without publish_posts capability', async () => {
    const skill = makeSkill({ enabledCapabilities: ['send_messages'] });
    const r = await skill.postTweet({ text: 'hello world' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', code: 'capability_disabled' });
  });

  it('rejects empty text', async () => {
    const skill = makeSkill();
    const r = await skill.postTweet({ text: '   ' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', message: 'text required' });
  });

  it('returns approval gate envelope when requirePostApproval is set', async () => {
    const skill = makeSkill({
      requirePostApproval: true,
      requestApproval: async () => ({ pending: true, draftId: 'draft-1' }),
    });
    const r = (await skill.postTweet({ text: 'hello' })) as {
      ok: true;
      data: { pending?: boolean; draftId?: string };
    };
    expect(r.ok).toBe(true);
    expect(r.data.pending).toBe(true);
    expect(r.data.draftId).toBe('draft-1');
  });

  it('POSTs to /2/tweets and returns the new id', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ data: { id: 'tw-1' } })),
    );
    const skill = makeSkill();
    const r = (await skill.postTweet({ text: 'hello world' })) as {
      ok: true;
      data: { externalMessageId?: string };
    };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('tw-1');
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://api.twitter.com/2/tweets');
    expect(init.method).toBe('POST');
  });

  it('chains thread tweets with reply.in_reply_to_tweet_id', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify({ data: { id: 't1' } })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ data: { id: 't2' } })));
    const skill = makeSkill();
    const r = (await skill.postThread({
      tweets: [{ text: 'first' }, { text: 'second' }],
    })) as { ok: true; data: { tweetIds?: string[] } };
    expect(r.ok).toBe(true);
    expect(r.data.tweetIds).toEqual(['t1', 't2']);
    const second = fetchSpy.mock.calls[1]![1] as RequestInit;
    expect(second.body as string).toContain('"in_reply_to_tweet_id":"t1"');
  });

  it('DELETEs tweet by id', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ data: { deleted: true } }), { status: 200 }),
    );
    const skill = makeSkill();
    const r = (await skill.deleteTweet({ tweet_id: 'tw-2' })) as {
      ok: true;
      data: { deleted?: boolean };
    };
    expect(r.ok).toBe(true);
    expect(r.data.deleted).toBe(true);
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(init.method).toBe('DELETE');
  });
});

describe('XSkill webhook endpoints', () => {
  it('CRC challenge returns sha256= base64 HMAC', async () => {
    const skill = makeSkill();
    const req = new Request('https://example.com/messaging/x/webhook?crc_token=abc');
    const res = await skill.crc(req);
    const json = (await res.json()) as { response_token: string };
    const expected = 'sha256=' + createHmac('sha256', 'secret-abc').update('abc').digest('base64');
    expect(json.response_token).toBe(expected);
  });

  it('webhook rejects bad signature', async () => {
    const skill = makeSkill();
    const body = '{"for_user_id":"1"}';
    const req = new Request('https://example.com/messaging/x/webhook', {
      method: 'POST',
      headers: { 'x-twitter-webhooks-signature': 'sha256=bad' },
      body,
    });
    const res = await skill.webhook(req);
    expect(res.status).toBe(403);
  });

  it('webhook accepts valid signature', async () => {
    const skill = makeSkill();
    const body = '{"for_user_id":"1"}';
    const sig = 'sha256=' + createHmac('sha256', 'secret-abc').update(body).digest('base64');
    const req = new Request('https://example.com/messaging/x/webhook', {
      method: 'POST',
      headers: { 'x-twitter-webhooks-signature': sig },
      body,
    });
    const res = await skill.webhook(req);
    expect(res.status).toBe(200);
  });
});
