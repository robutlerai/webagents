/**
 * BlueskySkill — capability gating, AT Proto record shape.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BlueskySkill } from '../../../../src/skills/messaging/bluesky';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof BlueskySkill>[0]> = {}) {
  return new BlueskySkill({
    agentId: 'agent-1',
    enabledCapabilities: ['publish_posts'],
    getToken: async () => ({ token: 'jwt', metadata: { did: 'did:plc:abc' } }),
    ...opts,
  });
}

describe('BlueskySkill', () => {
  it('rejects without DID metadata', async () => {
    const skill = new BlueskySkill({
      agentId: 'a',
      enabledCapabilities: ['publish_posts'],
      getToken: async () => ({ token: 'jwt', metadata: {} }),
    });
    const r = await skill.post({ text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'provider_api_error', message: 'bluesky_did_missing' });
  });

  it('POSTs createRecord with app.bsky.feed.post collection', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(
        new Response(JSON.stringify({ uri: 'at://did:plc:abc/app.bsky.feed.post/1', cid: 'C' })),
      );
    const skill = makeSkill();
    const r = (await skill.post({ text: 'hi' })) as {
      ok: true;
      data: { externalMessageId?: string };
    };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toMatch(/^at:\/\//);
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://bsky.social/xrpc/com.atproto.repo.createRecord');
    const body = JSON.parse(init.body as string) as { collection: string; repo: string };
    expect(body.collection).toBe('app.bsky.feed.post');
    expect(body.repo).toBe('did:plc:abc');
  });
});
