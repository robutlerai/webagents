/**
 * RedditSkill — capability gating, User-Agent header, submit shape.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { RedditSkill } from '../../../../src/skills/messaging/reddit';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof RedditSkill>[0]> = {}) {
  return new RedditSkill({
    agentId: 'agent-1',
    enabledCapabilities: ['publish_posts'],
    getToken: async () => ({ token: 'reddit-token', metadata: { userAgent: 'test/1.0' } }),
    ...opts,
  });
}

describe('RedditSkill', () => {
  it('refuses without publish capability', async () => {
    const skill = makeSkill({ enabledCapabilities: ['some_other'] });
    const r = await skill.post({ subreddit: 'test', title: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', code: 'capability_disabled' });
  });

  it('POSTs /api/submit with kind=self and User-Agent', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({ json: { data: { id: 'abc', url: 'https://reddit.com/r/test/abc' }, errors: [] } }),
      ),
    );
    const skill = makeSkill();
    const r = (await skill.post({ subreddit: 'test', title: 'hi', text: 'body' })) as {
      ok: true;
      data: { externalMessageId?: string; url?: string };
    };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('abc');
    expect(r.data.url).toBe('https://reddit.com/r/test/abc');
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://oauth.reddit.com/api/submit');
    expect((init.headers as Record<string, string>)['User-Agent']).toBe('test/1.0');
    expect(init.body as string).toMatch(/kind=self/);
  });
});
