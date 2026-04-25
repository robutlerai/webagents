/**
 * LinkedInSkill — capability gating, approval gate short-circuit,
 * Posts API (rest/posts) payload shape.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { LinkedInSkill } from '../../../../src/skills/messaging/linkedin';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof LinkedInSkill>[0]> = {}) {
  return new LinkedInSkill({
    agentId: 'agent-1',
    enabledCapabilities: ['publish_posts'],
    getToken: async () => ({
      token: 'li-token',
      metadata: { personUrn: 'urn:li:person:abc' },
    }),
    ...opts,
  });
}

describe('LinkedInSkill', () => {
  it('refuses without publish_posts capability', async () => {
    const skill = makeSkill({ enabledCapabilities: ['some_other_capability'] });
    const r = await skill.post({ text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', code: 'capability_disabled' });
  });

  it('routes to approval gate when configured', async () => {
    const requestApproval = vi.fn().mockResolvedValue({ pending: true, draftId: 'draft-1' });
    const skill = makeSkill({ requirePostApproval: true, requestApproval });
    const r = (await skill.post({ text: 'hello' })) as {
      ok: true;
      data: { pending?: true; draftId?: string };
    };
    expect(r.ok).toBe(true);
    expect(r.data.pending).toBe(true);
    expect(r.data.draftId).toBe('draft-1');
    expect(requestApproval).toHaveBeenCalledOnce();
  });

  it('publishes via Posts API (rest/posts) with the LinkedIn-Version header', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('', {
        status: 201,
        headers: { 'x-restli-id': 'urn:li:share:1' },
      }),
    );
    const skill = makeSkill();
    const r = (await skill.post({ text: 'hello' })) as {
      ok: true;
      data: { externalMessageId?: string };
    };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('urn:li:share:1');
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://api.linkedin.com/rest/posts');
    const headers = init.headers as Record<string, string>;
    expect(headers['LinkedIn-Version']).toMatch(/^\d{6}$/);
    expect(headers['X-Restli-Protocol-Version']).toBe('2.0.0');
    const body = JSON.parse(init.body as string) as {
      author: string;
      commentary: string;
      lifecycleState: string;
      visibility: string;
      distribution: { feedDistribution: string };
    };
    expect(body.author).toBe('urn:li:person:abc');
    expect(body.commentary).toBe('hello');
    expect(body.lifecycleState).toBe('PUBLISHED');
    expect(body.visibility).toBe('PUBLIC');
    expect(body.distribution.feedDistribution).toBe('MAIN_FEED');
  });

  it('embeds an article block when a link is supplied', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('', {
        status: 201,
        headers: { 'x-restli-id': 'urn:li:share:2' },
      }),
    );
    const skill = makeSkill();
    await skill.post({
      text: 'check this out',
      link: 'https://example.com/article',
      link_title: 'Example title',
      link_description: 'Example description',
    });
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      content?: { article?: { source: string; title?: string; description?: string } };
    };
    expect(body.content?.article?.source).toBe('https://example.com/article');
    expect(body.content?.article?.title).toBe('Example title');
    expect(body.content?.article?.description).toBe('Example description');
  });

  it('honors metadata.linkedinApiVersion override', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response('', { status: 201, headers: { 'x-restli-id': 'urn:li:share:3' } }));
    const skill = makeSkill({
      getToken: async () => ({
        token: 'li-token',
        metadata: { personUrn: 'urn:li:person:abc', linkedinApiVersion: '202612' },
      }),
    });
    await skill.post({ text: 'pin a future version' });
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers['LinkedIn-Version']).toBe('202612');
  });

  it('errors when as_organization is requested but no org urn is set', async () => {
    const skill = makeSkill();
    const r = await skill.post({ text: 'hi', as_organization: true });
    expect(r).toMatchObject({ ok: false, reason: 'provider_api_error', message: 'linkedin_author_urn_missing' });
  });
});
