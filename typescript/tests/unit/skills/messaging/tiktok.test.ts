/**
 * TikTokSkill — capability gating, init+poll publish flow, approval gate.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TikTokSkill } from '../../../../src/skills/messaging/tiktok';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof TikTokSkill>[0]> = {}) {
  return new TikTokSkill({
    agentId: 'agent-1',
    integrationId: 'integ-1',
    enabledCapabilities: ['publish_posts'],
    getToken: async () => ({
      token: 'tiktok-token',
      metadata: { openId: 'open-id-abc' },
    }),
    ...opts,
  });
}

describe('TikTokSkill.publishVideo', () => {
  it('refuses without publish_posts capability', async () => {
    const skill = makeSkill({ enabledCapabilities: ['unrelated'] });
    const r = await skill.publishVideo({ videoUrl: 'https://x/y.mp4' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', code: 'capability_disabled' });
  });

  it('rejects missing videoUrl', async () => {
    const skill = makeSkill();
    const r = await skill.publishVideo({ videoUrl: '' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', message: 'videoUrl required' });
  });

  it('returns approval-gate envelope when requirePostApproval is set', async () => {
    const skill = makeSkill({
      requirePostApproval: true,
      requestApproval: async () => ({ pending: true, draftId: 'draft-1' }),
    });
    const r = (await skill.publishVideo({
      videoUrl: 'https://x/y.mp4',
    })) as { ok: true; data: { pending?: boolean; draftId?: string } };
    expect(r.ok).toBe(true);
    expect(r.data.pending).toBe(true);
  });

  it('initiates publish via PULL_FROM_URL and polls until PUBLISH_COMPLETE', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ data: { publish_id: 'pub-1' } })),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            data: {
              status: 'PUBLISH_COMPLETE',
              publicaly_available_post_id: ['post-99'],
            },
          }),
        ),
      );
    const skill = makeSkill();
    const r = (await skill.publishVideo({
      videoUrl: 'https://x/y.mp4',
      caption: 'hello',
    })) as {
      ok: true;
      data: { externalMessageId?: string; publishedPostIds?: string[] };
    };
    expect(r.ok).toBe(true);
    expect(r.data.externalMessageId).toBe('pub-1');
    expect(r.data.publishedPostIds).toEqual(['post-99']);
    const [initUrl, initInit] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(initUrl).toBe('https://open.tiktokapis.com/v2/post/publish/video/init/');
    const body = JSON.parse(initInit.body as string) as {
      source_info: { source: string; video_url: string };
    };
    expect(body.source_info.source).toBe('PULL_FROM_URL');
    expect(body.source_info.video_url).toBe('https://x/y.mp4');
  });
});

describe('TikTokSkill.publishPhoto', () => {
  it('rejects empty photoUrls', async () => {
    const skill = makeSkill();
    const r = await skill.publishPhoto({ photoUrls: [] });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', message: 'photoUrls required' });
  });

  it('rejects out-of-range photoCoverIndex', async () => {
    const skill = makeSkill();
    const r = await skill.publishPhoto({
      photoUrls: ['https://x/a.jpg', 'https://x/b.jpg'],
      photoCoverIndex: 5,
    });
    expect(r).toMatchObject({
      ok: false,
      reason: 'invalid_input',
      message: 'photoCoverIndex out of range',
    });
  });

  it('sends photo_cover_index (required by TikTok API), defaulting to 0', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ data: { publish_id: 'pub-photo-1' } })),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            data: {
              status: 'PUBLISH_COMPLETE',
              publicaly_available_post_id: ['photo-99'],
            },
          }),
        ),
      );
    const skill = makeSkill();
    const r = (await skill.publishPhoto({
      photoUrls: ['https://x/a.jpg', 'https://x/b.jpg'],
      caption: 'hi',
    })) as { ok: true; data: { externalMessageId?: string } };
    expect(r.ok).toBe(true);
    const [initUrl, initInit] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(initUrl).toBe('https://open.tiktokapis.com/v2/post/publish/content/init/');
    const body = JSON.parse(initInit.body as string) as {
      post_mode: string;
      media_type: string;
      source_info: { photo_images: string[]; photo_cover_index: number };
    };
    expect(body.post_mode).toBe('DIRECT_POST');
    expect(body.media_type).toBe('PHOTO');
    expect(body.source_info.photo_images).toEqual(['https://x/a.jpg', 'https://x/b.jpg']);
    expect(body.source_info.photo_cover_index).toBe(0);
  });

  it('honors an explicit photoCoverIndex', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ data: { publish_id: 'pub-photo-2' } })),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ data: { status: 'PUBLISH_COMPLETE' } })),
      );
    const skill = makeSkill();
    await skill.publishPhoto({
      photoUrls: ['https://x/a.jpg', 'https://x/b.jpg', 'https://x/c.jpg'],
      photoCoverIndex: 2,
    });
    const [, initInit] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(initInit.body as string) as {
      source_info: { photo_cover_index: number };
    };
    expect(body.source_info.photo_cover_index).toBe(2);
  });
});
