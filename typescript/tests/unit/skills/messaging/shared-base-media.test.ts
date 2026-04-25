import { describe, expect, it, vi } from 'vitest';

import { MessagingSkill } from '../../../../src/skills/messaging/shared/messaging-skill-base';
import type {
  ApiCallResult,
  OutboundMediaResolver,
  ResolvedOutboundMedia,
} from '../../../../src/skills/messaging/shared/options';

const VALID_UUID = '11111111-2222-4333-8444-555555555555';

class TestSkill extends MessagingSkill {
  readonly provider = 'test';
  constructor(opts: ConstructorParameters<typeof MessagingSkill>[1] = {}) {
    super('test', opts);
  }

  callResolveOutboundMedia(input: { contentId?: string | null; url?: string | null }) {
    return this.resolveOutboundMedia(input);
  }

  callSendMediaWithFallback<T>(input: {
    callType: string;
    media: ResolvedOutboundMedia;
    sendByUrl: (url: string) => Promise<ApiCallResult<T>>;
    sendByBytes?: (bytes: {
      buffer: Uint8Array;
      contentType: string;
      filename?: string;
    }) => Promise<ApiCallResult<T>>;
    extraUrlFailurePatterns?: RegExp[];
  }) {
    return this.sendMediaWithFallback<T>(input);
  }
}

describe('MessagingSkill base — resolveOutboundMedia', () => {
  it('passes through absolute https URLs without invoking the resolver', async () => {
    const resolver = vi.fn();
    const skill = new TestSkill({ resolveMediaForOutbound: resolver });

    const out = await skill.callResolveOutboundMedia({ url: 'https://cdn.example/x.jpg' });
    expect(out).toEqual({ media: { url: 'https://cdn.example/x.jpg' } });
    expect(resolver).not.toHaveBeenCalled();
  });

  it('passes through absolute http URLs (legacy hosts)', async () => {
    const skill = new TestSkill();
    const out = await skill.callResolveOutboundMedia({ url: 'http://legacy.example/x.jpg' });
    expect(out).toEqual({ media: { url: 'http://legacy.example/x.jpg' } });
  });

  it('returns invalid_input when neither contentId nor url is provided', async () => {
    const skill = new TestSkill();
    const out = await skill.callResolveOutboundMedia({});
    expect('error' in out).toBe(true);
    if ('error' in out) {
      expect(out.error.ok).toBe(false);
      if (!out.error.ok) {
        expect(out.error.reason).toBe('invalid_input');
        expect(out.error.code).toBe('invalid_media_reference');
      }
    }
  });

  it('returns invalid_input when only a relative URL with no UUID is given', async () => {
    const skill = new TestSkill();
    const out = await skill.callResolveOutboundMedia({ url: '/some/relative/path.jpg' });
    expect('error' in out).toBe(true);
    if ('error' in out) {
      expect(out.error.ok).toBe(false);
      if (!out.error.ok) {
        expect(out.error.code).toBe('invalid_media_reference');
      }
    }
  });

  it('returns invalid_input when the host has no resolver configured', async () => {
    const skill = new TestSkill();
    const out = await skill.callResolveOutboundMedia({ contentId: VALID_UUID });
    expect('error' in out).toBe(true);
    if ('error' in out) {
      expect(out.error.ok).toBe(false);
      if (!out.error.ok) {
        expect(out.error.code).toBe('media_resolver_unavailable');
      }
    }
  });

  it('invokes the resolver and returns its media when it resolves', async () => {
    const resolved: ResolvedOutboundMedia = { url: 'https://cdn.example/signed/abc' };
    const resolver: OutboundMediaResolver = vi.fn().mockResolvedValue(resolved);
    const skill = new TestSkill({ resolveMediaForOutbound: resolver });

    const out = await skill.callResolveOutboundMedia({ contentId: VALID_UUID });
    expect(out).toEqual({ media: resolved });
    expect(resolver).toHaveBeenCalledWith(VALID_UUID);
  });

  it('extracts the UUID from a relative /api/content/<uuid> URL', async () => {
    const resolver: OutboundMediaResolver = vi
      .fn()
      .mockResolvedValue({ url: 'https://cdn.example/x' });
    const skill = new TestSkill({ resolveMediaForOutbound: resolver });

    await skill.callResolveOutboundMedia({ url: `/api/content/${VALID_UUID}` });
    expect(resolver).toHaveBeenCalledWith(VALID_UUID);
  });

  it('returns content_not_found when the resolver returns null', async () => {
    const resolver: OutboundMediaResolver = vi.fn().mockResolvedValue(null);
    const skill = new TestSkill({ resolveMediaForOutbound: resolver });

    const out = await skill.callResolveOutboundMedia({ contentId: VALID_UUID });
    expect('error' in out).toBe(true);
    if ('error' in out) {
      expect(out.error.ok).toBe(false);
      if (!out.error.ok) {
        expect(out.error.code).toBe('content_not_found');
      }
    }
  });
});

describe('MessagingSkill base — sendMediaWithFallback', () => {
  it('returns the URL attempt when it succeeds, without consulting bytes', async () => {
    const skill = new TestSkill();
    const sendByUrl = vi.fn().mockResolvedValue({ ok: true, data: { id: '1' } });
    const sendByBytes = vi.fn();
    const fetchBytes = vi.fn();

    const out = await skill.callSendMediaWithFallback({
      callType: 'send_photo',
      media: { url: 'https://cdn.example/x.jpg', fetchBytes },
      sendByUrl,
      sendByBytes,
    });
    expect(out).toEqual({ ok: true, data: { id: '1' } });
    expect(sendByBytes).not.toHaveBeenCalled();
    expect(fetchBytes).not.toHaveBeenCalled();
  });

  it('returns the original failure when the failure is not a URL-fetch failure', async () => {
    const skill = new TestSkill();
    const failure = {
      ok: false,
      retriable: false,
      reason: 'provider_api_error' as const,
      message: 'Forbidden: bot was blocked',
    };
    const sendByUrl = vi.fn().mockResolvedValue(failure);
    const sendByBytes = vi.fn();
    const fetchBytes = vi.fn();

    const out = await skill.callSendMediaWithFallback({
      callType: 'send_photo',
      media: { url: 'https://cdn.example/x.jpg', fetchBytes },
      sendByUrl,
      sendByBytes,
    });
    expect(out).toEqual(failure);
    expect(sendByBytes).not.toHaveBeenCalled();
    expect(fetchBytes).not.toHaveBeenCalled();
  });

  it('returns the original failure when fetchBytes is not provided', async () => {
    const skill = new TestSkill();
    const failure = {
      ok: false,
      retriable: false,
      reason: 'provider_api_error' as const,
      message: 'Bad Request: failed to get HTTP URL content',
    };
    const sendByUrl = vi.fn().mockResolvedValue(failure);
    const sendByBytes = vi.fn();

    const out = await skill.callSendMediaWithFallback({
      callType: 'send_photo',
      media: { url: 'https://cdn.example/x.jpg' },
      sendByUrl,
      sendByBytes,
    });
    expect(out).toEqual(failure);
    expect(sendByBytes).not.toHaveBeenCalled();
  });

  it('falls back to sendByBytes on a URL-fetch failure when both fetchBytes and sendByBytes are present', async () => {
    const skill = new TestSkill();
    const sendByUrl = vi.fn().mockResolvedValue({
      ok: false,
      retriable: false,
      reason: 'provider_api_error',
      message: 'Bad Request: failed to get HTTP URL content',
    });
    const bytes = {
      buffer: new Uint8Array([1, 2, 3]),
      contentType: 'image/png',
      filename: 'x.png',
    };
    const fetchBytes = vi.fn().mockResolvedValue(bytes);
    const sendByBytes = vi.fn().mockResolvedValue({ ok: true, data: { id: '2' } });

    const out = await skill.callSendMediaWithFallback({
      callType: 'send_photo',
      media: { url: 'https://cdn.example/x.jpg', fetchBytes },
      sendByUrl,
      sendByBytes,
    });
    expect(out).toEqual({ ok: true, data: { id: '2' } });
    expect(fetchBytes).toHaveBeenCalledTimes(1);
    expect(sendByBytes).toHaveBeenCalledWith(bytes);
  });

  it('honours extraUrlFailurePatterns', async () => {
    const skill = new TestSkill();
    const sendByUrl = vi.fn().mockResolvedValue({
      ok: false,
      retriable: false,
      reason: 'provider_api_error',
      message: 'Failed to download media from URL',
    });
    const fetchBytes = vi
      .fn()
      .mockResolvedValue({ buffer: new Uint8Array([7]), contentType: 'image/png' });
    const sendByBytes = vi.fn().mockResolvedValue({ ok: true, data: { id: '3' } });

    const out = await skill.callSendMediaWithFallback({
      callType: 'send_photo',
      media: { url: 'https://cdn.example/x.jpg', fetchBytes },
      sendByUrl,
      sendByBytes,
      extraUrlFailurePatterns: [/failed to download/i],
    });
    expect(out).toEqual({ ok: true, data: { id: '3' } });
  });

  it('returns the original URL failure when fetchBytes throws', async () => {
    const skill = new TestSkill();
    const failure = {
      ok: false,
      retriable: false,
      reason: 'provider_api_error' as const,
      message: 'Bad Request: failed to get HTTP URL content',
    };
    const sendByUrl = vi.fn().mockResolvedValue(failure);
    const fetchBytes = vi.fn().mockRejectedValue(new Error('disk read error'));
    const sendByBytes = vi.fn();

    const out = await skill.callSendMediaWithFallback({
      callType: 'send_photo',
      media: { url: 'https://cdn.example/x.jpg', fetchBytes },
      sendByUrl,
      sendByBytes,
    });
    expect(out).toEqual(failure);
    expect(sendByBytes).not.toHaveBeenCalled();
  });
});
