/**
 * TelegramSkill — capability gating, fetch shape, bridge prompt, media
 * resolution (content_id, signed-URL-first with multipart-bytes fallback).
 *
 * Skills must stay DB-agnostic, so the test injects an in-memory
 * `getToken` resolver and a `fetch` mock to assert the wire shape
 * (sendMessage with the right body) without touching any portal code.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TelegramSkill } from '../../../../src/skills/messaging/telegram';
import type { ResolvedOutboundMedia } from '../../../../src/skills/messaging/shared';

const TOKEN = 'bot-test-token';
const UUID = 'd9f5bdaa-5055-4094-8233-c101f4cf782e';

const okJson = (result: unknown) =>
  Promise.resolve(
    new Response(JSON.stringify({ ok: true, result }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }),
  );

const errorJson = (status: number, description: string) =>
  Promise.resolve(
    new Response(JSON.stringify({ ok: false, description, error_code: status }), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }),
  );

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill(opts: Partial<ConstructorParameters<typeof TelegramSkill>[0]> = {}) {
  return new TelegramSkill({
    agentId: 'agent-1',
    integrationId: 'integ-1',
    enabledCapabilities: ['send_messages'],
    getToken: async () => ({ token: TOKEN, metadata: {} }),
    ...opts,
  });
}

describe('TelegramSkill', () => {
  it('refuses to send when capability is disabled', async () => {
    const skill = makeSkill({ enabledCapabilities: ['receive_messages'] });
    const r = await skill.sendText({ chat_id: '42', text: 'hi' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', code: 'capability_disabled' });
  });

  it('sends sendMessage with chat_id+text via fetch', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(() => okJson({ message_id: 7 }));
    const skill = makeSkill();
    const r = (await skill.sendText({ chat_id: '42', text: 'hi' })) as {
      ok: true;
      data: { message_id?: number };
      providerMessageId?: string;
    };
    expect(r.ok).toBe(true);
    expect(r.data).toEqual({ message_id: 7 });
    expect(r.providerMessageId).toBe('7');
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`https://api.telegram.org/bot${TOKEN}/sendMessage`);
    expect(init.method).toBe('POST');
    const sent = JSON.parse(init.body as string) as { chat_id: string; text: string };
    expect(sent.chat_id).toBe('42');
    expect(sent.text).toBe('hi');
  });

  it('uses bridge contactExternalId as recipient when chat_id omitted', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(() => okJson({ message_id: 1 }));
    const skill = makeSkill();
    await skill.sendText(
      { text: 'reply' },
      {
        metadata: {
          bridge: { source: 'telegram', contactExternalId: '999', contactDisplayName: 'Alice' },
        },
      } as never,
    );
    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit;
    expect(JSON.parse(init.body as string).chat_id).toBe('999');
  });

  it('returns invalid_input when no recipient and no bridge', async () => {
    const skill = makeSkill();
    const r = await skill.sendText({ text: 'no chat' });
    expect(r).toMatchObject({ ok: false, reason: 'invalid_input', message: 'chat_id required' });
  });

  it('bridgePrompt is null without a bridge ctx', async () => {
    const skill = makeSkill();
    const out = await skill.bridgePrompt(undefined);
    expect(out).toBeNull();
  });

  it('bridgePrompt renders contact name + send tool when bridge matches', async () => {
    const skill = makeSkill();
    const out = await skill.bridgePrompt({
      metadata: { bridge: { source: 'telegram', contactExternalId: '1', contactDisplayName: 'Bob' } },
    } as never);
    expect(out).toMatch(/telegram/i);
    expect(out).toMatch(/Bob/);
    expect(out).toMatch(/telegram_send_text/);
  });

  // ---------------------------------------------------------------------
  // sendPhoto media resolution — see plan
  //   webagents/typescript/src/skills/messaging/telegram/skill.ts
  // ---------------------------------------------------------------------

  describe('sendPhoto media resolution', () => {
    it('happy path: content_id → signed URL via JSON sendPhoto, no bytes fetch', async () => {
      const fetchSpy = vi
        .spyOn(globalThis, 'fetch')
        .mockImplementation(() => okJson({ message_id: 11 }));
      const fetchBytes = vi.fn();
      const resolveMediaForOutbound = vi.fn(async (id: string): Promise<ResolvedOutboundMedia> => {
        expect(id).toBe(UUID);
        return { url: `https://signed.example.com/api/content/${UUID}?sig=abc&exp=999`, fetchBytes };
      });
      const skill = makeSkill({ resolveMediaForOutbound });
      const r = (await skill.sendPhoto({ chat_id: '42', content_id: UUID, caption: 'pic' })) as {
        ok: true;
      };
      expect(r.ok).toBe(true);
      expect(resolveMediaForOutbound).toHaveBeenCalledTimes(1);
      expect(fetchBytes).not.toHaveBeenCalled();
      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe(`https://api.telegram.org/bot${TOKEN}/sendPhoto`);
      expect(init.method).toBe('POST');
      const sent = JSON.parse(init.body as string) as { chat_id: string; photo: string; caption: string };
      expect(sent.chat_id).toBe('42');
      expect(sent.photo).toBe(`https://signed.example.com/api/content/${UUID}?sig=abc&exp=999`);
      expect(sent.caption).toBe('pic');
    });

    it('relative /api/content/<uuid> photo_url is normalized through the resolver', async () => {
      const fetchSpy = vi
        .spyOn(globalThis, 'fetch')
        .mockImplementation(() => okJson({ message_id: 12 }));
      const resolveMediaForOutbound = vi.fn(async (id: string): Promise<ResolvedOutboundMedia> => {
        expect(id).toBe(UUID);
        return { url: `https://signed.example.com/x.jpg` };
      });
      const skill = makeSkill({ resolveMediaForOutbound });
      const r = (await skill.sendPhoto({
        chat_id: '42',
        photo_url: `/api/content/${UUID}`,
      })) as { ok: true };
      expect(r.ok).toBe(true);
      expect(resolveMediaForOutbound).toHaveBeenCalledWith(UUID);
      const init = fetchSpy.mock.calls[0]?.[1] as RequestInit;
      expect(JSON.parse(init.body as string).photo).toBe('https://signed.example.com/x.jpg');
    });

    it('absolute external https URL passes through with no resolver call', async () => {
      const fetchSpy = vi
        .spyOn(globalThis, 'fetch')
        .mockImplementation(() => okJson({ message_id: 13 }));
      const resolveMediaForOutbound = vi.fn();
      const skill = makeSkill({ resolveMediaForOutbound: resolveMediaForOutbound as never });
      const r = (await skill.sendPhoto({
        chat_id: '42',
        photo_url: 'https://example.com/cat.png',
      })) as { ok: true };
      expect(r.ok).toBe(true);
      expect(resolveMediaForOutbound).not.toHaveBeenCalled();
      const init = fetchSpy.mock.calls[0]?.[1] as RequestInit;
      expect(JSON.parse(init.body as string).photo).toBe('https://example.com/cat.png');
    });

    it('Telegram URL-fetch failure → SDK retries via multipart upload', async () => {
      const fetchSpy = vi
        .spyOn(globalThis, 'fetch')
        .mockImplementationOnce(() =>
          errorJson(400, 'Bad Request: failed to get HTTP URL content'),
        )
        .mockImplementationOnce(() => okJson({ message_id: 14 }));

      const fetchBytes = vi.fn(async () => ({
        buffer: new Uint8Array([1, 2, 3, 4, 5]),
        contentType: 'image/png',
        filename: 'cat.png',
      }));
      const resolveMediaForOutbound = vi.fn(
        async (): Promise<ResolvedOutboundMedia> => ({
          url: 'https://signed.example.com/x.png',
          fetchBytes,
        }),
      );
      const skill = makeSkill({ resolveMediaForOutbound });
      const r = (await skill.sendPhoto({ chat_id: '42', content_id: UUID, caption: 'pic' })) as {
        ok: true;
      };
      expect(r.ok).toBe(true);
      expect(fetchBytes).toHaveBeenCalledTimes(1);
      expect(fetchSpy).toHaveBeenCalledTimes(2);

      // First call: JSON URL attempt
      const firstInit = fetchSpy.mock.calls[0]?.[1] as RequestInit;
      expect(firstInit.headers).toMatchObject({ 'Content-Type': 'application/json' });

      // Second call: multipart upload (no JSON content-type, body is FormData)
      const secondInit = fetchSpy.mock.calls[1]?.[1] as RequestInit;
      // The fetch runtime fills in the multipart Content-Type with boundary;
      // we just assert we did NOT manually set application/json.
      const headers = secondInit.headers as Record<string, string> | undefined;
      expect(headers?.['Content-Type']).toBeUndefined();
      expect(secondInit.body).toBeInstanceOf(FormData);
      const form = secondInit.body as FormData;
      expect(form.get('chat_id')).toBe('42');
      expect(form.get('caption')).toBe('pic');
      const photo = form.get('photo');
      expect(photo).toBeInstanceOf(Blob);
      expect((photo as Blob).type).toBe('image/png');
      expect((photo as Blob).size).toBe(5);
    });

    it('Telegram non-URL error (e.g. 403) is NOT retried', async () => {
      const fetchSpy = vi
        .spyOn(globalThis, 'fetch')
        .mockImplementation(() => errorJson(403, 'Forbidden: bot was blocked by the user'));
      const fetchBytes = vi.fn();
      const resolveMediaForOutbound = vi.fn(
        async (): Promise<ResolvedOutboundMedia> => ({
          url: 'https://signed.example.com/x.png',
          fetchBytes,
        }),
      );
      const skill = makeSkill({ resolveMediaForOutbound });
      const r = (await skill.sendPhoto({ chat_id: '42', content_id: UUID })) as {
        ok: false;
        message: string;
      };
      expect(r.ok).toBe(false);
      expect(r.message).toMatch(/blocked by the user/);
      expect(fetchBytes).not.toHaveBeenCalled();
      expect(fetchSpy).toHaveBeenCalledTimes(1);
    });

    it('resolver returning null surfaces invalid_input without calling Telegram', async () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(() => okJson({}));
      const skill = makeSkill({ resolveMediaForOutbound: async () => null });
      const r = (await skill.sendPhoto({ chat_id: '42', content_id: UUID })) as {
        ok: false;
        reason: string;
        code?: string;
      };
      expect(r.ok).toBe(false);
      expect(r.reason).toBe('invalid_input');
      expect(r.code).toBe('content_not_found');
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it('URL-fetch failure with no fetchBytes returns the original error (no fallback available)', async () => {
      const fetchSpy = vi
        .spyOn(globalThis, 'fetch')
        .mockImplementation(() =>
          errorJson(400, 'Bad Request: wrong remote file identifier specified: Wrong string length'),
        );
      const skill = makeSkill({
        resolveMediaForOutbound: async () => ({ url: 'https://signed.example.com/x.png' }),
      });
      const r = (await skill.sendPhoto({ chat_id: '42', content_id: UUID })) as {
        ok: false;
        message: string;
      };
      expect(r.ok).toBe(false);
      expect(r.message).toMatch(/wrong remote file identifier/i);
      expect(fetchSpy).toHaveBeenCalledTimes(1);
    });

    it('rejects when neither content_id, photo_url, nor a bridge contact yields a usable reference', async () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(() => okJson({}));
      const skill = makeSkill();
      const r = (await skill.sendPhoto({ chat_id: '42' })) as { ok: false; code?: string };
      expect(r.ok).toBe(false);
      expect(r.code).toBe('invalid_media_reference');
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it('rejects content_id when no resolver is configured (standalone host)', async () => {
      const skill = makeSkill();
      const r = (await skill.sendPhoto({ chat_id: '42', content_id: UUID })) as {
        ok: false;
        code?: string;
      };
      expect(r.ok).toBe(false);
      expect(r.code).toBe('media_resolver_unavailable');
    });
  });

  describe('sendDocument media resolution', () => {
    it('content_id resolves through the same path and uploads via multipart on URL failure', async () => {
      const fetchSpy = vi
        .spyOn(globalThis, 'fetch')
        .mockImplementationOnce(() =>
          errorJson(400, 'Bad Request: failed to get HTTP URL content'),
        )
        .mockImplementationOnce(() => okJson({ message_id: 21 }));

      const fetchBytes = vi.fn(async () => ({
        buffer: new Uint8Array([0xff, 0xd8, 0xff]),
        contentType: 'application/pdf',
        filename: 'report.pdf',
      }));
      const skill = makeSkill({
        resolveMediaForOutbound: async () => ({ url: 'https://signed.example.com/r.pdf', fetchBytes }),
      });
      const r = (await skill.sendDocument({ chat_id: '42', content_id: UUID })) as { ok: true };
      expect(r.ok).toBe(true);
      expect(fetchSpy).toHaveBeenCalledTimes(2);
      const [url] = fetchSpy.mock.calls[0] as [string];
      expect(url).toBe(`https://api.telegram.org/bot${TOKEN}/sendDocument`);
      const second = fetchSpy.mock.calls[1]?.[1] as RequestInit;
      const form = second.body as FormData;
      expect(form.get('document')).toBeInstanceOf(Blob);
      expect((form.get('document') as Blob).type).toBe('application/pdf');
    });
  });
});
