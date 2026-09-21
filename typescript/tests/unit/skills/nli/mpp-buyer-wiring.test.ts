/**
 * The NLI skill's buyer wiring (machine-purchase design section 6.4; pass
 * P9a, 2026-09-18): with `mppBuyer` set, a delegate over HTTP goes through
 * the buyer's `payingFetch` instead of the bare `fetch`, the UAMP client is
 * built with the buyer as its in-band purchase hook, and the model-facing
 * failure-mode prompt stops telling the model not to retry a 402 (the
 * retry already happened inside the operator's policy). Without a buyer
 * every one of those is exactly what it was.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NLISkill, type NLIMppBuyer } from '../../../../src/skills/nli/skill.js';
import type { Context } from '../../../../src/core/types.js';

function makeContext(): Context {
  const store = new Map<string, unknown>();
  return {
    get: vi.fn((key: string) => store.get(key)),
    set: vi.fn((key: string, value: unknown) => store.set(key, value)),
    delete: vi.fn((key: string) => store.delete(key)),
    signal: undefined,
    auth: { authenticated: false },
    payment: { token: undefined },
    metadata: {},
  } as unknown as Context;
}

function sse(content: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const lines = [`data: {"choices":[{"delta":{"content":"${content}"}}]}\n\n`, 'data: [DONE]\n\n'];
  return new ReadableStream({
    start(controller) {
      for (const line of lines) controller.enqueue(encoder.encode(line));
      controller.close();
    },
  });
}

function stubBuyer(): NLIMppBuyer & { payingFetch: ReturnType<typeof vi.fn>; purchase: ReturnType<typeof vi.fn>; purchaseAt: ReturnType<typeof vi.fn> } {
  return {
    payingFetch: vi.fn(async () => new Response(sse('bought'), { status: 200 })),
    purchase: vi.fn(async () => ({ ok: true })),
    // 2026-09-19: the purchase pointer (an `mpp` entry with no challenge) is a buyer's to follow.
    purchaseAt: vi.fn(async () => ({ ok: true })),
  };
}

async function collect(gen: AsyncGenerator<string, void, unknown>): Promise<string> {
  let out = '';
  for await (const chunk of gen) out += chunk;
  return out;
}

const mockFetch = vi.fn();

describe('NLI skill and the MPP buyer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal('fetch', mockFetch);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('the failure-mode prompt tells the model not to auto-retry a 402 only when no buyer is configured', () => {
    const plain = new NLISkill({}).nliFailureModes(makeContext());
    expect(plain).toContain('Do NOT auto-retry');
    expect(plain).not.toContain('purchase policy');

    const bought = new NLISkill({ mppBuyer: stubBuyer() }).nliFailureModes(makeContext());
    expect(bought).not.toContain('Do NOT auto-retry');
    expect(bought).toContain('402 / payment_required');
    expect(bought).toContain('purchase policy');
    expect(bought).toContain('re-sent the request once');
    // The 402 line still ends in an instruction not to retry from the model's side.
    expect(bought).toMatch(/do NOT retry, the operator must change the policy or the funding/);
    // Copy rule: usage is bought from Robutler; no agent or creator is paid.
    expect(bought).not.toMatch(/pay (the|an) (agent|creator)/i);
  });

  it('a delegate over HTTP goes through the buyer, with the same method, URL, body and headers it would have fetched with', async () => {
    const buyer = stubBuyer();
    const skill = new NLISkill({ transport: 'http', mppBuyer: buyer, timeout: 5000 });
    const text = await collect(skill.streamMessage('https://portal.example.com/agents/acme', [{ role: 'user', content: 'hi' }], makeContext()));
    expect(text).toBe('bought');
    expect(mockFetch).not.toHaveBeenCalled();
    expect(buyer.payingFetch).toHaveBeenCalledTimes(1);
    const [url, init] = buyer.payingFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://portal.example.com/agents/acme/chat/completions');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ messages: [{ role: 'user', content: 'hi' }], stream: true });
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  // 2026-09-18: the review found the buyer posting to
  // `/agents/{name}/chat/completions`, which the portal serves with no
  // handler (a redirect to the profile page), while its agent HTTP door
  // prices only `/v1/chat/completions`. This test used to pin the dead URL.
  it('a platform agent is posted at /v1/chat/completions, the route the portal door prices', async () => {
    const onList = { ...stubBuyer(), allowsUrl: vi.fn((u: string) => new URL(u).host === 'robutler.ai') };
    const skill = new NLISkill({ transport: 'http', mppBuyer: onList, timeout: 5000 });
    await collect(skill.streamMessage('https://robutler.ai/agents/acme', [{ role: 'user', content: 'hi' }], makeContext()));
    expect(onList.payingFetch.mock.calls[0][0]).toBe('https://robutler.ai/agents/acme/v1/chat/completions');
    // A host off the buyer's list keeps the path every SDK server serves.
    await collect(skill.streamMessage('https://other.example/agents/acme', [{ role: 'user', content: 'hi' }], makeContext()));
    expect(onList.payingFetch.mock.calls[1][0]).toBe('https://other.example/agents/acme/chat/completions');

    // A buyer without `allowsUrl` falls back to the skill's own baseUrl.
    const plainBuyer = stubBuyer();
    const byBase = new NLISkill({ transport: 'http', mppBuyer: plainBuyer, baseUrl: 'https://robutler.ai', timeout: 5000 });
    await collect(byBase.streamMessage(byBase.normalizeUrl('@acme'), [{ role: 'user', content: 'hi' }], makeContext()));
    expect(plainBuyer.payingFetch.mock.calls[0][0]).toBe('https://robutler.ai/agents/acme/v1/chat/completions');
  });

  it('without a buyer the HTTP delegate is the bare fetch it always was', async () => {
    mockFetch.mockResolvedValue({ ok: true, body: sse('plain') });
    const skill = new NLISkill({ transport: 'http', timeout: 5000 });
    const text = await collect(skill.streamMessage('https://portal.example.com/agents/acme', [{ role: 'user', content: 'hi' }], makeContext()));
    expect(text).toBe('plain');
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('a 402 the buyer could not fund still fails the delegate closed, as a plain 402 did', async () => {
    const buyer = stubBuyer();
    buyer.payingFetch.mockResolvedValue(new Response('{"status":402}', { status: 402, statusText: 'Payment Required' }));
    const skill = new NLISkill({ transport: 'http', mppBuyer: buyer, timeout: 5000 });
    await expect(collect(skill.streamMessage('https://portal.example.com/agents/acme', [{ role: 'user', content: 'hi' }], makeContext()))).rejects.toThrow(/402/);
  });

  it('the UAMP client is built with the buyer as its in-band purchase hook', async () => {
    const configs: unknown[] = [];
    vi.doMock('../../../../src/uamp/client.js', async (importOriginal) => {
      const mod = await importOriginal<typeof import('../../../../src/uamp/client.js')>();
      return {
        ...mod,
        UAMPClient: class extends mod.UAMPClient {
          constructor(config: ConstructorParameters<typeof mod.UAMPClient>[0]) {
            super(config);
            configs.push(config);
            throw new Error('stop before any socket');
          }
        },
      };
    });
    vi.resetModules();
    const { NLISkill: Patched } = await import('../../../../src/skills/nli/skill.js');
    const buyer = stubBuyer();
    const skill = new Patched({ transport: 'uamp', mppBuyer: buyer, timeout: 5000 });
    await expect(collect(skill.streamMessage('https://portal.example.com/agents/acme', [{ role: 'user', content: 'hi' }], makeContext()))).rejects.toThrow('stop before any socket');
    expect(configs).toHaveLength(1);
    expect((configs[0] as { buyer?: unknown }).buyer).toBe(buyer);
    // The same object, so the client can hand it a purchase pointer as well as a challenge.
    expect(typeof (configs[0] as { buyer: NLIMppBuyer }).buyer.purchaseAt).toBe('function');

    const plain = new Patched({ transport: 'uamp', timeout: 5000 });
    await expect(collect(plain.streamMessage('https://portal.example.com/agents/acme', [{ role: 'user', content: 'hi' }], makeContext()))).rejects.toThrow('stop before any socket');
    expect('buyer' in (configs[1] as object)).toBe(false);
    vi.doUnmock('../../../../src/uamp/client.js');
  });
});
