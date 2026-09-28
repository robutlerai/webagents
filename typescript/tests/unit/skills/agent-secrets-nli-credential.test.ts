/**
 * The delegate fallback sends the platform credential to the platform's own
 * origin only (S-308, 2026-09-27, the agent-secrets lane): the origins are
 * the shared fixture's (`python/tests/fixtures/agent_secrets/delegate_credentials.json`),
 * and a non-platform https target receives neither `Authorization` nor the
 * caller's forwarded token, over HTTP and on the UAMP upgrade. The Python
 * suite runs the same in `tests/test_agent_secrets_nli_credential.py`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const captured = vi.hoisted(() => ({ configs: [] as Array<Record<string, unknown>> }));

vi.mock('../../../src/uamp/client.js', () => ({
  UAMPClient: vi.fn().mockImplementation((config: Record<string, unknown>) => {
    captured.configs.push(config);
    return {
      on: vi.fn(),
      connect: vi.fn(async () => {
        throw new Error('no socket in this test');
      }),
      sendEvents: vi.fn(),
      close: vi.fn(),
    };
  }),
}));

import { NLISkill } from '../../../src/skills/nli/skill';
import type { Context } from '../../../src/core/types';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/agent_secrets/delegate_credentials.json'), 'utf8')) as {
  platform_base: string;
  credential_headers: string[];
  cases: Array<{ name: string; target: string; sends: boolean }>;
};

const PLATFORM_TOKEN = 'platform-token-fixture';
const CALLER_TOKEN = 'caller-token-fixture';
const MESSAGES = [{ role: 'user' as const, content: 'hi' }];

function skill(): NLISkill {
  return new NLISkill({ baseUrl: FIXTURE.platform_base, transport: 'http', apiKey: PLATFORM_TOKEN, timeout: 5000 });
}

function callerContext(): Context {
  return {
    auth: { authenticated: true, scope: 'user' },
    metadata: { authToken: CALLER_TOKEN },
    get: () => undefined,
    signal: new AbortController().signal,
  } as unknown as Context;
}

const requests: Array<{ url: string; headers: Record<string, string> }> = [];
const originalFetch = globalThis.fetch;

beforeEach(() => {
  requests.length = 0;
  captured.configs.length = 0;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const headers: Record<string, string> = {};
    new Headers(init?.headers as HeadersInit).forEach((value, key) => {
      headers[key.toLowerCase()] = value;
    });
    requests.push({ url: String(input), headers });
    return new Response('data: [DONE]\n', { status: 200, headers: { 'content-type': 'text/event-stream' } });
  }) as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

async function drain(iterable: AsyncGenerator<string, void, unknown>): Promise<void> {
  try {
    for await (const _chunk of iterable) {
      // nothing to keep
    }
  } catch {
    // the UAMP fake refuses to connect; the headers were built before that
  }
}

function lower(names: string[]): string[] {
  return names.map((name) => name.toLowerCase());
}

describe('where the platform credential may go (the fixture’s cases)', () => {
  it.each(FIXTURE.cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    expect(skill().sendsCredentialTo(c.target)).toBe(c.sends);
  });

  it('a target that is not an absolute URL gets nothing', () => {
    expect(skill().sendsCredentialTo('')).toBe(false);
    expect(skill().sendsCredentialTo('not a url at all')).toBe(false);
  });
});

describe('the HTTP fallback', () => {
  it.each(FIXTURE.cases.map((c) => [c.name, c] as const))('%s', async (_name, c) => {
    await drain(skill().streamMessage(c.target, MESSAGES, callerContext()));
    expect(requests).toHaveLength(1);
    const sent = requests[0].headers;
    for (const header of lower(FIXTURE.credential_headers)) {
      if (c.sends && header === 'authorization') expect(sent[header]).toBe(`Bearer ${PLATFORM_TOKEN}`);
      else if (c.sends && header === 'x-forwarded-auth') expect(sent[header]).toBe(CALLER_TOKEN);
      else if (c.sends) expect(sent[header]).toBeUndefined(); // X-API-Key is the Python skill's header
      else expect(sent[header], header).toBeUndefined();
    }
    // What is not a credential goes as before.
    expect(sent['content-type']).toBe('application/json');
  });

  it('the payment token and the chat id still travel to another origin', async () => {
    const other = FIXTURE.cases.find((c) => !c.sends)!;
    await drain(skill().streamMessage(other.target, MESSAGES, callerContext(), 'payment-token-fixture', 'chat-1'));
    expect(requests[0].headers['x-payment-token']).toBe('payment-token-fixture');
    expect(requests[0].headers['x-chat-id']).toBe('chat-1');
    expect(requests[0].headers['authorization']).toBeUndefined();
  });
});

describe('the UAMP upgrade', () => {
  it.each(FIXTURE.cases.map((c) => [c.name, c] as const))('%s', async (_name, c) => {
    await drain(skill().streamMessageUAMP(c.target, MESSAGES, callerContext()));
    expect(captured.configs).toHaveLength(1);
    const headers = (captured.configs[0].headers ?? {}) as Record<string, string>;
    const names = Object.keys(headers).map((name) => name.toLowerCase());
    if (c.sends) expect(headers['Authorization']).toBe(`Bearer ${PLATFORM_TOKEN}`);
    for (const header of lower(FIXTURE.credential_headers)) {
      if (!(c.sends && header === 'authorization')) expect(names, header).not.toContain(header);
    }
  });
});
