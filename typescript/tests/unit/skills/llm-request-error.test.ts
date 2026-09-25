/**
 * A failed model request says which server and why (2026-09-24,
 * `src/skills/llm/request.ts`). Node's fetch rejects with "fetch failed" and
 * keeps the reason in `cause`, so a mistyped OPENAI_BASE_URL reached the
 * person as "Error: fetch failed" and nothing else.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';

import { describeRequestError, fetchModel, setRequestErrorDetail } from '../../../src/skills/llm/request';

afterEach(() => {
  vi.unstubAllGlobals();
  setRequestErrorDetail(false);
});

function refused(): TypeError {
  const cause = Object.assign(new Error('connect ECONNREFUSED 127.0.0.1:9'), { code: 'ECONNREFUSED' });
  return new TypeError('fetch failed', { cause });
}

describe('a failed model request', () => {
  it('names the server and the reason, not just "fetch failed"', () => {
    expect(describeRequestError(refused(), 'http://127.0.0.1:9/v1/chat/completions')).toBe(
      'Could not reach http://127.0.0.1:9: connect ECONNREFUSED 127.0.0.1:9',
    );
  });

  it('names only the origin, since a path or query can carry a key', () => {
    const text = describeRequestError(refused(), 'https://gateway.example/v1/sk-secret/chat?key=secret');
    expect(text).toContain('https://gateway.example');
    expect(text).not.toContain('secret');
  });

  it('reads a refused dual-stack connection, whose message is empty', () => {
    const cause = Object.assign(new AggregateError([new Error('connect ECONNREFUSED ::1:9')], ''), { code: 'ECONNREFUSED' });
    const error = new TypeError('fetch failed', { cause });
    expect(describeRequestError(error, 'http://localhost:9/v1')).toBe(
      'Could not reach http://localhost:9: connect ECONNREFUSED ::1:9',
    );
  });

  it('keeps the message when there is no cause', () => {
    expect(describeRequestError(new Error('boom'))).toBe('boom');
  });

  it('says more only at the terminal: a server sends the message to its callers (S-228)', async () => {
    const error = refused();
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(error));
    await expect(fetchModel('http://10.0.0.5:4000/v1', {})).rejects.toBe(error);
  });

  it('at the terminal, wraps a failure but leaves a cancelled request alone', async () => {
    setRequestErrorDetail(true);
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(refused()));
    await expect(fetchModel('http://127.0.0.1:9/v1', {})).rejects.toThrow('Could not reach http://127.0.0.1:9');

    const abort = Object.assign(new Error('This operation was aborted'), { name: 'AbortError' });
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(abort));
    await expect(fetchModel('http://127.0.0.1:9/v1', {})).rejects.toBe(abort);
  });
});
