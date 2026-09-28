/**
 * `serve` answers a refused chat completion with its status and a JSON error,
 * streaming or not (the ptypass-fixes lane, 2026-09-27, brief item 14; a
 * chat-fixes leftover). The Python server's `_refusal` (S-236) already did:
 * a `PaymentRequiredError` thrown by the run came back here as 500
 * `completions_error`, so a caller could not tell "pay first" from a crash.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { IAgent } from '../../../src/core/types';
import { createFetchHandler } from '../../../src/server/handler';
import { PaymentRequiredError } from '../../../src/skills/payments/x402';

const CREDENTIAL = { 'content-type': 'application/json', authorization: 'Bearer any-presented-credential' };
const ACCEPTS = [{ scheme: 'token', network: 'robutler', maxAmountRequired: '1' }];

function refusing(): IAgent {
  const failure = () => new PaymentRequiredError('This agent requires payment.', { accepts: ACCEPTS });
  return {
    name: 'priced',
    description: 'refuses',
    skills: [],
    getCapabilities: () => ({}),
    getToolDefinitions: () => [],
    run: async () => {
      throw failure();
    },
    runStreaming: async function* () {
      throw failure();
    },
    processUAMP: async function* () {
      throw failure();
    },
  } as unknown as IAgent;
}

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => undefined);
  vi.spyOn(console, 'log').mockImplementation(() => undefined);
});
afterEach(() => vi.restoreAllMocks());

describe('a refused completion', () => {
  for (const stream of [false, true]) {
    it(`is a 402 with a JSON error${stream ? ', before any stream' : ''}`, async () => {
      const request = new Request('http://127.0.0.1/chat/completions', {
        method: 'POST',
        headers: CREDENTIAL,
        body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], stream }),
      });
      const res = await createFetchHandler(refusing())(request);
      expect(res.status).toBe(402);
      expect(res.headers.get('content-type')).toContain('application/json');
      expect(await res.json()).toEqual({ error: { code: 'payment_required', message: 'This agent requires payment.', accepts: ACCEPTS } });
    });
  }
});
