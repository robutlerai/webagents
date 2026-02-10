/**
 * Completions Transport Payment Tests - 402 handling
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { CompletionsTransportSkill } from '../../../src/skills/transport/completions/skill.js';
import { PaymentRequiredError } from '../../../src/skills/payments/x402.js';
import type { IAgent } from '../../../src/core/types.js';
import type { ServerEvent } from '../../../src/uamp/events.js';
import {
  createResponseDeltaEvent,
  createResponseDoneEvent,
} from '../../../src/uamp/events.js';

function createMockAgent(processUAMP: IAgent['processUAMP']): IAgent {
  return {
    name: 'test-agent',
    processUAMP,
    getCapabilities: vi.fn(() => ({})),
  } as unknown as IAgent;
}

function createRequest(body: object, headers?: Record<string, string>): Request {
  return new Request('http://localhost/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify(body),
  });
}

function createMockContext() {
  const store = new Map<string, unknown>();
  return {
    get: vi.fn((key: string) => store.get(key)),
    set: vi.fn((key: string, value: unknown) => { store.set(key, value); }),
    metadata: {},
    _store: store,
  };
}

describe('Completions Transport - Payment Handling', () => {
  let skill: CompletionsTransportSkill;

  beforeEach(() => {
    skill = new CompletionsTransportSkill();
  });

  describe('streaming 402 pre-flight', () => {
    it('returns 402 JSON when processUAMP throws PaymentRequiredError', async () => {
      async function* failingProcessUAMP(): AsyncGenerator<ServerEvent> {
        throw new PaymentRequiredError('Payment required', {
          accepts: [{ scheme: 'token', amount: '0.01' }],
        });
        yield undefined as never; // unreachable
      }

      const agent = createMockAgent(failingProcessUAMP as any);
      (skill as any).agent = agent;

      const req = createRequest(
        { model: 'gpt-4', messages: [{ role: 'user', content: 'Hi' }], stream: true },
      );
      const ctx = createMockContext();

      const response = await skill.handleCompletions(req, ctx as any);

      expect(response.status).toBe(402);
      const body = await response.json();
      expect(body.status_code).toBe(402);
      expect(body.context.accepts).toEqual([{ scheme: 'token', amount: '0.01' }]);
    });

    it('returns 402 JSON for non-streaming requests', async () => {
      async function* failingProcessUAMP(): AsyncGenerator<ServerEvent> {
        throw new PaymentRequiredError('Payment required');
        yield undefined as never;
      }

      const agent = createMockAgent(failingProcessUAMP as any);
      (skill as any).agent = agent;

      const req = createRequest(
        { model: 'gpt-4', messages: [{ role: 'user', content: 'Hi' }], stream: false },
      );
      const ctx = createMockContext();

      const response = await skill.handleCompletions(req, ctx as any);
      expect(response.status).toBe(402);
    });
  });

  describe('payment token from headers', () => {
    it('sets payment_token in context from X-Payment-Token header', async () => {
      async function* okProcessUAMP(): AsyncGenerator<ServerEvent> {
        yield createResponseDeltaEvent('r1', { type: 'text', text: 'Hello' });
        yield createResponseDoneEvent('r1', 'completed', [{ type: 'text', text: 'Hello' }]);
      }

      const agent = createMockAgent(okProcessUAMP as any);
      (skill as any).agent = agent;

      const req = createRequest(
        { model: 'gpt-4', messages: [{ role: 'user', content: 'Hi' }], stream: false },
        { 'X-Payment-Token': 'tok_abc' },
      );
      const ctx = createMockContext();

      await skill.handleCompletions(req, ctx as any);

      expect(ctx.set).toHaveBeenCalledWith('payment_token', 'tok_abc');
    });

    it('reads x-payment header (lowercase)', async () => {
      async function* okProcessUAMP(): AsyncGenerator<ServerEvent> {
        yield createResponseDeltaEvent('r1', { type: 'text', text: 'Hi' });
        yield createResponseDoneEvent('r1', 'completed', [{ type: 'text', text: 'Hi' }]);
      }

      const agent = createMockAgent(okProcessUAMP as any);
      (skill as any).agent = agent;

      const req = createRequest(
        { model: 'gpt-4', messages: [{ role: 'user', content: 'Hi' }] },
        { 'x-payment': 'tok_lowercase' },
      );
      const ctx = createMockContext();

      await skill.handleCompletions(req, ctx as any);

      expect(ctx.set).toHaveBeenCalledWith('payment_token', 'tok_lowercase');
    });
  });
});
