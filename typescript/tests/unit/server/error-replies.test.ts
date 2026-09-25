/**
 * A failed run does not hand its error message to the caller (S-228,
 * 2026-09-24).
 *
 * `handler.ts` and `node.ts` answered a failed run with
 * `(error as Error).message`: `completions_error`, `uamp_error`,
 * `handler_error`. The LLM skills throw `<Provider> API returned <status>:
 * <body>`, and OpenAI's 401 body quotes a masked form of the rejected key.
 *
 * Pinned per door: the message is a fixed sentence and a reference, never the
 * error's text; the reference and the text are in the server's log; status
 * codes and body shapes are unchanged. And what keeps its text: this SDK's
 * auth refusals and `PaymentRequiredError`, and the daemon while it listens
 * on loopback, where the caller is the developer's own CLI.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { IAgent } from '../../../src/core/types.js';
import { createFetchHandler } from '../../../src/server/handler.js';
import { createAgentApp } from '../../../src/server/node.js';
import { WebAgentsDaemon } from '../../../src/daemon/server.js';
import { PaymentRequiredError } from '../../../src/skills/payments/x402.js';
import { isLoopbackAddress, isMeantToBeShown } from '../../../src/server/error-reply.js';

const SECRET =
  'OpenAI API returned 401: {"error":{"message":"Incorrect API key provided: sk-proj-****abcd"}} /home/owner/agent';
const REFERENCE = /^The agent could not complete this request\. Reference: ([0-9a-f]{8})$/;
const CREDENTIAL = { 'content-type': 'application/json', authorization: 'Bearer any-presented-credential' };

function named(name: string, message: string): Error {
  const error = new Error(message);
  error.name = name;
  return error;
}

/** An agent whose every way of running throws `failure()`. */
function failingAgent(failure: () => unknown = () => new Error(SECRET)): IAgent {
  return {
    name: 'leaky',
    description: 'fails',
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
    httpRegistry: new Map([
      [
        'POST:/boom',
        {
          path: '/boom',
          method: 'POST',
          handler: async () => {
            throw failure();
          },
        },
      ],
    ]),
  } as unknown as IAgent;
}

function post(path: string, body: unknown = [], headers: Record<string, string> = CREDENTIAL): Request {
  return new Request(`http://127.0.0.1${path}`, { method: 'POST', headers, body: JSON.stringify(body) });
}

let logged: unknown[][];
beforeEach(() => {
  logged = [];
  vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    logged.push(args);
  });
  vi.spyOn(console, 'log').mockImplementation(() => undefined);
});
afterEach(() => {
  vi.restoreAllMocks();
});

/** The reference in a reply, after checking the reply carries none of the error's text. */
function referenceIn(message: string): string {
  expect(message).not.toContain('sk-proj');
  expect(message).not.toContain('/home/owner');
  const match = REFERENCE.exec(message);
  expect(match, message).not.toBeNull();
  return match![1];
}

/** The same reference, with the error itself, in what the server logged. */
function expectLogged(reference: string): void {
  const line = logged.find((args) => String(args[0]).includes(reference));
  expect(line, JSON.stringify(logged.map((args) => String(args[0])))).toBeDefined();
  expect((line![1] as Error).message).toBe(SECRET);
}

describe('createFetchHandler', () => {
  const chat = { messages: [{ role: 'user', content: 'hi' }] };

  it('answers a failed completion with a reference, not the error', async () => {
    const res = await createFetchHandler(failingAgent())(post('/chat/completions', chat));
    expect(res.status).toBe(500);
    const body = (await res.json()) as { error: { code: string; message: string } };
    expect(body.error.code).toBe('completions_error');
    expectLogged(referenceIn(body.error.message));
  });

  it('keeps an auth refusal as the 401 it was, message and all', async () => {
    const agent = failingAgent(() => named('AuthenticationError', 'Invalid API key'));
    const res = await createFetchHandler(agent)(post('/chat/completions', chat));
    expect(res.status).toBe(401);
    expect(((await res.json()) as { error: { message: string } }).error.message).toBe('Invalid API key');
  });

  it('answers a failed UAMP run with a reference', async () => {
    const res = await createFetchHandler(failingAgent())(post('/uamp'));
    expect(res.status).toBe(500);
    const body = (await res.json()) as { error: { code: string; message: string } };
    expect(body.error.code).toBe('uamp_error');
    expectLogged(referenceIn(body.error.message));
  });

  it("answers a skill endpoint's failure with a reference", async () => {
    const res = await createFetchHandler(failingAgent())(post('/boom', {}, { 'content-type': 'application/json' }));
    expect(res.status).toBe(500);
    const body = (await res.json()) as { error: { code: string; message: string } };
    expect(body.error.code).toBe('handler_error');
    expectLogged(referenceIn(body.error.message));
  });

  it('keeps a payment refusal readable', async () => {
    const agent = failingAgent(() => new PaymentRequiredError('This agent requires payment.'));
    const res = await createFetchHandler(agent)(post('/boom', {}, { 'content-type': 'application/json' }));
    expect(((await res.json()) as { error: { message: string } }).error.message).toBe('This agent requires payment.');
  });
});

describe('createAgentApp', () => {
  it('answers a failed UAMP run with a reference', async () => {
    const { app } = createAgentApp(failingAgent(), { logging: false });
    const res = await app.fetch(post('/uamp'));
    expect(res.status).toBe(500);
    const body = (await res.json()) as { error: { code: string; message: string } };
    expect(body.error.code).toBe('uamp_error');
    expectLogged(referenceIn(body.error.message));
  });

  it("answers a mounted skill endpoint's failure with a reference", async () => {
    const { app } = createAgentApp(failingAgent(), { logging: false });
    const res = await app.fetch(post('/boom', {}, { 'content-type': 'application/json' }));
    expect(res.status).toBe(500);
    const body = (await res.json()) as { error: { code: string; message: string } };
    expect(body.error.code).toBe('handler_error');
    expectLogged(referenceIn(body.error.message));
  });
});

describe('the daemon', () => {
  function daemon(hostname: string): WebAgentsDaemon {
    const d = new WebAgentsDaemon({ port: 0, hostname, watch: false, cron: false });
    d.registry.registerLocal(failingAgent());
    return d;
  }
  const chat = (stream: boolean) =>
    post('/agents/leaky/chat/completions', { messages: [{ role: 'user', content: 'hi' }], stream });

  it("keeps the error's text on loopback, where the caller is the developer's CLI", async () => {
    const res = await daemon('127.0.0.1').app.fetch(chat(false));
    expect(res.status).toBe(500);
    expect(((await res.json()) as { error: string }).error).toBe(SECRET);
  });

  it('answers like any server once it listens beyond this machine', async () => {
    const plain = await daemon('0.0.0.0').app.fetch(chat(false));
    expect(plain.status).toBe(500);
    expectLogged(referenceIn(((await plain.json()) as { error: string }).error));

    const streamed = await daemon('0.0.0.0').app.fetch(chat(true));
    const event = (await streamed.text()).split('\n').find((line) => line.startsWith('data: {'));
    expect(event).toBeDefined();
    referenceIn((JSON.parse(event!.slice(6)) as { error: string }).error);
  });
});

describe('what keeps its text', () => {
  it("is this SDK's refusals, by name", () => {
    expect(isMeantToBeShown(named('AuthenticationError', 'x'))).toBe(true);
    expect(isMeantToBeShown(named('AuthorizationError', 'x'))).toBe(true);
    expect(isMeantToBeShown(new PaymentRequiredError())).toBe(true);
    expect(isMeantToBeShown(new Error(SECRET))).toBe(false);
  });

  it('decides loopback from literal addresses only', () => {
    for (const host of ['127.0.0.1', '127.0.0.2', '::1', '[::1]', 'localhost']) {
      expect(isLoopbackAddress(host), host).toBe(true);
    }
    for (const host of ['0.0.0.0', '::', '192.168.1.5', 'example.com', undefined]) {
      expect(isLoopbackAddress(host), String(host)).toBe(false);
    }
  });
});
