/**
 * Who pays for a served agent's model, where it listens, and what a failed
 * stream says (S-327 and B1/B6/B9, 2026-09-28).
 *
 * `serve` and the daemon ran callers' turns with no model decision at all
 * here (a file naming no LLM skill served with no model, silently, even with
 * the key exported), and the Python twin ran them on the signed-in owner's
 * credits. `WEBAGENTS_PUBLIC_URL` alone made `serve` listen on every
 * interface with no AuthSkill, and a stream that failed after its first chunk
 * answered 200 with nothing in it. Pinned here, against the shared fixture
 * `cli/final_sdk_serve_model.json` the Python suite reads too.
 *
 * Every test runs under a throwaway HOME with the FILE secrets backend, so
 * nothing here reads or writes the machine's keychain or its real login.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { attachModelForCallers, callersUnavailableMessage, resolveModelAccess } from '../../../src/cli/model-access';
import { serveAction } from '../../../src/cli/serve-action';
import type { IAgent, ISkill } from '../../../src/core/types';
import { createFetchHandler } from '../../../src/server/handler';
import { defaultHostname, loopbackBindLine } from '../../../src/server/origin-policy';
import { CALLERS_PAY_REFUSAL, LLMProxySkill } from '../../../src/skills/llm/proxy/skill';
import { resolveSkillsByName } from '../../../src/skills/resolve';
import { tempDirs } from '../../helpers/cli';

const FIXTURE = JSON.parse(
  readFileSync(path.resolve(__dirname, '../../../../python/tests/fixtures/cli/final_sdk_serve_model.json'), 'utf8'),
) as {
  decision: Array<{ name: string; model: string | null; env: Record<string, string>; signed_in: boolean; own_credential: boolean; kind: string; sent?: string }>;
  refusals: Array<{ name: string; model: string; message: string }>;
  serve_refusal_line: string;
  no_payment_token: { status: number; code: string; message: string };
  bind: { loopback: string; loopback_with_public_url: string; cases: Array<{ public_url: string | null; auth_skill: boolean; host: string; line: string | null }> };
  stream_error: { type: string; code: string; message_starts: string };
};

const tempDir = tempDirs();
const ISOLATED = [
  'HOME', 'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'WEBAGENTS_AGENT_TOKEN',
  'WEBAGENTS_PUBLIC_URL', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL',
  'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'GEMINI_API_KEY', 'XAI_API_KEY', 'FIREWORKS_API_KEY',
];
const saved: Record<string, string | undefined> = {};

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-final-sdk-serve-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
});

afterEach(() => {
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  vi.restoreAllMocks();
});

function decide(model: string | null, env: Record<string, string>, ownCredential: boolean) {
  const skills: ISkill[] = [];
  return attachModelForCallers({
    agentName: 'helper',
    declaredSkills: [],
    skills,
    byName: new Map(),
    notBuilt: new Set(),
    model: model ?? undefined,
    apiKeys: {},
    env,
    ownCredential: async () => ownCredential,
  }).then((decided) => ({ decided, skills }));
}

describe('the decision for other callers never uses the sign-in', () => {
  for (const c of FIXTURE.decision) {
    it(c.name, async () => {
      // Signed in or not, the answer is the fixture's: the sign-in is not an input.
      if (c.signed_in) process.env.WEBAGENTS_TOKEN = 'jwt.owner-login';
      const { decided, skills } = await decide(c.model, c.env, c.own_credential);
      expect(decided.access.kind).toBe(c.kind);
      if (c.kind === 'none') {
        expect(decided.problem).toMatch(/^No model for this agent's callers: /);
        expect(decided.problem).toContain('never runs on your sign-in');
        expect(skills).toEqual([]);
        return;
      }
      expect(decided.problem).toBeUndefined();
      expect(decided.access.model).toBe(c.sent);
      if (c.kind === 'proxy') {
        const proxy = skills[0] as unknown as { modelConfig: { callersPay?: boolean; platformToken?: unknown } };
        expect(skills[0]).toBeInstanceOf(LLMProxySkill);
        expect(proxy.modelConfig.callersPay).toBe(true);
        expect(proxy.modelConfig.platformToken).toBeUndefined();
      }
    });
  }

  for (const c of FIXTURE.refusals) {
    it(`says both ways out: ${c.name}`, async () => {
      const { decided } = await decide(c.model, {}, false);
      expect(decided.problem).toBe(c.message);
    });
  }

  it('words a callers access alone; the chat keeps its own sentence', async () => {
    const access = await resolveModelAccess('openai/gpt-4o-mini', { signedIn: () => false, env: {} });
    expect(callersUnavailableMessage({ ...access, forCallers: true })).toBe(FIXTURE.refusals[0].message);
  });

  it('a listed proxy skill for callers carries no sign-in, even one written in the file', async () => {
    const { skills } = await resolveSkillsByName(['proxy'], {
      proxy: { proxyUrl: 'ws://127.0.0.1:9/llm', callersPay: true, platformToken: 'written-in-the-file' },
    });
    const config = (skills[0] as unknown as { modelConfig: { callersPay?: boolean; platformToken?: unknown } }).modelConfig;
    expect(config.callersPay).toBe(true);
    expect(config.platformToken).toBeUndefined();
  });
});

describe("the proxy skill a served agent's callers pay through", () => {
  it('refuses a caller with no payment token before anything is dialled', async () => {
    const skill = new LLMProxySkill({ model: 'auto/balanced', callersPay: true, platformToken: 'jwt.owner-login', proxyUrl: 'ws://127.0.0.1:9/llm' });
    const context = { get: () => undefined, set: () => undefined, metadata: {} } as never;
    const events: Array<{ type: string; error?: { code: string; message: string; details?: unknown } }> = [];
    for await (const event of skill.processUAMP([], context)) events.push(event as never);
    const error = events.find((e) => e.type === 'response.error')!.error!;
    expect(error).toEqual({
      code: FIXTURE.no_payment_token.code,
      message: FIXTURE.no_payment_token.message,
      details: { status: FIXTURE.no_payment_token.status, shown: true },
    });
    expect(CALLERS_PAY_REFUSAL).toBe(FIXTURE.no_payment_token.message);
  });
});

describe('where serve listens', () => {
  for (const c of FIXTURE.bind.cases) {
    it(`${c.public_url ?? 'no public URL'}, ${c.auth_skill ? 'an' : 'no'} AuthSkill: ${c.host}`, () => {
      expect(defaultHostname({ publicUrl: c.public_url ?? undefined, verifiesCredentials: c.auth_skill })).toBe(c.host);
      if (c.line) {
        expect(loopbackBindLine('helper', c.public_url ?? undefined)).toBe(
          (FIXTURE.bind as unknown as Record<string, string>)[c.line].replace('{name}', 'helper'),
        );
      }
    });
  }
});

describe('serve refuses to start with no model for its callers', () => {
  const config = () => ({ name: 'served', model: 'openai/gpt-4o-mini', skills: [], skillEntries: [] });

  it('signed in, no key, no credential of its own: the sentence, and nothing served', async () => {
    process.env.WEBAGENTS_TOKEN = 'jwt.owner-login';
    const serve = vi.fn();
    const refusal = serveAction('.', { port: '0' }, { loadConfig: config, serve });
    await expect(refusal).rejects.toMatchObject({ name: 'AgentFileError' });
    await expect(refusal).rejects.toThrow(
      FIXTURE.serve_refusal_line.replace('{name}', 'served').replace('{message}', FIXTURE.refusals[0].message),
    );
    expect(serve).not.toHaveBeenCalled();
  });

  it("on the agent's own credential it serves Robutler's models, paid by each caller", async () => {
    process.env.WEBAGENTS_AGENT_TOKEN = 'agent-own-key';
    let served: IAgent | undefined;
    await serveAction('.', { port: '0' }, { loadConfig: config, serve: async (agent) => { served = agent; } });
    const proxy = ((served as unknown as { skills: ISkill[] }).skills ?? []).find((skill) => skill instanceof LLMProxySkill) as unknown as {
      modelConfig: { callersPay?: boolean; platformToken?: unknown; model?: string };
    };
    expect(proxy.modelConfig).toMatchObject({ callersPay: true, model: 'openai/gpt-4o-mini' });
    expect(proxy.modelConfig.platformToken).toBeUndefined();
  });

  it('with the key a file with no LLM skill gets that model (B6)', async () => {
    process.env.OPENAI_API_KEY = 'sk-test';
    let served: IAgent | undefined;
    await serveAction('.', { port: '0' }, { loadConfig: config, serve: async (agent) => { served = agent; } });
    const names = ((served as unknown as { skills: Array<{ name?: string }> }).skills ?? []).map((skill) => skill.name);
    expect(names).toContain('openai');
  });

  it('mcp serve, whose tools need no model, says it and starts', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const serve = vi.fn();
    await serveAction('.', { port: '0' }, { loadConfig: config, serve, noModel: 'warn' });
    expect(serve).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls.map((call) => String(call[0]))).toContain(
      FIXTURE.serve_refusal_line.replace('{name}', 'served').replace('{message}', FIXTURE.refusals[0].message),
    );
  });

  it('a skill that does not exist is one sentence, not a stack (B9)', async () => {
    process.env.OPENAI_API_KEY = 'sk-test';
    const refusal = serveAction('.', { port: '0' }, {
      loadConfig: () => ({ name: 'served', skills: ['no-such-skill'], skillEntries: ['no-such-skill'] }),
      serve: vi.fn(),
    });
    await expect(refusal).rejects.toMatchObject({ name: 'AgentFileError' });
    await expect(refusal).rejects.toThrow('Unknown skill(s) in agent config: no-such-skill.');
  });
});

describe('what a failed completion answers', () => {
  const CREDENTIAL = { 'content-type': 'application/json', authorization: 'Bearer any-presented-credential' };

  function agentWith(stream: () => AsyncGenerator<unknown>, run: () => Promise<unknown>): IAgent {
    return {
      name: 'served',
      description: '',
      skills: [],
      getCapabilities: () => ({}),
      getToolDefinitions: () => [],
      run,
      runStreaming: stream,
      processUAMP: async function* () {},
    } as unknown as IAgent;
  }

  function post(agent: IAgent, stream: boolean) {
    return createFetchHandler(agent)(new Request('http://127.0.0.1/chat/completions', {
      method: 'POST',
      headers: CREDENTIAL,
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], stream }),
    }));
  }

  const shown = () => Object.assign(new Error(FIXTURE.no_payment_token.message), {
    code: FIXTURE.no_payment_token.code,
    details: { status: FIXTURE.no_payment_token.status, shown: true },
  });

  it("a refusal written for the caller keeps its status and words, streamed or not", async () => {
    const agent = agentWith(
      async function* () { yield { type: 'error', error: shown() }; },
      async () => { throw shown(); },
    );
    for (const stream of [false, true]) {
      const res = await post(agent, stream);
      expect(res.status).toBe(FIXTURE.no_payment_token.status);
      expect(await res.json()).toEqual({ error: { code: FIXTURE.no_payment_token.code, message: FIXTURE.no_payment_token.message } });
    }
  });

  it('a stream that fails after it began sends the error and logs it (B1)', async () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const agent = agentWith(
      async function* () {
        yield { type: 'delta', delta: 'par' };
        yield { type: 'error', error: new Error('OpenAI API returned 402: {"secret":"provider body"}') };
      },
      async () => ({ content: '' }),
    );
    const res = await post(agent, true);
    expect(res.status).toBe(200);
    const events = (await res.text()).split('\n').filter((line) => line.startsWith('data: '));
    expect(events.at(-1)).not.toBe('data: [DONE]');
    const failure = JSON.parse(events.at(-1)!.slice('data: '.length)) as { error: { message: string; type: string; code: string } };
    expect(failure.error.type).toBe(FIXTURE.stream_error.type);
    expect(failure.error.code).toBe(FIXTURE.stream_error.code);
    expect(failure.error.message.startsWith(FIXTURE.stream_error.message_starts)).toBe(true);
    expect(failure.error.message).not.toContain('provider body');
    const reference = failure.error.message.slice(FIXTURE.stream_error.message_starts.length);
    expect(logged.mock.calls.some((call) => String(call[0]).includes(reference))).toBe(true);
    // No `stop` chunk claims the turn finished.
    expect(events.some((line) => line.includes('"finish_reason":"stop"'))).toBe(false);
  });
});

describe("ACP and stdio MCP are the owner's own turns", () => {
  it('the sign-in may pay there, and never under serve or mcp serve --http', async () => {
    // The editor or MCP client the owner started on this machine; the ACP
    // `login` auth method promises the sign-in works.
    const { createServedAgent } = await import('../../../src/cli/serve-action');
    process.env.WEBAGENTS_TOKEN = 'jwt.owner-login';
    const dir = tempDir('wa-final-sdk-owner-');
    const { writeFileSync } = await import('node:fs');
    writeFileSync(path.join(dir, 'AGENT.md'), '---\nname: served\nmodel: openai/gpt-4o-mini\n---\nHelp.\n');
    const proxyOf = (agent: IAgent) =>
      ((agent as unknown as { skills: ISkill[] }).skills ?? []).find((skill) => skill instanceof LLMProxySkill) as unknown as
        | { modelConfig: { callersPay?: boolean; platformToken?: () => Promise<string | undefined> } }
        | undefined;
    const own = await createServedAgent(dir, { forCallers: false });
    expect(proxyOf(own)?.modelConfig.callersPay).toBeUndefined();
    expect(await proxyOf(own)?.modelConfig.platformToken?.()).toBe('jwt.owner-login');
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const callers = await createServedAgent(dir);
    expect(proxyOf(callers)).toBeUndefined();
    expect(warn.mock.calls.map((call) => String(call[0]))).toContain(
      FIXTURE.serve_refusal_line.replace('{name}', 'served').replace('{message}', FIXTURE.refusals[0].message),
    );
  });
});
