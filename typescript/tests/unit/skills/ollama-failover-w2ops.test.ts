/**
 * Local models and model failover (2026-09-26, gap-closure plan items 2.8
 * and 2.4, lane w2-ops), against the shared fixture
 * `python/tests/fixtures/w2ops/models.json`, which the Python suite reads
 * too (`tests/test_ollama_failover_w2ops.py`):
 *
 *  * `base_url` in a model skill's entry reaches the skill (it mapped to a
 *    key `OpenAISkill` never read, so only OPENAI_BASE_URL worked);
 *  * Ollama as a named provider: the registry row, the placeholder key, the
 *    model-access decision, doctor's words and the `models` row;
 *  * `fallback_models:`: the chain moves on a provider error, says so in
 *    the transcript, and stops on anything else.
 *
 * Everything runs against a stub OpenAI-compatible server on loopback:
 * never a real Ollama, never a real key.
 */

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as http from 'node:http';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import type { StreamChunk } from '../../../src/core/types';
import { AgentFileError, parseAgentMarkdown } from '../../../src/agents/index';
import { resolveModelAccess, describeLocalRoute } from '../../../src/cli/model-access';
import { resolveSkillsByName, withFallbackModels } from '../../../src/skills/resolve';
import { MODELS_READY_FOOTNOTE, findProvider, providerBaseUrl, providerNeeds } from '../../../src/skills/llm/providers';
import { OpenAISkill } from '../../../src/skills/llm/openai/skill';
import { OllamaSkill, OLLAMA_PLACEHOLDER_KEY } from '../../../src/skills/llm/ollama/skill';
import { ollamaModelCheck, probeOllama, servesModel } from '../../../src/skills/llm/ollama/probe';
import { FailoverLLMSkill, failoverNote, providerFailure } from '../../../src/skills/llm/failover/skill';
import { setRequestErrorDetail } from '../../../src/skills/llm/request';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/w2ops/models.json'), 'utf8')) as {
  base_url: { file_key: string; skill_config_keys: { typescript: string[]; python: string[] }; env_var: string };
  ollama: {
    provider: Record<string, unknown>;
    placeholder_key: string;
    models_row: { id: string; format: string; needs: string };
    models_footnote: string;
    status_route: string;
    doctor: { reachable: string; not_pulled: { detail: string; fix: string }; unreachable: { detail: string; fix: string } };
    model_match: Array<{ wanted: string; served: string; match: boolean }>;
  };
  failover: {
    file_key: string;
    invalid: string;
    note: string;
    retryable: { statuses: number[]; network: boolean };
    not_retryable: number[];
    reason: { status: string; network: string; network_served: string };
    cases: Array<{
      case: string;
      chain: string[];
      server: Record<string, number>;
      requests: string[];
      answered_by: string | null;
      notes: string[];
    }>;
  };
};

// ---------------------------------------------------------------------------
// The stub: an OpenAI-compatible server on loopback
// ---------------------------------------------------------------------------

interface Seen {
  path: string;
  model: string;
  authorization: string | undefined;
}

let server: http.Server;
let base = '';
const seen: Seen[] = [];
/** Per model: the status the stub answers; 200 streams a reply. */
let statuses: Record<string, number> = {};
const served = ['llama3.2:latest', 'gemma3:latest'];

beforeAll(async () => {
  server = http.createServer((req, res) => {
    if (req.method === 'GET' && req.url?.endsWith('/models')) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ data: served.map((id) => ({ id })) }));
      return;
    }
    let body = '';
    req.on('data', (chunk) => { body += chunk; });
    req.on('end', () => {
      const parsed = JSON.parse(body || '{}') as { model?: string };
      const model = parsed.model ?? '';
      seen.push({ path: req.url ?? '', model, authorization: req.headers.authorization });
      const status = statuses[model] ?? 200;
      if (status !== 200) {
        res.writeHead(status, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: { message: `stub says ${status}` } }));
        return;
      }
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { role: 'assistant', content: `hello from ${model}` }, finish_reason: null }] })}\n\n`);
      res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`);
      res.write(`data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } })}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address() as { port: number };
  base = `http://127.0.0.1:${address.port}/v1`;
});

afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

const ENV = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OLLAMA_BASE_URL', 'ANTHROPIC_API_KEY'];
const saved: Record<string, string | undefined> = {};
beforeEach(() => {
  for (const name of ENV) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  seen.length = 0;
  statuses = {};
  setRequestErrorDetail(true);
});
afterEach(() => {
  for (const name of ENV) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  setRequestErrorDetail(false);
});

/** One streamed turn through an agent built on `skills`: its text, its notes and its error. */
async function turn(skills: unknown[]): Promise<{ text: string; notes: string[]; error?: string }> {
  const agent = new BaseAgent({ name: 'w2ops', skills: skills as never });
  await agent.initialize();
  let text = '';
  const notes: string[] = [];
  let error: string | undefined;
  for await (const chunk of agent.runStreaming([{ role: 'user', content: 'hi' }]) as AsyncGenerator<StreamChunk>) {
    if (chunk.type === 'delta' && chunk.delta) text += chunk.delta;
    if (chunk.type === 'note' && chunk.note) notes.push(chunk.note);
    if (chunk.type === 'error') error = chunk.error?.message;
  }
  return { text, notes, ...(error ? { error } : {}) };
}

// ---------------------------------------------------------------------------
// base_url (item 5)
// ---------------------------------------------------------------------------

describe("the entry's base_url (fixture base_url)", () => {
  it('reaches the OpenAI skill under both spellings, and the request goes there', async () => {
    const { skills, failed } = await resolveSkillsByName([{ openai: { [FIXTURE.base_url.file_key]: base, api_key: 'k-file', model: 'primary-model' } }]);
    expect(failed).toEqual([]);
    const config = (skills[0] as unknown as { modelConfig: Record<string, unknown> }).modelConfig;
    for (const key of FIXTURE.base_url.skill_config_keys.typescript) expect(config[key]).toBe(base);
    const { text } = await turn(skills);
    expect(text).toBe('hello from primary-model');
    expect(seen.map((s) => [s.path, s.model, s.authorization])).toEqual([['/v1/chat/completions', 'primary-model', 'Bearer k-file']]);
  });

  it('the environment variable stands in when the entry names none', async () => {
    process.env[FIXTURE.base_url.env_var] = base;
    process.env.OPENAI_API_KEY = 'k-env';
    const { skills } = await resolveSkillsByName(['openai'], { model: 'openai/primary-model' });
    expect((await turn(skills)).text).toBe('hello from primary-model');
    expect(seen[0].authorization).toBe('Bearer k-env');
  });
});

// ---------------------------------------------------------------------------
// Ollama (item 3)
// ---------------------------------------------------------------------------

describe('Ollama as a named provider (fixture ollama)', () => {
  it('is the fixture row of the registry', () => {
    const p = findProvider('ollama')!;
    const row = FIXTURE.ollama.provider;
    expect({
      id: p.id, aliases: p.aliases, description: p.description, credential: p.credential, model_format: p.modelFormat,
      default_model: p.defaultModel, base_url_var: p.baseUrlVar, default_base_url: p.defaultBaseUrl,
    }).toEqual(row);
    expect(providerBaseUrl(p, {})).toBe(row.default_base_url);
    expect(providerBaseUrl(p, { OLLAMA_BASE_URL: base })).toBe(base);
    expect(OLLAMA_PLACEHOLDER_KEY).toBe(FIXTURE.ollama.placeholder_key);
  });

  it('sends the OpenAI wire shape to OLLAMA_BASE_URL with the placeholder key', async () => {
    process.env.OLLAMA_BASE_URL = base;
    const { skills, failed } = await resolveSkillsByName(['ollama'], { model: 'ollama/llama3.2' });
    expect(failed).toEqual([]);
    expect(skills[0]).toBeInstanceOf(OllamaSkill);
    const { text } = await turn(skills);
    expect(text).toBe('hello from llama3.2');
    expect(seen.map((s) => [s.model, s.authorization])).toEqual([['llama3.2', `Bearer ${FIXTURE.ollama.placeholder_key}`]]);
    // The skill answers as `ollama`, not `openai`.
    expect(new OllamaSkill({}).getCapabilities().provider).toBe('ollama');
    expect(new OpenAISkill({}).getCapabilities().provider).toBe('openai');
  });

  it('is reached directly when named, never chosen when the file names no model', async () => {
    const named = await resolveModelAccess('ollama/llama3.2', { signedIn: () => false, env: {} });
    expect([named.kind, named.model]).toEqual(['direct', 'ollama/llama3.2']);
    expect(describeLocalRoute(named, {})).toBe(FIXTURE.ollama.status_route.replace('{model}', 'ollama/llama3.2').replace('{base_url}', 'http://localhost:11434/v1'));
    expect(describeLocalRoute(await resolveModelAccess('openai/x', { signedIn: () => false, env: { OPENAI_API_KEY: 'k' } }))).toBeUndefined();
    const unnamed = await resolveModelAccess(undefined, { signedIn: () => false, env: {} });
    expect(unnamed.kind).toBe('none');
    const signedIn = await resolveModelAccess(undefined, { signedIn: () => true, env: {} });
    expect([signedIn.kind, signedIn.model]).toEqual(['proxy', 'auto/balanced']);
  });

  it('is listed by `webagents models` with the fixture words', () => {
    const p = findProvider('ollama')!;
    expect(p.id).toBe(FIXTURE.ollama.models_row.id);
    expect(p.modelFormat).toBe(FIXTURE.ollama.models_row.format);
    expect(providerNeeds(p, 'webagents login')).toBe(FIXTURE.ollama.models_row.needs);
    expect(MODELS_READY_FOOTNOTE.replace('{command}', 'webagents secrets set')).toBe(FIXTURE.ollama.models_footnote);
  });

  it.each(FIXTURE.ollama.model_match)('$wanted served as $served: $match', ({ wanted, served: id, match }) => {
    expect(servesModel([id], wanted)).toBe(match);
  });

  it("doctor's model check: answering with the model, without it, and not at all", async () => {
    const reachable = await probeOllama(base);
    expect(reachable).toEqual({ ok: true, models: served });
    const words = FIXTURE.ollama.doctor;
    const fill = (s: string, model: string) => s.replace('{model}', model).replace('{base_url}', base).replace('{bare}', model.slice('ollama/'.length));
    expect(ollamaModelCheck('ollama/llama3.2', base, reachable)).toEqual({ status: 'ok', detail: fill(words.reachable, 'ollama/llama3.2') });
    expect(ollamaModelCheck('ollama/mistral', base, reachable)).toEqual({
      status: 'warn', detail: fill(words.not_pulled.detail, 'ollama/mistral'), fix: fill(words.not_pulled.fix, 'ollama/mistral'),
    });
    const dead = 'http://127.0.0.1:1/v1';
    const unreachable = await probeOllama(dead, 500);
    expect(unreachable.ok).toBe(false);
    expect(ollamaModelCheck('ollama/llama3.2', dead, unreachable)).toEqual({
      status: 'fail',
      detail: words.unreachable.detail.replace('{model}', 'ollama/llama3.2').replace('{base_url}', dead),
      fix: words.unreachable.fix.replace('{bare}', 'llama3.2'),
    });
  });
});

// ---------------------------------------------------------------------------
// Failover (item 4)
// ---------------------------------------------------------------------------

describe('model failover (fixture failover)', () => {
  it('the agent file key: a list of provider/model strings, and nothing else', () => {
    const md = (value: string) => `---\nname: t\nmodel: openai/a\n${FIXTURE.failover.file_key}: ${value}\n---\nHi\n`;
    expect(parseAgentMarkdown(md('[openai/b, anthropic/c]'), '/x/AGENT.md').fallbackModels).toEqual(['openai/b', 'anthropic/c']);
    expect(() => parseAgentMarkdown(md('openai/b'), '/x/AGENT.md')).toThrow(new AgentFileError(`/x/AGENT.md: ${FIXTURE.failover.invalid}`));
    expect(() => parseAgentMarkdown(md('[1]'), '/x/AGENT.md')).toThrow(AgentFileError);
    // The sibling key of item 1, parsed on the same pass.
    expect(parseAgentMarkdown('---\nname: t\nobservability: {otel: true}\n---\nHi\n', '/x/AGENT.md').observability).toEqual({ otel: true });
    expect(() => parseAgentMarkdown('---\nname: t\nobservability: {otle: true}\n---\nHi\n', '/x/AGENT.md')).toThrow(
      new AgentFileError("/x/AGENT.md: observability: unknown key 'otle'. It takes otel."),
    );
  });

  it('names a failure the way the fixture does, and moves on only for a provider error', () => {
    for (const status of FIXTURE.failover.retryable.statuses) {
      expect(providerFailure({ message: `OpenAI API returned ${status}: x` })).toEqual({ reason: FIXTURE.failover.reason.status.replace('{status}', String(status)) });
    }
    for (const status of FIXTURE.failover.not_retryable) expect(providerFailure({ message: `OpenAI API returned ${status}: x` })).toBeUndefined();
    expect(providerFailure({ message: 'Could not reach http://127.0.0.1:1: connect ECONNREFUSED' })).toEqual({
      reason: FIXTURE.failover.reason.network.replace('{origin}', 'http://127.0.0.1:1'),
    });
    // Under `serve` the request error carries no origin (`fetchModel` keeps
    // it for the terminal), so a caller reads the served wording (S-228).
    expect(providerFailure({ message: 'fetch failed' })).toEqual({ reason: FIXTURE.failover.reason.network_served });
    expect(providerFailure({ message: 'OpenAI API key not configured' })).toBeUndefined();
    expect(failoverNote('a', 'HTTP 503', 'b')).toBe(FIXTURE.failover.note.replace('{failed}', 'a').replace('{reason}', 'HTTP 503').replace('{next}', 'b'));
  });

  it.each(FIXTURE.failover.cases)('$case', async (c) => {
    process.env.OPENAI_BASE_URL = base;
    process.env.OPENAI_API_KEY = 'k';
    statuses = c.server;
    const { skills } = await resolveSkillsByName(['openai'], { model: c.chain[0] });
    const chained = await withFallbackModels(skills, c.chain.slice(1), { primaryModel: c.chain[0], env: process.env });
    expect(chained.failed).toEqual([]);
    expect(chained.failover?.models).toEqual(c.chain);
    const result = await turn(chained.skills);
    expect(seen.map((s) => s.model)).toEqual(c.requests);
    expect(result.notes).toEqual(c.notes);
    if (c.answered_by === null) {
      expect(result.text).toBe('');
      expect(result.error).toBeDefined();
      expect(chained.failover?.answeredModel).toBeUndefined();
    } else {
      expect(result.text).toBe(`hello from ${c.answered_by.slice('openai/'.length)}`);
      expect(chained.failover?.answeredModel).toBe(c.answered_by);
    }
  });

  it('a primary that cannot be reached moves on with the origin in the note at the terminal, and without it when served', async () => {
    const chain = () => {
      const primary = new OpenAISkill({ baseURL: 'http://127.0.0.1:1/v1', apiKey: 'k', model: 'primary-model' });
      const fallback = new OpenAISkill({ baseURL: base, apiKey: 'k', model: 'fallback-model' });
      return new FailoverLLMSkill({ chain: [{ skill: primary, model: 'openai/primary-model' }, { skill: fallback, model: 'openai/fallback-model' }] });
    };
    // The chat and `-p` switch the request error detail on: the owner reads where the primary was.
    const result = await turn([chain()]);
    expect(result.text).toBe('hello from fallback-model');
    expect(result.notes).toEqual([failoverNote('openai/primary-model', 'could not reach http://127.0.0.1:1', 'openai/fallback-model')]);
    // `serve` does not: a caller reads the served wording (S-228).
    setRequestErrorDetail(false);
    const served = await turn([chain()]);
    expect(served.text).toBe('hello from fallback-model');
    expect(served.notes).toEqual([failoverNote('openai/primary-model', FIXTURE.failover.reason.network_served, 'openai/fallback-model')]);
  });

  it('a fallback that cannot be built here is reported and left out, never fatal', async () => {
    process.env.OPENAI_BASE_URL = base;
    process.env.OPENAI_API_KEY = 'k';
    const { skills } = await resolveSkillsByName(['openai'], { model: 'openai/primary-model' });
    const chained = await withFallbackModels(skills, ['anthropic/claude-x', 'nope/model', 'auto/balanced', 'openai/fallback-model'], { primaryModel: 'openai/primary-model', env: process.env });
    expect(chained.failed).toEqual([
      { name: 'fallback anthropic/claude-x', reason: 'ANTHROPIC_API_KEY is not set' },
      { name: 'fallback nope/model', reason: 'this SDK has no client for nope' },
      { name: 'fallback auto/balanced', reason: "Robutler's models need a sign-in" },
    ]);
    expect(chained.failover?.models).toEqual(['openai/primary-model', 'openai/fallback-model']);
    // With nothing buildable, the skills are left as they were.
    const alone = await withFallbackModels(skills, ['nope/model'], { primaryModel: 'openai/primary-model', env: process.env });
    expect(alone.failover).toBeUndefined();
    expect(alone.skills).toBe(skills);
  });
});
