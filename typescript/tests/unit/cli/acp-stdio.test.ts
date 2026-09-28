/**
 * `webagents acp` end to end (gap-closure plan item 1.6, 2026-09-26): the CLI
 * is spawned as an editor would spawn it (through this repo's tsx, with HOME
 * in a temporary folder) and an OpenAI-compatible model scripted on loopback
 * (no network, a dummy key), and each transcript in
 * `python/tests/fixtures/acp/acp_transcripts.json` is driven through its stdin
 * and read back from its stdout. The Python CLI runs the same cases in
 * `tests/test_acp_stdio.py`. Every stdout line must be a JSON-RPC message,
 * and the startup line must be on stderr.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

import { CLI_ARGS, cliPrerequisite, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures/acp');
const PROTOCOL = JSON.parse(readFileSync(path.join(FIXTURES, 'acp_protocol.json'), 'utf8')) as {
  startup: string;
  session_id_prefix: string;
  agent_capabilities: unknown;
  auth_methods: unknown;
  permission: { options: unknown };
};
type Json = Record<string, unknown>;
interface Step {
  send?: Json;
  send_raw?: string;
  then_send?: Json;
  then_send_after_call?: number;
  expect: Json[];
  model_sees?: { call: number; messages_contain: string[] }[];
}
interface Case {
  name: string;
  model: ModelEntry[];
  steps: Step[];
  restart?: boolean;
  after_restart?: Step[];
  files_after?: Record<string, string>;
  files_absent?: string[];
}
interface ModelEntry {
  text?: string;
  tool_call?: { name: string; arguments: Json };
  delay_ms?: number;
}
const TRANSCRIPTS = JSON.parse(readFileSync(path.join(FIXTURES, 'acp_transcripts.json'), 'utf8')) as {
  agent_file: string;
  env: Record<string, string>;
  base_url_env: string;
  cases: Case[];
};
const ECHO_SERVER = path.resolve(HERE, '../../fixtures/mcp-echo-server.mjs');
const LINE_TIMEOUT_MS = 60_000;

const tempDir = tempDirs();
const isRecord = (v: unknown): v is Json => !!v && typeof v === 'object' && !Array.isArray(v);
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// -- the scripted model ------------------------------------------------------------------------

/**
 * `POST /v1/chat/completions` answered from a script, one entry per call (the
 * last repeats), streamed as SSE when asked to; every request's messages are
 * kept for `model_sees`.
 */
class StubModel {
  script: ModelEntry[] = [];
  calls: Json[] = [];
  private server: Server;
  baseUrl = '';

  constructor() {
    this.server = createServer((req, res) => {
      let body = '';
      req.on('data', (chunk: Buffer) => void (body += chunk.toString()));
      req.on('end', () => {
        void this.answer(JSON.parse(body || '{}') as Json, res);
      });
    });
  }

  listen(): Promise<void> {
    return new Promise((resolve) => {
      this.server.listen(0, '127.0.0.1', () => {
        const { port } = this.server.address() as { port: number };
        this.baseUrl = `http://127.0.0.1:${port}/v1`;
        resolve();
      });
    });
  }

  close(): Promise<void> {
    return new Promise((resolve) => this.server.close(() => resolve()));
  }

  reset(script: ModelEntry[]): void {
    this.script = script;
    this.calls = [];
  }

  private async answer(body: Json, res: import('node:http').ServerResponse): Promise<void> {
    const index = this.calls.length;
    this.calls.push(body);
    const entry = this.script[Math.min(index, this.script.length - 1)] ?? { text: '' };
    if (entry.delay_ms) await sleep(entry.delay_ms);
    res.on('error', () => undefined);
    if (res.destroyed) return;
    if (body.stream) {
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
      for (const chunk of this.chunks(entry)) res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(this.completion(entry)));
    }
  }

  private message(entry: ModelEntry): Json {
    if (entry.tool_call) {
      return {
        role: 'assistant',
        content: null,
        tool_calls: [{ id: 'call_1', type: 'function', function: { name: entry.tool_call.name, arguments: JSON.stringify(entry.tool_call.arguments) } }],
      };
    }
    return { role: 'assistant', content: entry.text ?? '' };
  }

  private completion(entry: ModelEntry): Json {
    const message = this.message(entry);
    return {
      id: 'chatcmpl-stub',
      object: 'chat.completion',
      created: 0,
      model: 'stub',
      choices: [{ index: 0, message, finish_reason: 'tool_calls' in message ? 'tool_calls' : 'stop' }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    };
  }

  private chunks(entry: ModelEntry): Json[] {
    const base = { id: 'chatcmpl-stub', object: 'chat.completion.chunk', created: 0, model: 'stub' };
    const delta: Json = entry.tool_call
      ? { role: 'assistant', tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: entry.tool_call.name, arguments: JSON.stringify(entry.tool_call.arguments) } }] }
      : { role: 'assistant', content: entry.text ?? '' };
    const finish = entry.tool_call ? 'tool_calls' : 'stop';
    return [
      { ...base, choices: [{ index: 0, delta, finish_reason: null }] },
      { ...base, choices: [{ index: 0, delta: {}, finish_reason: finish }] },
      { ...base, choices: [], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } },
    ];
  }
}

// -- the agent process -------------------------------------------------------------------------

/** `webagents acp <folder>` with its stdout read line by line. */
class AgentProcess {
  private readonly child: ChildProcess;
  private readonly lines: Array<string | null> = [];
  private waiter: (() => void) | null = null;
  stderr = '';
  rawStdout: string[] = [];
  private readonly exited: Promise<number | null>;

  constructor(folder: string, env: NodeJS.ProcessEnv) {
    this.child = spawn(process.execPath, [...CLI_ARGS, 'acp', folder], { env, stdio: ['pipe', 'pipe', 'pipe'] });
    const rl = readline.createInterface({ input: this.child.stdout as NodeJS.ReadableStream, crlfDelay: Infinity });
    rl.on('line', (line) => {
      this.rawStdout.push(line);
      this.push(line);
    });
    rl.on('close', () => this.push(null));
    this.child.stderr?.on('data', (chunk: Buffer) => void (this.stderr += chunk.toString()));
    this.exited = new Promise((resolve) => this.child.on('exit', (code) => resolve(code)));
  }

  private push(line: string | null): void {
    this.lines.push(line);
    this.waiter?.();
  }

  sendRaw(text: string): void {
    this.child.stdin?.write(`${text}\n`);
  }

  send(message: Json): void {
    this.sendRaw(JSON.stringify(message));
  }

  async nextMessage(): Promise<Json> {
    const deadline = Date.now() + LINE_TIMEOUT_MS;
    while (this.lines.length === 0) {
      if (Date.now() > deadline) throw new Error(`no line from the agent within ${LINE_TIMEOUT_MS}ms; stderr so far: ${this.stderr}`);
      await new Promise<void>((resolve) => {
        this.waiter = resolve;
        setTimeout(resolve, 200);
      });
      this.waiter = null;
    }
    const line = this.lines.shift() as string | null;
    if (line === null) throw new Error(`the agent exited; stderr: ${this.stderr}`);
    return JSON.parse(line) as Json;
  }

  async close(): Promise<number | null> {
    this.child.stdin?.end();
    const code = await Promise.race([this.exited, sleep(LINE_TIMEOUT_MS).then(() => 'timeout' as const)]);
    if (code === 'timeout') {
      this.child.kill();
      throw new Error(`the agent did not exit after stdin closed; stderr: ${this.stderr}`);
    }
    return code;
  }
}

// -- matching --------------------------------------------------------------------------------

type Captures = Record<string, unknown> & { cwd: string; mcp_command: string; mcp_args: string[] };

/** Placeholders in a message to SEND, from what was captured. */
function substitute(value: unknown, captures: Captures): unknown {
  if (typeof value === 'string') {
    if (value.startsWith('<') && value.endsWith('>') && value.slice(1, -1) in captures) return captures[value.slice(1, -1)];
    return value.includes('<cwd>') ? value.replace(/<cwd>/g, captures.cwd) : value;
  }
  if (Array.isArray(value)) return value.map((v) => substitute(v, captures));
  if (isRecord(value)) return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, substitute(v, captures)]));
  return value;
}

/** Null when `actual` has the shape of `expected` (placeholders captured and held), else what differs. */
function matches(expected: unknown, actual: unknown, captures: Captures, at = '$'): string | null {
  if (typeof expected === 'string' && expected.startsWith('<') && expected.endsWith('>')) {
    const name = expected.slice(1, -1);
    if (name === 'any') return null;
    if (name === 'version') return typeof actual === 'string' && actual ? null : `${at}: not a version: ${JSON.stringify(actual)}`;
    if (name === 'capabilities') return JSON.stringify(actual) === JSON.stringify(PROTOCOL.agent_capabilities) ? null : `${at}: capabilities ${JSON.stringify(actual)}`;
    if (name === 'auth_methods') return JSON.stringify(actual) === JSON.stringify(PROTOCOL.auth_methods) ? null : `${at}: authMethods ${JSON.stringify(actual)}`;
    if (name === 'permission_options') return JSON.stringify(actual) === JSON.stringify(PROTOCOL.permission.options) ? null : `${at}: options ${JSON.stringify(actual)}`;
    if (name in captures) return JSON.stringify(captures[name]) === JSON.stringify(actual) ? null : `${at}: <${name}> was ${JSON.stringify(captures[name])}, now ${JSON.stringify(actual)}`;
    if (name === 'sessionId' && !(typeof actual === 'string' && actual.startsWith(PROTOCOL.session_id_prefix))) return `${at}: not a session id: ${JSON.stringify(actual)}`;
    if (name === 'toolCallId' && !(typeof actual === 'string' && actual)) return `${at}: not a tool call id: ${JSON.stringify(actual)}`;
    if (name === 'requestId' && !Number.isInteger(actual)) return `${at}: not a request id: ${JSON.stringify(actual)}`;
    captures[name] = actual;
    return null;
  }
  if (isRecord(expected) && Object.keys(expected).join() === '<contains>') {
    const needle = expected['<contains>'] as string;
    return typeof actual === 'string' && actual.includes(needle) ? null : `${at}: ${JSON.stringify(actual)} lacks ${JSON.stringify(needle)}`;
  }
  if (isRecord(expected)) {
    if (!isRecord(actual)) return `${at}: expected an object, got ${JSON.stringify(actual)}`;
    const wanted = Object.keys(expected).sort();
    const got = Object.keys(actual).sort();
    if (wanted.join() !== got.join()) return `${at}: keys ${JSON.stringify(got)} != ${JSON.stringify(wanted)}`;
    for (const key of wanted) {
      const problem = matches(expected[key], actual[key], captures, `${at}.${key}`);
      if (problem) return problem;
    }
    return null;
  }
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length !== expected.length) return `${at}: expected ${expected.length} items, got ${JSON.stringify(actual)}`;
    for (let i = 0; i < expected.length; i += 1) {
      const problem = matches(expected[i], actual[i], captures, `${at}[${i}]`);
      if (problem) return problem;
    }
    return null;
  }
  if (typeof expected === 'string' && expected.includes('<cwd>')) expected = expected.replace(/<cwd>/g, captures.cwd);
  return expected === actual ? null : `${at}: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`;
}

const updateOf = (line: Json): Json | null =>
  line.method === 'session/update' && isRecord(line.params) && isRecord(line.params.update) ? line.params.update : null;

/** Consecutive `agent_message_chunk` updates as one, whatever the split. */
function mergeChunks(lines: Json[]): Json[] {
  const merged: Json[] = [];
  for (const line of lines) {
    const update = updateOf(line);
    const last = merged[merged.length - 1];
    const previous = last ? updateOf(last) : null;
    const text = (u: Json | null) => (u && isRecord(u.content) && u.content.type === 'text' && u.sessionUpdate === 'agent_message_chunk' ? (u.content.text as string) : null);
    if (update && previous && text(update) !== null && text(previous) !== null) {
      const copy = JSON.parse(JSON.stringify(last)) as Json;
      ((copy.params as Json).update as Json).content = { type: 'text', text: `${text(previous)}${text(update)}` };
      merged[merged.length - 1] = copy;
      continue;
    }
    merged.push(line);
  }
  return merged;
}

async function runSteps(agent: AgentProcess, steps: Step[], captures: Captures, stub: StubModel): Promise<void> {
  for (const step of steps) {
    let requestId: unknown = null;
    if (step.send_raw !== undefined) {
      agent.sendRaw(step.send_raw);
    } else {
      const message = substitute(step.send, captures) as Json;
      agent.send(message);
      requestId = message.id ?? null;
    }
    if (step.then_send) {
      // A cancel must land while the model is being asked, not before the
      // request left, or it cancels nothing and the case races the stub.
      const deadline = Date.now() + LINE_TIMEOUT_MS;
      while (step.then_send_after_call && stub.calls.length < step.then_send_after_call && Date.now() < deadline) await sleep(20);
      agent.send(substitute(step.then_send, captures) as Json);
    }
    const replies = step.expect.filter((e) => 'reply' in e);
    const received: Json[] = [];
    for (;;) {
      const line = await agent.nextMessage();
      received.push(line);
      if ('method' in line && 'id' in line) {
        // A request from the agent: answer it as the fixture says.
        const spec = replies.shift();
        expect(spec, `unexpected request from the agent: ${JSON.stringify(line)}`).toBeDefined();
        const { reply, ...shape } = spec as Json;
        const problem = matches(shape, line, captures);
        expect(problem, JSON.stringify(line)).toBeNull();
        agent.send({ jsonrpc: '2.0', id: line.id, result: substitute(reply, captures) });
        continue;
      }
      if ('id' in line && ('result' in line || 'error' in line) && (requestId === null || line.id === requestId)) break;
    }
    const merged = mergeChunks(received);
    expect(merged.length, `lines:\n${merged.map((l) => JSON.stringify(l)).join('\n')}`).toBe(step.expect.length);
    step.expect.forEach((e, i) => {
      const { reply: _reply, ...shape } = e;
      const problem = matches(shape, merged[i], captures);
      expect(problem, `line: ${JSON.stringify(merged[i])}`).toBeNull();
    });
    for (const check of step.model_sees ?? []) {
      const text = JSON.stringify(stub.calls[check.call - 1]?.messages);
      for (const needle of check.messages_contain) expect(text, `the model's call ${check.call}`).toContain(needle);
    }
  }
}

// -- the cases ---------------------------------------------------------------------------------

const stub = new StubModel();
beforeAll(() => stub.listen());
afterAll(() => stub.close());

function makeEnv(home: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env, ...TRANSCRIPTS.env, HOME: home, [TRANSCRIPTS.base_url_env]: stub.baseUrl };
  delete env.WEBAGENTS_PROFILE;
  delete env.WEBAGENTS_TOKEN;
  delete env.WEBAGENTS_DEBUG;
  return env;
}

// Skipped, with the reason in the title, where the CLI cannot be spawned and
// driven over stdio (tests/helpers/cli.ts, 2026-09-26): from the portal root
// these ran with the portal's tsconfig and failed for that reason alone.
const prerequisite = cliPrerequisite();

describe.skipIf(prerequisite !== null).each(TRANSCRIPTS.cases)(
  prerequisite ? `transcript $name (SKIPPED: ${prerequisite})` : 'transcript $name',
  (testCase) => {
  it(testCase.about ?? testCase.name, async () => {
    const folder = tempDir('wa-acp-project-');
    writeFileSync(path.join(folder, 'AGENT.md'), TRANSCRIPTS.agent_file);
    const home = tempDir('wa-acp-home-');
    const env = makeEnv(home);
    stub.reset(testCase.model);
    const captures: Captures = { cwd: folder, mcp_command: process.execPath, mcp_args: [ECHO_SERVER] };

    const agent = new AgentProcess(folder, env);
    let code: number | null;
    try {
      await runSteps(agent, testCase.steps, captures, stub);
    } finally {
      code = await agent.close();
    }
    expect(code, agent.stderr).toBe(0);
    expect(agent.stderr).toContain(PROTOCOL.startup);
    for (const line of agent.rawStdout) {
      expect((JSON.parse(line) as Json).jsonrpc, `not a protocol message on stdout: ${line}`).toBe('2.0');
    }

    if (testCase.restart) {
      const second = new AgentProcess(folder, env);
      try {
        await runSteps(second, testCase.after_restart ?? [], captures, stub);
      } finally {
        expect(await second.close(), second.stderr).toBe(0);
      }
    }

    for (const [name, content] of Object.entries(testCase.files_after ?? {})) {
      expect(readFileSync(path.join(folder, name), 'utf8')).toBe(content);
    }
    for (const name of testCase.files_absent ?? []) expect(existsSync(path.join(folder, name)), name).toBe(false);
  }, 180_000);
});
