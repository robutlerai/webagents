/**
 * `--json` is honoured by `init`, `templates list`, `models`, `skills add`
 * and `publish --dry-run`, and with `-p` it is `--output-format json`
 * (2026-09-27, the final e2e re-run: all of them ignored the flag). The
 * documents are pinned by `python/tests/fixtures/cli/json_documents.json` and
 * the refusals by `cli/json_errors.json`, which the Python suite runs too
 * (`tests/cli/test_json_commands_finalfix.py`). The CLI is spawned under a
 * scratch HOME with the file backend, the platform at a loopback stand-in,
 * and for `-p` a fake OpenAI server on loopback that answers "You said: hi."
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { spawn, spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { CLI_ARGS, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures/cli');
const read = (name: string) => JSON.parse(fs.readFileSync(path.join(FIXTURES, name), 'utf8'));
const DOCS = read('json_documents.json') as {
  templates_list: { data_keys: string[]; row_keys: string[] };
  models: { data_keys: string[]; row_keys: string[] };
  init: { data_keys: string[]; files: string[] };
  skills_add: { data_keys: string[] };
  publish_dry_run: { data_keys: string[] };
  prompt: { case: { argv: string[]; content: string; explicit_wins: string[] } };
};
const ERRORS = read('json_errors.json') as Record<string, { code: string; message: string; exit: number; unknown_skill_starts?: string }>;
const TEMPLATES = Object.keys((read('init_templates.json') as { templates: Record<string, unknown> }).templates);

const tempDir = tempDirs();
const AGENT_MD = '---\nname: my-agent\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nBody\n';

function cliEnv(home: string, extra: Record<string, string>): NodeJS.ProcessEnv {
  const env = { ...process.env, HOME: home, WEBAGENTS_SECRETS_BACKEND: 'file', ROBUTLER_API_URL: 'http://127.0.0.1:9', ...extra } as Record<string, string | undefined>;
  for (const name of ['WEBAGENTS_PROFILE', 'WEBAGENTS_TOKEN', 'WEBAGENTS_SECRETS_DIR', 'WEBAGENTS_DEBUG', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'FIREWORKS_API_KEY']) {
    if (!(name in extra)) delete env[name];
  }
  return env as NodeJS.ProcessEnv;
}

function cli(home: string, cwd: string, args: string[], extra: Record<string, string> = {}) {
  return spawnSync(process.execPath, [...CLI_ARGS, ...args], { cwd, env: cliEnv(home, extra), encoding: 'utf-8', input: '', timeout: 90_000 });
}

/**
 * The same, without blocking this process: the `-p` runs talk to the fake
 * model server THIS process hosts, which `spawnSync` would starve.
 */
function cliAsync(home: string, cwd: string, args: string[], extra: Record<string, string> = {}): Promise<{ status: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [...CLI_ARGS, ...args], { cwd, env: cliEnv(home, extra), stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk: Buffer) => (stdout += chunk.toString()));
    child.stderr.on('data', (chunk: Buffer) => (stderr += chunk.toString()));
    const timer = setTimeout(() => child.kill('SIGKILL'), 60_000);
    child.on('close', (status) => {
      clearTimeout(timer);
      resolve({ status, stdout, stderr });
    });
  });
}

function document(stdout: string): { ok: boolean; data?: Record<string, unknown>; error?: { code: string; message: string } } {
  // ONE document and nothing else on stdout.
  return JSON.parse(stdout);
}

function project(): { home: string; cwd: string } {
  const home = tempDir('wa-json-home-');
  const cwd = tempDir('wa-json-project-');
  fs.writeFileSync(path.join(cwd, 'AGENT.md'), AGENT_MD);
  return { home, cwd };
}

describe('--json templates list', () => {
  it('answers the templates as one document', () => {
    const { home, cwd } = project();
    const res = cli(home, cwd, ['--json', 'templates', 'list']);
    expect(res.status, res.stderr).toBe(0);
    const doc = document(res.stdout);
    expect(doc.ok).toBe(true);
    expect(Object.keys(doc.data!)).toEqual(DOCS.templates_list.data_keys);
    const rows = doc.data!.templates as Array<Record<string, unknown>>;
    expect(rows.map((row) => row.name)).toEqual(TEMPLATES);
    for (const row of rows) expect(Object.keys(row)).toEqual(DOCS.templates_list.row_keys);
  });
});

describe('--json models', () => {
  it('answers the providers as one document', () => {
    const { home, cwd } = project();
    const res = cli(home, cwd, ['--json', 'models'], { OPENAI_API_KEY: 'sk-dummy' });
    expect(res.status, res.stderr).toBe(0);
    const doc = document(res.stdout);
    expect(doc.ok).toBe(true);
    expect(Object.keys(doc.data!)).toEqual(DOCS.models.data_keys);
    const rows = doc.data!.providers as Array<Record<string, unknown>>;
    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) {
      expect(Object.keys(row)).toEqual(DOCS.models.row_keys);
      expect(typeof row.ready).toBe('boolean');
    }
    expect(rows.find((row) => row.id === 'openai')?.ready).toBe(true);
    expect(rows.find((row) => row.id === 'anthropic')?.ready).toBe(false);
  });
});

describe('--json init', () => {
  it('creates the project and answers one document; a second run and a bad template are error envelopes', () => {
    const home = tempDir('wa-json-home-');
    const cwd = tempDir('wa-json-init-');
    const made = cli(home, cwd, ['--json', 'init', 'proj', '-t', 'chatbot']);
    expect(made.status, made.stderr).toBe(0);
    const doc = document(made.stdout);
    expect(doc.ok).toBe(true);
    expect(Object.keys(doc.data!)).toEqual(DOCS.init.data_keys);
    expect(doc.data!.name).toBe('proj');
    expect(doc.data!.template).toBe('chatbot');
    // The CLI resolves against its real working directory (a symlinked temp folder on macOS).
    expect(doc.data!.path).toBe(path.join(fs.realpathSync(cwd), 'proj'));
    expect(doc.data!.files).toEqual(DOCS.init.files);
    // The model the file names: none with no provider key here (B3, 2026-09-28).
    expect(doc.data!.model === null || typeof doc.data!.model === 'string').toBe(true);
    expect(fs.existsSync(path.join(cwd, 'proj', 'AGENT.md'))).toBe(true);

    const again = cli(home, cwd, ['--json', 'init', 'proj']);
    expect(again.status).toBe(ERRORS.directory_exists.exit);
    expect(document(again.stdout)).toEqual({ ok: false, error: { code: 'directory_exists', message: ERRORS.directory_exists.message.replace('{name}', 'proj') } });

    const typo = cli(home, cwd, ['--json', 'init', 'other', '-t', 'nope']);
    expect(typo.status).toBe(ERRORS.unknown_template.exit);
    expect(document(typo.stdout)).toEqual({
      ok: false,
      error: { code: 'unknown_template', message: ERRORS.unknown_template.message.replace('{template}', 'nope').replace('{templates}', TEMPLATES.join(', ')) },
    });
    expect(fs.existsSync(path.join(cwd, 'other'))).toBe(false);
  });
});

describe('--json skills add', () => {
  it('answers the edit as one document, and an unknown skill as the error envelope', () => {
    const { home, cwd } = project();
    const added = cli(home, cwd, ['--json', 'skills', 'add', 'todo']);
    expect(added.status, added.stderr).toBe(0);
    const doc = document(added.stdout);
    expect(doc.ok).toBe(true);
    expect(Object.keys(doc.data!)).toEqual(DOCS.skills_add.data_keys);
    expect(doc.data!.file).toBe('AGENT.md');
    expect(doc.data!.added).toEqual(['todo']);
    expect(doc.data!.already).toEqual([]);
    expect((doc.data!.messages as string[])[0]).toBe('Added todo to AGENT.md.');
    expect(fs.readFileSync(path.join(cwd, 'AGENT.md'), 'utf8')).toContain('- todo');

    const bad = cli(home, cwd, ['--json', 'skills', 'add', 'nosuchskill']);
    expect(bad.status).toBe(ERRORS.skills_add_failed.exit);
    const refusal = document(bad.stdout);
    expect(refusal.ok).toBe(false);
    expect(refusal.error!.code).toBe('skills_add_failed');
    expect(refusal.error!.message.startsWith(ERRORS.skills_add_failed.unknown_skill_starts!.replace('{name}', 'nosuchskill'))).toBe(true);
    expect(fs.readFileSync(path.join(cwd, 'AGENT.md'), 'utf8')).not.toContain('nosuchskill');
  });
});

describe('--json publish --dry-run', () => {
  it('answers the request that would have been sent, and sends nothing', () => {
    const { home, cwd } = project();
    const res = cli(home, cwd, ['--json', 'publish', '--dry-run']);
    expect(res.status, res.stderr).toBe(0);
    const doc = document(res.stdout);
    expect(doc.ok).toBe(true);
    expect(Object.keys(doc.data!)).toEqual(DOCS.publish_dry_run.data_keys);
    expect(doc.data!.method).toBe('POST');
    expect(doc.data!.url).toBe('http://127.0.0.1:9/api/agents');
    expect((doc.data!.body as { name: string }).name).toBe('my-agent');
    expect(doc.data!.created).toBe(true);
  });

  it('a folder with no agent file is the error envelope', () => {
    const home = tempDir('wa-json-home-');
    const cwd = tempDir('wa-json-empty-');
    const res = cli(home, cwd, ['--json', 'publish', '--dry-run']);
    expect(res.status).toBe(ERRORS.publish_failed.exit);
    const doc = document(res.stdout);
    expect(doc.ok).toBe(false);
    expect(doc.error!.code).toBe('publish_failed');
    expect(doc.error!.message).toContain('No AGENT.md');
  });
});

describe('--json -p', () => {
  let server: http.Server;
  let base = '';
  beforeAll(async () => {
    // A fake OpenAI: streams or answers "You said: <last user message>."
    server = http.createServer((req, res) => {
      let raw = '';
      req.on('data', (chunk) => (raw += chunk));
      req.on('end', () => {
        const body = JSON.parse(raw || '{}') as { stream?: boolean; messages?: Array<{ role: string; content: unknown }> };
        const last = [...(body.messages ?? [])].reverse().find((m) => m.role === 'user')?.content;
        const text = `You said: ${typeof last === 'string' ? last : JSON.stringify(last)}.`;
        const usage = { prompt_tokens: 12, completion_tokens: 6, total_tokens: 18 };
        if (body.stream) {
          res.writeHead(200, { 'Content-Type': 'text/event-stream' });
          const chunk = (delta: Record<string, unknown>, finish: string | null, extra: Record<string, unknown> = {}) =>
            `data: ${JSON.stringify({ id: 'cmpl-test', object: 'chat.completion.chunk', created: 1, model: 'gpt-4o-mini', choices: [{ index: 0, delta, finish_reason: finish }], ...extra })}\n\n`;
          res.write(chunk({ role: 'assistant', content: '' }, null));
          res.write(chunk({ content: text }, null));
          res.write(chunk({}, 'stop', { usage }));
          res.write('data: [DONE]\n\n');
          res.end();
          return;
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ id: 'cmpl-test', object: 'chat.completion', created: 1, model: 'gpt-4o-mini', choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop' }], usage }));
      });
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`;
  });
  afterAll(async () => {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });

  it('is --output-format json: the response document on stdout', async () => {
    const { home, cwd } = project();
    const res = await cliAsync(home, cwd, DOCS.prompt.case.argv, { OPENAI_API_KEY: 'sk-dummy', OPENAI_BASE_URL: base });
    expect(res.status, res.stderr).toBe(0);
    const response = JSON.parse(res.stdout) as { content: string };
    expect(response.content).toBe(DOCS.prompt.case.content);
  }, 90_000);

  it('an explicit --output-format wins', async () => {
    const { home, cwd } = project();
    const res = await cliAsync(home, cwd, DOCS.prompt.case.explicit_wins, { OPENAI_API_KEY: 'sk-dummy', OPENAI_BASE_URL: base });
    expect(res.status, res.stderr).toBe(0);
    const lines = res.stdout.trim().split('\n').map((line) => JSON.parse(line) as { type: string });
    expect(lines.length).toBeGreaterThan(1);
    expect(lines[lines.length - 1].type).toBe('done');
  }, 90_000);
});
