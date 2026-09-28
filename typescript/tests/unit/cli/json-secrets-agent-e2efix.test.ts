/**
 * `--json` is honoured by `secrets list` and by the `-a <unknown agent>`
 * refusal (2026-09-26, the new-developer e2e run: both ignored it). Pinned by
 * `list_json` in `python/tests/fixtures/cli/secrets.json` and by
 * `python/tests/fixtures/cli/json_errors.json`, which the Python suite runs
 * too (`tests/cli/test_json_secrets_agent_e2efix.py`). The CLI is spawned
 * under a scratch HOME with the file backend; the value is piped in.
 */

import { describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { CLI_SOURCE, TSX_CLI, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures/cli');
const LIST = (JSON.parse(fs.readFileSync(path.join(FIXTURES, 'secrets.json'), 'utf8')) as { list_json: ListJson }).list_json;
const ERRORS = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'json_errors.json'), 'utf8')) as {
  agent_not_found: { code: string; message: string; exit: number };
};
interface ListJson {
  fields: string[];
  case: { stored: string; in_shell: string; data: { keys: Record<string, unknown>[]; complete: boolean } };
}

const tempDir = tempDirs();

function cli(home: string, cwd: string, args: string[], extra: Record<string, string> = {}, stdin = '') {
  const env = { ...process.env, HOME: home, WEBAGENTS_SECRETS_BACKEND: 'file', ROBUTLER_API_URL: 'http://127.0.0.1:9', ...extra } as Record<string, string | undefined>;
  for (const name of ['WEBAGENTS_PROFILE', 'WEBAGENTS_TOKEN', 'WEBAGENTS_SECRETS_DIR', 'WEBAGENTS_DEBUG', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'FIREWORKS_API_KEY']) {
    if (!(name in extra)) delete env[name];
  }
  return spawnSync(process.execPath, [TSX_CLI, CLI_SOURCE, ...args], { cwd, env: env as NodeJS.ProcessEnv, encoding: 'utf-8', input: stdin, timeout: 60_000 });
}

describe('--json secrets list', () => {
  it('answers one document with the fixture’s rows', () => {
    const home = tempDir('wa-json-home-');
    const stored = cli(home, home, ['secrets', 'set', LIST.case.stored], {}, 'dummy-value-not-a-credential\n');
    expect(stored.status, stored.stderr).toBe(0);
    const listed = cli(home, home, ['--json', 'secrets', 'list'], { [LIST.case.in_shell]: 'sk-from-the-shell' });
    expect(listed.status, listed.stderr).toBe(0);
    const document = JSON.parse(listed.stdout) as { ok: boolean; data: ListJson['case']['data'] };
    expect(document.ok).toBe(true);
    expect(document.data).toEqual(LIST.case.data);
    for (const key of document.data.keys) expect(Object.keys(key)).toEqual(LIST.fields);
    expect(listed.stdout).not.toContain('dummy-value');
  });
});

describe('--json -a <unknown agent>', () => {
  it('answers the error envelope and exits 1', () => {
    const home = tempDir('wa-json-home-');
    const project = tempDir('wa-json-project-');
    fs.writeFileSync(path.join(project, 'AGENT.md'), '---\nname: my-agent\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nBody\n');
    const result = cli(home, project, ['--json', '-a', 'nosuch', '-p', 'hi'], { OPENAI_API_KEY: 'sk-dummy' });
    expect(result.status).toBe(ERRORS.agent_not_found.exit);
    const document = JSON.parse(result.stdout) as { ok: boolean; error: { code: string; message: string } };
    expect(document.ok).toBe(false);
    expect(document.error.code).toBe(ERRORS.agent_not_found.code);
    expect(document.error.message).toBe(ERRORS.agent_not_found.message.replace('{name}', 'nosuch').replace('{agents}', 'my-agent'));
  });
});
