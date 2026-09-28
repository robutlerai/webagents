/**
 * Global options after the subcommand (the sandbox-default lane,
 * 2026-09-27): `webagents login --profile local` used to answer "unknown
 * option". The hoisting is pinned against the shared fixture
 * `python/tests/fixtures/cli/sandbox_default_global_options.json` that
 * `python/tests/cli/test_sandbox_default_global_options.py` reads too, and
 * proved on the real CLI: `config path --profile <name>` prints the
 * profile's own config path.
 */

import { describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { HOISTED_OPTIONS, hoistGlobalOptions } from '../../../src/cli/sandbox-default-argv';
import { CLI_ARGS, TSX_PROBLEM, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/sandbox_default_global_options.json'), 'utf8'));
const tempDir = tempDirs();

describe('hoisting the global options', () => {
  it('lifts the fixture options', () => {
    expect([...HOISTED_OPTIONS]).toEqual(FIXTURE.hoisted);
  });

  it.each(FIXTURE.cases as Array<{ argv: string[]; hoisted: string[] }>)('$argv', ({ argv, hoisted }) => {
    expect(hoistGlobalOptions(argv)).toEqual(hoisted);
  });
});

describe('the real CLI', () => {
  const real = TSX_PROBLEM ? it.skip : it;
  if (TSX_PROBLEM) console.warn(`real CLI test skipped: ${TSX_PROBLEM}`);

  function cliEnv(home: string): Record<string, string | undefined> {
    const env = { ...process.env, HOME: home, WEBAGENTS_SECRETS_BACKEND: 'file', NO_COLOR: '1', ROBUTLER_API_URL: 'http://127.0.0.1:9', OPENAI_API_KEY: 'sk-global-options-dummy' } as Record<string, string | undefined>;
    for (const name of ['WEBAGENTS_PROFILE', 'WEBAGENTS_TOKEN', 'WEBAGENTS_DEBUG', 'WEBAGENTS_NO_SANDBOX']) delete env[name];
    return env;
  }

  real('takes --profile after the subcommand', () => {
    const home = tempDir('wa-global-home-');
    const cwd = tempDir('wa-global-cwd-');
    const profile = 'sd-probe';
    const result = spawnSync(process.execPath, [...CLI_ARGS, 'config', 'path', '--profile', profile], { cwd, env: cliEnv(home), encoding: 'utf8', timeout: 120_000 });
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout).toContain((FIXTURE.real_cli.config_path_contains as string).replace('{profile}', profile));
  }, 150_000);

  real('takes --no-sandbox after the subcommand, and doctor reports the sandbox off for the run', () => {
    const home = tempDir('wa-global-home2-');
    const cwd = tempDir('wa-global-cwd2-');
    fs.writeFileSync(path.join(cwd, 'AGENT.md'), '---\nname: a\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n  - shell\n---\nBody\n');
    const result = spawnSync(process.execPath, [...CLI_ARGS, '--json', 'doctor', '--no-sandbox'], { cwd, env: cliEnv(home), encoding: 'utf8', timeout: 120_000 });
    // The CLI's one JSON document: `{ok, data: {checks}}` (`output.ts`, `emit`).
    const document = JSON.parse(result.stdout) as { data: { checks: Array<{ name: string; detail: string }> } };
    const sandbox = document.data.checks.find((check) => check.name === 'sandbox');
    expect(sandbox?.detail, result.stderr).toBe(FIXTURE.real_cli.doctor_sandbox_detail);
  }, 150_000);
});
