/**
 * `webagents secrets set|list|remove|get` (S-292, 2026-09-26), pinned by
 * `python/tests/fixtures/cli/secrets.json`, which the Python suite runs too
 * (`tests/cli/test_secrets_cli_mcpsecrets.py`): the help words, the name
 * grammar (the `${secret:NAME}` grammar), the sentences, and a round trip
 * through the keystore's FILE fallback in a scratch HOME with the CLI
 * spawned, the value piped in, never given as an argument and never printed
 * back except by `get --show`.
 */

import { describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { REFERENCE_NAME } from '../../../src/skills/secrets/references';
import { CLI_SOURCE, TSX_CLI, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/secrets.json'), 'utf8')) as {
  help: Record<string, string>;
  name_pattern: string;
  names: { accepted: string[]; refused: string[] };
  lines: Record<string, unknown>;
  round_trip: { file: string; value: string; steps: { argv: string[]; stdin?: string; stdout?: string[]; stderr?: string[]; exit: number }[] };
};

const tempDir = tempDirs();

/** The CLI, spawned under `home` with the file backend and no key exported. */
function cli(home: string, args: string[], stdin?: string) {
  const env = { ...process.env, HOME: home, WEBAGENTS_SECRETS_BACKEND: 'file' } as Record<string, string | undefined>;
  for (const name of ['WEBAGENTS_PROFILE', 'WEBAGENTS_TOKEN', 'WEBAGENTS_SECRETS_DIR', 'GITHUB_TOKEN', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'FIREWORKS_API_KEY', 'WEBAGENTS_DEBUG']) {
    delete env[name];
  }
  return spawnSync(process.execPath, [TSX_CLI, CLI_SOURCE, ...args], {
    cwd: home,
    env: env as NodeJS.ProcessEnv,
    encoding: 'utf-8',
    // A pipe, never a terminal: `set` reads the value from it.
    input: stdin ?? '',
    timeout: 60_000,
  });
}

describe('the help words match the fixture', () => {
  it('the group, each command, and --show', () => {
    // The `secrets` block of index.ts only: other groups declare a `list` too.
    const whole = fs.readFileSync(CLI_SOURCE, 'utf8');
    const start = whole.indexOf("const secretsCmd = program.command('secrets')");
    const end = whole.indexOf('// ====', start);
    const source = whole.slice(start, end);
    const declared = (pattern: RegExp) => source.match(pattern)?.[1];
    expect(declared(/program\.command\('secrets'\)\.description\('([^']+)'\)/)).toBe(FIXTURE.help.group);
    expect(declared(/\.command\('list'\)\s*\.description\('([^']+)'\)/)).toBe(FIXTURE.help.list);
    expect(declared(/\.command\('set <name>'\)\s*\.description\('([^']+)'\)/)).toBe(FIXTURE.help.set);
    expect(declared(/\.command\('remove <name>'\)\.description\('([^']+)'\)/)).toBe(FIXTURE.help.remove);
    expect(declared(/\.command\('unset <name>', \{ hidden: true \}\)\s*\.description\('([^']+)'\)/)).toBe(FIXTURE.help.unset);
    expect(declared(/\.command\('get <name>'\)\s*\.description\('([^']+)'\)/)).toBe(FIXTURE.help.get);
    expect(declared(/\.option\('--show', '([^']+)'\)/)).toBe(FIXTURE.help.get_show);
  });

  it('unset is hidden from the help and remove is shown', () => {
    const out = cli(tempDir('wa-secrets-help-'), ['secrets', '-h']);
    expect(out.status).toBe(0);
    expect(out.stdout).toContain('remove');
    expect(out.stdout).not.toContain('unset');
  });
});

describe('the name grammar is the reference grammar', () => {
  it('accepts and refuses what the fixture says', () => {
    expect(REFERENCE_NAME.source).toBe(FIXTURE.name_pattern);
    for (const name of FIXTURE.names.accepted) expect(REFERENCE_NAME.test(name), name).toBe(true);
    for (const name of FIXTURE.names.refused) expect(REFERENCE_NAME.test(name), name).toBe(false);
  });
});

describe('the round trip through the file fallback', () => {
  it('runs every step of the fixture, in order, and never prints the value except under --show', () => {
    const home = tempDir('wa-secrets-home-');
    const { value, steps, file } = FIXTURE.round_trip;
    for (const step of steps) {
      const out = cli(home, step.argv, step.stdin);
      expect(out.status, `${step.argv.join(' ')}: ${out.stdout}${out.stderr}`).toBe(step.exit);
      if (step.stdout) expect(out.stdout.replace(/\n$/, '').split('\n'), step.argv.join(' ')).toEqual(step.stdout);
      for (const line of step.stderr ?? []) expect(out.stderr, step.argv.join(' ')).toContain(line);
      if (step.argv[step.argv.length - 1] !== '--show') {
        expect(out.stdout + out.stderr, step.argv.join(' ')).not.toContain(value);
      }
    }
    const stored = path.join(home, file);
    expect(fs.existsSync(stored)).toBe(true);
    expect((fs.statSync(stored).mode & 0o777).toString(8)).toBe('600');
    expect(JSON.parse(fs.readFileSync(stored, 'utf8'))).toEqual({});
  });

  it('set writes the owner-only file and list names it without the value', () => {
    const home = tempDir('wa-secrets-home-');
    const { value, file } = FIXTURE.round_trip;
    expect(cli(home, ['secrets', 'set', 'GITHUB_TOKEN'], `${value}\n`).status).toBe(0);
    expect(JSON.parse(fs.readFileSync(path.join(home, file), 'utf8'))).toEqual({ GITHUB_TOKEN: value });
    const listing = cli(home, ['secrets', 'list']);
    expect(listing.stdout).toContain('GITHUB_TOKEN');
    expect(listing.stdout).not.toContain(value);
  });

  it('a refused name stores nothing', () => {
    const home = tempDir('wa-secrets-home-');
    for (const name of FIXTURE.names.refused.filter(Boolean)) {
      const out = cli(home, ['secrets', 'set', name], 'x\n');
      expect(out.status).toBe(1);
      expect(out.stderr).toContain(String(FIXTURE.lines.bad_name).replace('{name}', name));
    }
    expect(fs.existsSync(path.join(home, FIXTURE.round_trip.file))).toBe(false);
  });

  it('the environment wins over a stored value, and says so', () => {
    const home = tempDir('wa-secrets-home-');
    const env = { ...process.env, HOME: home, WEBAGENTS_SECRETS_BACKEND: 'file', GITHUB_TOKEN: 'from-the-shell' };
    delete env.WEBAGENTS_PROFILE;
    const set = spawnSync(process.execPath, [TSX_CLI, CLI_SOURCE, 'secrets', 'set', 'GITHUB_TOKEN'], { cwd: home, env, encoding: 'utf-8', input: 'stored-dummy\n', timeout: 60_000 });
    expect(set.status).toBe(0);
    expect(set.stdout).toContain(String(FIXTURE.lines.environment_wins).replace('{name}', 'GITHUB_TOKEN'));
    const listing = spawnSync(process.execPath, [TSX_CLI, CLI_SOURCE, 'secrets', 'list'], { cwd: home, env, encoding: 'utf-8', input: '', timeout: 60_000 });
    expect(listing.stdout).toContain((FIXTURE.lines.list_where as Record<string, string>).shell_over_stored);
  });
});
