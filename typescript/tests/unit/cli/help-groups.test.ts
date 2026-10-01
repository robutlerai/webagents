/**
 * The root help's sections (2026-09-29), against the shared fixture
 * `python/tests/fixtures/cli/help_groups.json`, which the Python suite reads
 * too (`tests/cli/test_help_groups.py`). `index.ts` runs the CLI when it is
 * imported, so this reads what the real CLI prints: the headings and the
 * commands under each, in order, no "Commands:" left over, the hidden
 * commands not listed and still working.
 */

import { describe, expect, it } from 'vitest';
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { CLI_ARGS, TSX_PROBLEM, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/help_groups.json'), 'utf8')) as {
  groups: Array<[string, string[]]>;
  hidden: string[];
};
const tempDir = tempDirs();

function runCli(args: string[]): Promise<{ code: number | null; out: string; err: string }> {
  const home = tempDir('wa-help-groups-');
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [...CLI_ARGS, ...args], {
      cwd: home,
      env: { ...process.env, HOME: home, WEBAGENTS_PROFILE: '', WEBAGENTS_SECRETS_BACKEND: 'file' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => (out += d));
    child.stderr.on('data', (d) => (err += d));
    child.on('close', (code) => resolve({ code, out, err }));
  });
}

describe('the root help', () => {
  it.skipIf(TSX_PROBLEM !== null)('draws the fixture sections in order, and lists no hidden command', async () => {
    const { code, out } = await runCli(['--help']);
    expect(code).toBe(0);
    const at = FIXTURE.groups.map(([heading]) => out.indexOf(`\n${heading}\n`));
    expect(at.every((i) => i >= 0)).toBe(true);
    expect([...at].sort((a, b) => a - b)).toEqual(at);
    expect(out).not.toContain('\nCommands:\n');
    for (const [heading, names] of FIXTURE.groups) {
      const section = out.slice(out.indexOf(`\n${heading}\n`) + 1).split('\n\n')[0];
      const listed = section
        .split('\n')
        .slice(1)
        .filter((line) => line.startsWith('  ') && !line.startsWith('   '))
        .map((line) => line.trim().split(/\s+/)[0]);
      expect(listed).toEqual(names);
    }
    for (const hidden of FIXTURE.hidden) expect(out).not.toContain(`\n  ${hidden} `);
  }, 60_000);

  it.skipIf(TSX_PROBLEM !== null)('keeps the hidden commands working', async () => {
    expect((await runCli(['templates', 'list'])).code).toBe(0);
    expect((await runCli(['init', '--list'])).out).toContain('Available Templates');
    expect((await runCli(['connect', '--help'])).code).toBe(0);
  }, 60_000);
});
