/**
 * The `robutler` packages (`packages/robutler-npm`, `packages/robutler-py`)
 * are webagents under another name: the same version, a dependency on exactly
 * that version, and no code. `scripts/thin-packages.mjs` is the check; this
 * runs it on the repository, and on copies with one thing wrong each.
 */

import { describe, expect, it, afterAll } from 'vitest';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
const SCRIPT = path.join(REPO, 'scripts/thin-packages.mjs');
const FILES = [
  'typescript/package.json',
  'python/pyproject.toml',
  'packages/robutler-npm/package.json',
  'packages/robutler-npm/bin/robutler.js',
  'packages/robutler-py/pyproject.toml',
];

const copies: string[] = [];
afterAll(() => {
  for (const dir of copies) fs.rmSync(dir, { recursive: true, force: true });
});

/** A copy of the files the check reads, with `edit` applied to one of them. */
function copy(file?: string, edit?: (text: string) => string): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'thin-packages-'));
  copies.push(root);
  for (const rel of FILES) {
    fs.mkdirSync(path.dirname(path.join(root, rel)), { recursive: true });
    const text = fs.readFileSync(path.join(REPO, rel), 'utf8');
    fs.writeFileSync(path.join(root, rel), rel === file && edit ? edit(text) : text);
  }
  return root;
}

function check(root: string, ...args: string[]): { code: number | null; err: string } {
  const r = spawnSync(process.execPath, [SCRIPT, '--root', root, ...args], { encoding: 'utf8' });
  return { code: r.status, err: r.stderr };
}

describe('the robutler packages', () => {
  it('agree with webagents in this repository', () => {
    expect(check(REPO)).toEqual({ code: 0, err: '' });
  });

  it('the npm one runs the command webagents ships, and nothing else', () => {
    const thin = JSON.parse(fs.readFileSync(path.join(REPO, 'packages/robutler-npm/package.json'), 'utf8'));
    const main = JSON.parse(fs.readFileSync(path.join(REPO, 'typescript/package.json'), 'utf8'));
    expect(thin.dependencies).toEqual({ webagents: main.version });
    expect(thin.files).toEqual(['bin', 'README.md', 'LICENSE']);
    const bin = fs.readFileSync(path.join(REPO, 'packages/robutler-npm/bin/robutler.js'), 'utf8').trim().split('\n');
    expect(bin).toHaveLength(3);
    expect(bin[0]).toBe('#!/usr/bin/env node');
    expect(bin[2]).toBe("import 'webagents/cli/robutler';");
  });

  it.each([
    ['a version behind', 'packages/robutler-npm/package.json', (t: string) => t.replace(/"version": "[^"]+"/, '"version": "0.0.1"'), 'npm: robutler is 0.0.1'],
    ['a range instead of a pin', 'packages/robutler-npm/package.json', (t: string) => t.replace(/"webagents": "([^"]+)"/, '"webagents": "^$1"'), 'must depend on exactly webagents'],
    ['a second dependency', 'packages/robutler-npm/package.json', (t: string) => t.replace('"dependencies": {', '"dependencies": {\n    "left-pad": "1.0.0",'), 'must depend on exactly webagents'],
    ['webagents bumped alone', 'typescript/package.json', (t: string) => t.replace(/"version": "[^"]+"/, '"version": "9.9.9"'), 'webagents is 9.9.9'],
    ['the export gone', 'typescript/package.json', (t: string) => t.replace('"./cli/robutler"', '"./cli/gone"'), 'must export ./cli/robutler'],
    ['a bin that runs something else', 'packages/robutler-npm/bin/robutler.js', (t: string) => t.replace('webagents/cli/robutler', 'webagents'), "must import 'webagents/cli/robutler'"],
    ['a PyPI version behind', 'packages/robutler-py/pyproject.toml', (t: string) => t.replace(/^version = "[^"]+"$/m, 'version = "0.0.1"'), 'pypi: robutler is 0.0.1'],
    ['a PyPI range', 'packages/robutler-py/pyproject.toml', (t: string) => t.replace('webagents==', 'webagents>='), 'must depend on exactly webagents=='],
    ['a console script of its own', 'packages/robutler-py/pyproject.toml', (t: string) => `${t}\n[project.scripts]\nrobutler = "x:main"\n`, 'must install no command of its own'],
    ['the Python webagents bumped alone', 'python/pyproject.toml', (t: string) => t.replace(/^version = "[^"]+"$/m, 'version = "9.9.9"'), 'webagents is 9.9.9'],
  ])('fail with %s', (_name, file, edit, message) => {
    const r = check(copy(file, edit));
    expect(r.code).toBe(1);
    expect(r.err).toContain(message);
  });

  it('are checked against a released version, and written to one', () => {
    const root = copy();
    expect(check(root, '--only', 'npm', '--npm', '7.0.0').code).toBe(1);
    expect(check(root, '--write', '--npm', '7.0.0', '--pypi', '7.0.1').code).toBe(0);
    expect(check(root, '--only', 'npm', '--npm', '7.0.0').code).toBe(0);
    expect(check(root, '--only', 'pypi', '--pypi', '7.0.1').code).toBe(0);
    expect(JSON.parse(fs.readFileSync(path.join(root, 'packages/robutler-npm/package.json'), 'utf8')).dependencies).toEqual({ webagents: '7.0.0' });
    expect(fs.readFileSync(path.join(root, 'packages/robutler-py/pyproject.toml'), 'utf8')).toContain('dependencies = ["webagents==7.0.1"]');
    // The sources were not bumped, so the plain check now disagrees.
    expect(check(root).code).toBe(1);
  });
});
