/**
 * PluginSkill: owner-only tools, and modules only from the plugin
 * directories (S-249 addendum 2, 2026-09-26).
 *
 * `plugin_load` was open to every caller and `import()`ed any path it was
 * given. Each case here would have passed before the fix in the other
 * direction: a module outside the directories loaded, a symlink planted
 * inside them reached outside, and a stranger could call the tool at all.
 */

import { describe, it, expect } from 'vitest';
import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { afterAll } from 'vitest';
import { PluginSkill } from '../../../src/skills/plugin/skill';
import { callerScopes, scopeAllows } from '../../../src/core/scopes';

const MODULE = 'export default class Loaded { async initialize() {} async cleanup() {} }\n';

const roots: string[] = [];
function root(): string {
  const dir = mkdtempSync(path.join(tmpdir(), 'wa-plugin-s249-'));
  roots.push(dir);
  return dir;
}
afterAll(() => {
  for (const dir of roots) rmSync(dir, { recursive: true, force: true });
});

function skillWith(pluginDir: string): PluginSkill {
  return new PluginSkill({ pluginDirs: [pluginDir] });
}

describe('PluginSkill tools are owner-only', () => {
  it('declares plugin_list, plugin_load and plugin_unload for the owner', () => {
    const skill = skillWith(root());
    const byName = new Map(skill.tools.map((t) => [t.name, t.scopes]));
    for (const name of ['plugin_list', 'plugin_load', 'plugin_unload']) {
      expect(byName.get(name), name).toEqual(['owner']);
    }
  });

  it('a verified caller who is not the owner is refused; the owner and an admin are not', () => {
    const skill = skillWith(root());
    const load = skill.tools.find((t) => t.name === 'plugin_load')!;
    expect(scopeAllows(load.scopes, callerScopes({ authenticated: false }))).toBe(false);
    expect(scopeAllows(load.scopes, callerScopes({ authenticated: true, scope: 'user' as never }))).toBe(false);
    expect(scopeAllows(load.scopes, callerScopes({ authenticated: true, scope: 'owner' as never }))).toBe(true);
    expect(scopeAllows(load.scopes, callerScopes({ authenticated: true, scope: 'admin' as never }))).toBe(true);
  });
});

describe('plugin_load takes modules only from the plugin directories', () => {
  it('loads a module inside a plugin directory', async () => {
    const base = root();
    const plugins = path.join(base, 'plugins');
    mkdirSync(plugins);
    writeFileSync(path.join(plugins, 'good.mjs'), MODULE);
    const skill = skillWith(plugins);
    const answer = await skill.pluginLoad({ path: path.join(plugins, 'good.mjs'), name: 'good' }, {} as never);
    expect(answer).toBe('Plugin good loaded');
    expect((await skill.pluginList({}, {} as never)).loaded.map((p) => p.name)).toEqual(['good']);
  });

  it('refuses a module outside every plugin directory, and imports nothing', async () => {
    const base = root();
    const plugins = path.join(base, 'plugins');
    const elsewhere = path.join(base, 'elsewhere');
    mkdirSync(plugins);
    mkdirSync(elsewhere);
    writeFileSync(path.join(elsewhere, 'evil.mjs'), 'globalThis.__s249_loaded = true; export default class E { async initialize() {} }\n');
    const skill = skillWith(plugins);
    const answer = await skill.pluginLoad({ path: path.join(elsewhere, 'evil.mjs'), name: 'evil' }, {} as never);
    expect(answer).toMatch(/^Failed to load evil: .* is not a file inside a plugin directory/);
    expect((globalThis as { __s249_loaded?: boolean }).__s249_loaded).toBeUndefined();
    expect((await skill.pluginList({}, {} as never)).loaded).toEqual([]);
  });

  it('refuses `..` out of the directory', async () => {
    const base = root();
    const plugins = path.join(base, 'plugins');
    mkdirSync(plugins);
    writeFileSync(path.join(base, 'outside.mjs'), MODULE);
    const skill = skillWith(plugins);
    const answer = await skill.pluginLoad({ path: path.join(plugins, '..', 'outside.mjs'), name: 'outside' }, {} as never);
    expect(answer).toMatch(/is not a file inside a plugin directory/);
  });

  it('refuses a symlink planted inside the directory that points outside it', async () => {
    const base = root();
    const plugins = path.join(base, 'plugins');
    mkdirSync(plugins);
    writeFileSync(path.join(base, 'outside.mjs'), MODULE);
    symlinkSync(path.join(base, 'outside.mjs'), path.join(plugins, 'link.mjs'));
    const skill = skillWith(plugins);
    const answer = await skill.pluginLoad({ path: path.join(plugins, 'link.mjs'), name: 'link' }, {} as never);
    expect(answer).toMatch(/is not a file inside a plugin directory/);
  });

  it('refuses a path that does not exist, and the directory itself', async () => {
    const base = root();
    const plugins = path.join(base, 'plugins');
    mkdirSync(plugins);
    const skill = skillWith(plugins);
    expect(await skill.pluginLoad({ path: path.join(plugins, 'missing.mjs'), name: 'missing' }, {} as never)).toMatch(
      /is not a file inside a plugin directory/,
    );
    expect(await skill.pluginLoad({ path: plugins, name: 'dir' }, {} as never)).toMatch(/is not a file inside a plugin directory/);
  });

  it('refuses a name that is not letters, digits, - and _', async () => {
    const base = root();
    const plugins = path.join(base, 'plugins');
    mkdirSync(plugins);
    writeFileSync(path.join(plugins, 'good.mjs'), MODULE);
    const skill = skillWith(plugins);
    expect(await skill.pluginLoad({ path: path.join(plugins, 'good.mjs'), name: '../x' }, {} as never)).toBe(
      'Failed to load ../x: a plugin name is letters, digits, - and _ (at most 64)',
    );
  });

  it('the scan loads only what lies inside the directories', async () => {
    const base = root();
    const plugins = path.join(base, 'plugins');
    mkdirSync(plugins);
    writeFileSync(path.join(plugins, 'one.mjs'), MODULE);
    writeFileSync(path.join(base, 'outside.mjs'), MODULE);
    symlinkSync(path.join(base, 'outside.mjs'), path.join(plugins, 'two.mjs'));
    const skill = skillWith(plugins);
    expect(await skill.pluginLoad({}, {} as never)).toBe('Loaded 1 plugins: one');
  });
});
