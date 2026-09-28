/**
 * Plugin Skill
 *
 * Dynamic skill loading at runtime. Discovers and loads skill
 * modules from the filesystem or npm packages. Enables agents
 * to extend their capabilities by adding skills without restart.
 *
 * OWNER-ONLY, AND ONLY FROM THE PLUGIN DIRECTORIES (2026-09-26, S-249
 * addendum 2). `plugin_load` declared no scopes and `import()`ed whatever
 * path it was given, so anyone who could message an agent built with this
 * skill could run any module on the disk in the agent's own process, and,
 * with a writable file tool beside it, a module they wrote first. The three
 * tools are `audience: 'owner'` now, and a module loads only when its REAL
 * path (`fs.realpath`, so neither `..` nor a symlink planted inside a plugin
 * folder escapes it) lies inside one of the configured `pluginDirs`. A path
 * that does not exist, or lies elsewhere, is refused with the reason and
 * nothing is imported.
 */

import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context, ISkill } from '../../core/types';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

/** A plugin name is used in messages and export lookups: letters, digits, `-` and `_` only. */
const PLUGIN_NAME = /^[A-Za-z0-9_-]{1,64}$/;

export interface PluginConfig {
  name?: string;
  enabled?: boolean;
  /** Directories to scan for plugins */
  pluginDirs?: string[];
  /** Auto-load plugins on initialization */
  autoLoad?: boolean;
}

interface LoadedPlugin {
  name: string;
  path: string;
  skill: ISkill;
  loadedAt: number;
}

export class PluginSkill extends Skill {
  private pluginDirs: string[];
  private autoLoad: boolean;
  private plugins = new Map<string, LoadedPlugin>();
  private onPluginLoaded?: (skill: ISkill) => void;

  constructor(config: PluginConfig = {}) {
    super({ ...config, name: config.name || 'plugin' });
    this.pluginDirs = config.pluginDirs ?? [
      path.join(process.cwd(), 'plugins'),
      path.join(process.cwd(), '.webagents', 'plugins'),
    ];
    this.autoLoad = config.autoLoad ?? false;
  }

  /**
   * Set a callback for when a plugin is loaded.
   * The agent runtime uses this to register the skill.
   */
  setPluginLoadedCallback(cb: (skill: ISkill) => void): void {
    this.onPluginLoaded = cb;
  }

  override async initialize(): Promise<void> {
    await super.initialize();
    if (this.autoLoad) {
      await this.scanAndLoad();
    }
  }

  private async scanAndLoad(): Promise<string[]> {
    const loaded: string[] = [];
    for (const dir of this.pluginDirs) {
      try {
        const entries = await fs.readdir(dir, { withFileTypes: true });
        for (const entry of entries) {
          if (entry.isDirectory()) {
            const indexPath = path.join(dir, entry.name, 'index.js');
            try {
              await fs.access(indexPath);
              const refusal = await this.loadPlugin(indexPath, entry.name);
              if (refusal === null) loaded.push(entry.name);
            } catch {
              // No index.js, try index.ts via tsx or similar
            }
          } else if (entry.isFile() && (entry.name.endsWith('.js') || entry.name.endsWith('.mjs'))) {
            const pluginName = path.basename(entry.name, path.extname(entry.name));
            const refusal = await this.loadPlugin(path.join(dir, entry.name), pluginName);
            if (refusal === null) loaded.push(pluginName);
          }
        }
      } catch {
        // Dir doesn't exist
      }
    }
    return loaded;
  }

  /**
   * The real path of `candidate` when it lies inside one of the configured
   * plugin directories (each resolved the same way), else null. Resolving
   * both sides through `fs.realpath` is what makes `..` and symlinks
   * powerless: a link inside `plugins/` that points at `/etc` resolves to
   * `/etc`, outside every directory. A candidate that does not exist
   * resolves to nothing, and a configured directory that does not exist
   * holds no plugins.
   */
  async insidePluginDirs(candidate: string): Promise<string | null> {
    let real: string;
    try {
      real = await fs.realpath(path.resolve(candidate));
    } catch {
      return null;
    }
    for (const dir of this.pluginDirs) {
      let realDir: string;
      try {
        realDir = await fs.realpath(path.resolve(dir));
      } catch {
        continue;
      }
      const rel = path.relative(realDir, real);
      if (rel && rel !== '..' && !rel.startsWith(`..${path.sep}`) && !path.isAbsolute(rel)) return real;
    }
    return null;
  }

  /** Loads the module; null when it did, else why it did not (nothing was imported). */
  private async loadPlugin(modulePath: string, name: string): Promise<string | null> {
    if (this.plugins.has(name)) return `plugin ${name} is already loaded`;
    if (!PLUGIN_NAME.test(name)) return 'a plugin name is letters, digits, - and _ (at most 64)';

    const absPath = await this.insidePluginDirs(modulePath);
    if (absPath === null) {
      const why = `${modulePath} is not a file inside a plugin directory (${this.pluginDirs.join(', ')})`;
      console.warn(`[plugin] ${name}: refused, ${why}`);
      return why;
    }

    try {
      const mod = await import(/* @vite-ignore */ absPath);

      // Look for a default export that extends Skill, or a 'skill' export
      const SkillClass = mod.default ?? mod.skill ?? mod[`${name}Skill`] ?? mod[`${name.charAt(0).toUpperCase() + name.slice(1)}Skill`];

      if (!SkillClass || typeof SkillClass !== 'function') {
        console.warn(`[plugin] ${name}: no skill class found in ${modulePath}`);
        return `no skill class found in ${modulePath}`;
      }

      const instance: ISkill = new SkillClass();
      await instance.initialize?.();

      this.plugins.set(name, {
        name,
        path: absPath,
        skill: instance,
        loadedAt: Date.now(),
      });

      this.onPluginLoaded?.(instance);
      return null;
    } catch (err) {
      console.error(`[plugin] Failed to load ${name}:`, (err as Error).message);
      return (err as Error).message;
    }
  }

  @tool({
    audience: 'owner',
    name: 'plugin_list',
    description: 'List all loaded plugins and available plugin directories.',
    parameters: { type: 'object', properties: {} },
  })
  async pluginList(
    _params: Record<string, unknown>,
    _context: Context,
  ): Promise<{ loaded: Array<{ name: string; path: string; loadedAt: string }>; dirs: string[] }> {
    const loaded = [...this.plugins.values()].map((p) => ({
      name: p.name,
      path: p.path,
      loadedAt: new Date(p.loadedAt).toISOString(),
    }));
    return { loaded, dirs: this.pluginDirs };
  }

  @tool({
    audience: 'owner',
    name: 'plugin_load',
    description: 'Load a plugin from a file inside a plugin directory, or scan the plugin directories.',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Path to the plugin module, inside a plugin directory (optional: scans the directories if omitted)' },
        name: { type: 'string', description: 'Plugin name (required if path is given)' },
      },
    },
  })
  async pluginLoad(
    params: { path?: string; name?: string },
    _context: Context,
  ): Promise<string> {
    if (params.path && params.name) {
      const refusal = await this.loadPlugin(String(params.path), String(params.name));
      return refusal === null ? `Plugin ${params.name} loaded` : `Failed to load ${params.name}: ${refusal}`;
    }
    const loaded = await this.scanAndLoad();
    return loaded.length > 0
      ? `Loaded ${loaded.length} plugins: ${loaded.join(', ')}`
      : 'No new plugins found';
  }

  @tool({
    audience: 'owner',
    name: 'plugin_unload',
    description: 'Unload a plugin by name.',
    parameters: {
      type: 'object',
      properties: {
        name: { type: 'string', description: 'Plugin name to unload' },
      },
      required: ['name'],
    },
  })
  async pluginUnload(params: { name: string }, _context: Context): Promise<string> {
    const plugin = this.plugins.get(params.name);
    if (!plugin) return `Plugin ${params.name} not found`;
    await plugin.skill.cleanup?.();
    this.plugins.delete(params.name);
    return `Plugin ${params.name} unloaded`;
  }

  override async cleanup(): Promise<void> {
    for (const plugin of this.plugins.values()) {
      await plugin.skill.cleanup?.();
    }
    this.plugins.clear();
  }
}
