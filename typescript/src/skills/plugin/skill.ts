/**
 * Plugin Skill
 *
 * Dynamic skill loading at runtime. Discovers and loads skill
 * modules from the filesystem or npm packages. Enables agents
 * to extend their capabilities by adding skills without restart.
 */

import { Skill } from '../../core/skill.js';
import { tool } from '../../core/decorators.js';
import type { Context, ISkill } from '../../core/types.js';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

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
              const result = await this.loadPlugin(indexPath, entry.name);
              if (result) loaded.push(entry.name);
            } catch {
              // No index.js, try index.ts via tsx or similar
            }
          } else if (entry.isFile() && (entry.name.endsWith('.js') || entry.name.endsWith('.mjs'))) {
            const pluginName = path.basename(entry.name, path.extname(entry.name));
            const result = await this.loadPlugin(path.join(dir, entry.name), pluginName);
            if (result) loaded.push(pluginName);
          }
        }
      } catch {
        // Dir doesn't exist
      }
    }
    return loaded;
  }

  private async loadPlugin(modulePath: string, name: string): Promise<boolean> {
    if (this.plugins.has(name)) return false;

    try {
      const absPath = path.resolve(modulePath);
      const mod = await import(/* @vite-ignore */ absPath);

      // Look for a default export that extends Skill, or a 'skill' export
      const SkillClass = mod.default ?? mod.skill ?? mod[`${name}Skill`] ?? mod[`${name.charAt(0).toUpperCase() + name.slice(1)}Skill`];

      if (!SkillClass || typeof SkillClass !== 'function') {
        console.warn(`[plugin] ${name}: no skill class found in ${modulePath}`);
        return false;
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
      return true;
    } catch (err) {
      console.error(`[plugin] Failed to load ${name}:`, (err as Error).message);
      return false;
    }
  }

  @tool({
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
    name: 'plugin_load',
    description: 'Load a plugin from a file path or scan plugin directories.',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Path to plugin module (optional — scans dirs if omitted)' },
        name: { type: 'string', description: 'Plugin name (required if path is given)' },
      },
    },
  })
  async pluginLoad(
    params: { path?: string; name?: string },
    _context: Context,
  ): Promise<string> {
    if (params.path && params.name) {
      const ok = await this.loadPlugin(params.path, params.name);
      return ok ? `Plugin ${params.name} loaded` : `Failed to load ${params.name}`;
    }
    const loaded = await this.scanAndLoad();
    return loaded.length > 0
      ? `Loaded ${loaded.length} plugins: ${loaded.join(', ')}`
      : 'No new plugins found';
  }

  @tool({
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
