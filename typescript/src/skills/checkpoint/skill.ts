/**
 * Checkpoint Skill
 *
 * Creates file system snapshots of agent working directories.
 * Useful for versioning agent work, rolling back mistakes,
 * and auditing agent file modifications.
 */

import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context } from '../../core/types';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as crypto from 'node:crypto';

export interface CheckpointConfig {
  name?: string;
  enabled?: boolean;
  /** Base directory for checkpoints (default: .webagents/checkpoints) */
  checkpointDir?: string;
  /** Working directory to snapshot (default: cwd) */
  workDir?: string;
  /** Max checkpoints to retain (default: 50) */
  maxCheckpoints?: number;
  /** Patterns to exclude from snapshots */
  excludePatterns?: string[];
}

interface CheckpointMeta {
  id: string;
  label: string;
  createdAt: string;
  fileCount: number;
  totalSize: number;
}

export class CheckpointSkill extends Skill {
  private checkpointDir: string;
  private workDir: string;
  private maxCheckpoints: number;
  private excludePatterns: string[];

  constructor(config: CheckpointConfig = {}) {
    super({ ...config, name: config.name || 'checkpoint' });
    this.workDir = config.workDir ?? process.cwd();
    this.checkpointDir = config.checkpointDir
      ?? path.join(this.workDir, '.webagents', 'checkpoints');
    this.maxCheckpoints = config.maxCheckpoints ?? 50;
    this.excludePatterns = config.excludePatterns ?? [
      'node_modules', '.git', '.webagents/checkpoints', '__pycache__',
      '.venv', 'dist', 'build', '.next',
    ];
  }

  private shouldExclude(relativePath: string): boolean {
    return this.excludePatterns.some((p) =>
      relativePath.startsWith(p) || relativePath.includes(`/${p}/`),
    );
  }

  private async collectFiles(dir: string, base: string): Promise<Array<{ rel: string; abs: string; size: number }>> {
    const results: Array<{ rel: string; abs: string; size: number }> = [];
    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      for (const entry of entries) {
        const rel = path.relative(base, path.join(dir, entry.name));
        if (this.shouldExclude(rel)) continue;
        const abs = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          results.push(...await this.collectFiles(abs, base));
        } else if (entry.isFile()) {
          const stat = await fs.stat(abs);
          results.push({ rel, abs, size: stat.size });
        }
      }
    } catch {
      // Permission errors etc — skip
    }
    return results;
  }

  @tool({
    name: 'checkpoint_create',
    description: 'Create a snapshot of the current working directory.',
    parameters: {
      type: 'object',
      properties: {
        label: { type: 'string', description: 'Human-readable label for this checkpoint' },
      },
      required: ['label'],
    },
  })
  async checkpointCreate(
    params: { label: string },
    _context: Context,
  ): Promise<CheckpointMeta> {
    const id = `cp_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
    const cpDir = path.join(this.checkpointDir, id);
    await fs.mkdir(cpDir, { recursive: true });

    const files = await this.collectFiles(this.workDir, this.workDir);
    let totalSize = 0;

    for (const file of files) {
      const dest = path.join(cpDir, file.rel);
      await fs.mkdir(path.dirname(dest), { recursive: true });
      await fs.copyFile(file.abs, dest);
      totalSize += file.size;
    }

    const meta: CheckpointMeta = {
      id,
      label: params.label,
      createdAt: new Date().toISOString(),
      fileCount: files.length,
      totalSize,
    };

    await fs.writeFile(
      path.join(cpDir, '_checkpoint.json'),
      JSON.stringify(meta, null, 2),
    );

    await this.pruneOldCheckpoints();
    return meta;
  }

  @tool({
    name: 'checkpoint_list',
    description: 'List all available checkpoints.',
    parameters: { type: 'object', properties: {} },
  })
  async checkpointList(_params: Record<string, unknown>, _context: Context): Promise<CheckpointMeta[]> {
    try {
      const entries = await fs.readdir(this.checkpointDir, { withFileTypes: true });
      const metas: CheckpointMeta[] = [];
      for (const entry of entries) {
        if (!entry.isDirectory()) continue;
        try {
          const raw = await fs.readFile(
            path.join(this.checkpointDir, entry.name, '_checkpoint.json'),
            'utf-8',
          );
          metas.push(JSON.parse(raw));
        } catch {
          continue;
        }
      }
      return metas.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
    } catch {
      return [];
    }
  }

  @tool({
    name: 'checkpoint_restore',
    description: 'Restore files from a checkpoint to the working directory.',
    parameters: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Checkpoint ID to restore' },
      },
      required: ['id'],
    },
  })
  async checkpointRestore(
    params: { id: string },
    _context: Context,
  ): Promise<string> {
    const cpDir = path.join(this.checkpointDir, params.id);
    try {
      await fs.access(cpDir);
    } catch {
      return `Checkpoint ${params.id} not found`;
    }

    const files = await this.collectFiles(cpDir, cpDir);
    let restored = 0;
    for (const file of files) {
      if (file.rel === '_checkpoint.json') continue;
      const dest = path.join(this.workDir, file.rel);
      await fs.mkdir(path.dirname(dest), { recursive: true });
      await fs.copyFile(file.abs, dest);
      restored++;
    }
    return `Restored ${restored} files from checkpoint ${params.id}`;
  }

  @tool({
    name: 'checkpoint_diff',
    description: 'Show files changed since a checkpoint.',
    parameters: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Checkpoint ID to compare against' },
      },
      required: ['id'],
    },
  })
  async checkpointDiff(
    params: { id: string },
    _context: Context,
  ): Promise<{ added: string[]; modified: string[]; deleted: string[] }> {
    const cpDir = path.join(this.checkpointDir, params.id);
    const cpFiles = await this.collectFiles(cpDir, cpDir);
    const currentFiles = await this.collectFiles(this.workDir, this.workDir);

    const cpMap = new Map(cpFiles.filter((f) => f.rel !== '_checkpoint.json').map((f) => [f.rel, f]));
    const curMap = new Map(currentFiles.map((f) => [f.rel, f]));

    const added: string[] = [];
    const modified: string[] = [];
    const deleted: string[] = [];

    for (const [rel] of curMap) {
      if (!cpMap.has(rel)) {
        added.push(rel);
      } else if (cpMap.get(rel)!.size !== curMap.get(rel)!.size) {
        modified.push(rel);
      }
    }
    for (const [rel] of cpMap) {
      if (!curMap.has(rel)) deleted.push(rel);
    }

    return { added, modified, deleted };
  }

  private async pruneOldCheckpoints(): Promise<void> {
    const all = await this.checkpointList({}, {} as Context);
    if (all.length <= this.maxCheckpoints) return;
    const toDelete = all.slice(this.maxCheckpoints);
    for (const cp of toDelete) {
      await fs.rm(path.join(this.checkpointDir, cp.id), { recursive: true, force: true });
    }
  }
}
