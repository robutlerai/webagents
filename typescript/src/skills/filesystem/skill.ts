/**
 * Filesystem Skill
 *
 * Local file operations with whitelist/blacklist sandboxing.
 * TypeScript port of the Python FilesystemSkill.
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { existsSync, readFileSync, statSync, readdirSync } from 'fs';
import { execSync } from 'child_process';
import { homedir } from 'os';
import { Skill } from '../../core/skill.js';
import { tool } from '../../core/decorators.js';
import type { Context } from '../../core/types.js';

export interface FilesystemSkillConfig {
  baseDir?: string;
  whitelist?: string[];
  blacklist?: string[];
}

export class FilesystemSkill extends Skill {
  private baseDir: string;
  private whitelist: Set<string>;
  private blacklist: Set<string>;

  constructor(config: FilesystemSkillConfig = {}) {
    super({ name: 'FilesystemSkill' });
    this.baseDir = path.resolve(config.baseDir || process.cwd());
    this.whitelist = new Set([
      this.baseDir,
      ...(config.whitelist || []).map(p => path.resolve(p)),
    ]);
    this.blacklist = new Set([
      path.join(homedir(), '.ssh'),
      path.join(homedir(), '.aws'),
      path.join(homedir(), '.config', 'gcloud'),
      path.join(homedir(), '.gnupg'),
      ...(config.blacklist || []).map(p => path.resolve(p)),
    ]);
  }

  // ===========================================================================
  // Access Control
  // ===========================================================================

  private _checkAccess(filePath: string): boolean {
    let resolved: string;
    try {
      resolved = path.resolve(filePath);
    } catch {
      return false;
    }

    for (const blocked of this.blacklist) {
      if (resolved === blocked || resolved.startsWith(blocked + path.sep)) {
        return false;
      }
    }

    for (const allowed of this.whitelist) {
      if (resolved === allowed || resolved.startsWith(allowed + path.sep)) {
        return true;
      }
    }

    return false;
  }

  private _resolvePath(pathStr: string, baseDir?: string): string {
    const expandedPath = pathStr.startsWith('~')
      ? path.join(homedir(), pathStr.slice(1))
      : pathStr;

    if (path.isAbsolute(expandedPath)) {
      return path.resolve(expandedPath);
    }
    return path.resolve(baseDir || this.baseDir, expandedPath);
  }

  private _isBinary(filePath: string): boolean {
    try {
      const fd = readFileSync(filePath, { flag: 'r' });
      const chunk = fd.subarray(0, 1024);
      return chunk.includes(0);
    } catch {
      return false;
    }
  }

  private _isGitIgnored(filePath: string): boolean {
    try {
      const cwd = existsSync(filePath) && statSync(filePath).isDirectory()
        ? filePath
        : path.dirname(filePath);

      execSync('git rev-parse --is-inside-work-tree', {
        cwd,
        stdio: 'ignore',
      });

      execSync(`git check-ignore ${JSON.stringify(filePath)}`, {
        cwd,
        stdio: ['ignore', 'pipe', 'ignore'],
      });
      return true;
    } catch {
      return false;
    }
  }

  // ===========================================================================
  // Tools
  // ===========================================================================

  @tool({
    name: 'list_directory',
    description: 'Lists files and subdirectories in a directory.',
    parameters: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'The absolute path to the directory to list.',
        },
      },
      required: ['path'],
    },
  })
  async listDirectory(
    params: { path: string },
    _context: Context,
  ): Promise<string> {
    const dirPath = this._resolvePath(params.path);

    if (!this._checkAccess(dirPath)) {
      return `Access denied: ${params.path} is outside allowed directories`;
    }

    if (!existsSync(dirPath)) {
      return `Directory not found: ${params.path}`;
    }

    let stat;
    try {
      stat = statSync(dirPath);
    } catch {
      return `Cannot stat: ${params.path}`;
    }
    if (!stat.isDirectory()) {
      return `Not a directory: ${params.path}`;
    }

    try {
      const items = readdirSync(dirPath, { withFileTypes: true });
      const entries: Array<{ isDir: boolean; display: string }> = [];

      for (const item of items) {
        const fullPath = path.join(dirPath, item.name);

        if (this._isGitIgnored(fullPath)) {
          continue;
        }

        const isDir = item.isDirectory();
        const prefix = isDir ? '[DIR] ' : '';
        entries.push({ isDir, display: `${prefix}${item.name}` });
      }

      entries.sort((a, b) => {
        if (a.isDir !== b.isDir) return a.isDir ? -1 : 1;
        return a.display.localeCompare(b.display);
      });

      const lines = [`Directory listing for ${dirPath}:`];
      for (const entry of entries) {
        lines.push(entry.display);
      }

      return lines.join('\n');
    } catch (err) {
      return `Error listing directory: ${err}`;
    }
  }

  @tool({
    name: 'read_file',
    description: 'Reads and returns the content of a specified file.',
    parameters: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'The absolute path to the file to read.',
        },
        offset: {
          type: 'integer',
          description: 'Start line number (0-based).',
        },
        limit: {
          type: 'integer',
          description: 'Maximum number of lines to read.',
        },
      },
      required: ['path'],
    },
  })
  async readFile(
    params: { path: string; offset?: number; limit?: number },
    _context: Context,
  ): Promise<string> {
    const filePath = this._resolvePath(params.path);

    if (!this._checkAccess(filePath)) {
      return `Access denied: ${params.path} is outside allowed directories`;
    }

    if (!existsSync(filePath)) {
      return `File not found: ${params.path}`;
    }

    let stat;
    try {
      stat = statSync(filePath);
    } catch {
      return `Cannot stat: ${params.path}`;
    }
    if (!stat.isFile()) {
      return `Not a file: ${params.path}`;
    }

    if (this._isBinary(filePath)) {
      return `Cannot display content of binary file: ${params.path}`;
    }

    try {
      const raw = await fs.readFile(filePath, 'utf-8');
      const lines = raw.split('\n');
      const totalLines = lines.length;

      const start = params.offset ?? 0;
      let end = totalLines;

      if (params.limit != null) {
        end = Math.min(start + params.limit, totalLines);
      } else if (params.offset == null && totalLines > 2000) {
        end = 2000;
      }

      const contentLines = lines.slice(start, end);
      const content = contentLines.join('\n');

      if (start > 0 || end < totalLines) {
        return `[File content truncated: showing lines ${start + 1}-${end} of ${totalLines} total lines...]\n${content}`;
      }

      return content;
    } catch (err) {
      return `Error reading file: ${err}`;
    }
  }

  @tool({
    name: 'write_file',
    description: 'Writes content to a specified file, creating parent directories as needed.',
    parameters: {
      type: 'object',
      properties: {
        file_path: {
          type: 'string',
          description: 'The absolute path to the file.',
        },
        content: {
          type: 'string',
          description: 'The content to write.',
        },
      },
      required: ['file_path', 'content'],
    },
  })
  async writeFile(
    params: { file_path: string; content: string },
    _context: Context,
  ): Promise<string> {
    const filePath = this._resolvePath(params.file_path);

    if (!this._checkAccess(filePath)) {
      return `Access denied: ${params.file_path} is outside allowed directories`;
    }

    try {
      const existed = existsSync(filePath);

      await fs.mkdir(path.dirname(filePath), { recursive: true });
      await fs.writeFile(filePath, params.content, 'utf-8');

      return existed
        ? `Successfully overwrote file: ${params.file_path}`
        : `Successfully created and wrote to new file: ${params.file_path}`;
    } catch (err) {
      return `Error writing file: ${err}`;
    }
  }

  @tool({
    name: 'glob',
    description: 'Finds files matching a glob pattern, sorted by modification time (newest first).',
    parameters: {
      type: 'object',
      properties: {
        pattern: {
          type: 'string',
          description: 'Glob pattern to match (e.g. "**/*.ts").',
        },
        path: {
          type: 'string',
          description: 'Directory to search in (default: base directory).',
        },
      },
      required: ['pattern'],
    },
  })
  async globSearch(
    params: { pattern: string; path?: string },
    _context: Context,
  ): Promise<string> {
    const searchDir = params.path
      ? this._resolvePath(params.path)
      : this.baseDir;

    if (!this._checkAccess(searchDir)) {
      return `Access denied: ${params.path} is outside allowed directories`;
    }

    if (!existsSync(searchDir)) {
      return `Directory not found: ${searchDir}`;
    }

    try {
      const matches = await this._globWalk(searchDir, params.pattern);

      const fileResults: Array<{ filePath: string; mtime: number }> = [];
      for (const filePath of matches) {
        if (this._isGitIgnored(filePath)) continue;
        try {
          const st = statSync(filePath);
          if (st.isFile()) {
            fileResults.push({ filePath, mtime: st.mtimeMs });
          }
        } catch {
          // skip unreadable entries
        }
      }

      fileResults.sort((a, b) => b.mtime - a.mtime);

      if (fileResults.length === 0) {
        return `Found 0 file(s) matching "${params.pattern}" within ${searchDir}`;
      }

      const lines = [
        `Found ${fileResults.length} file(s) matching "${params.pattern}" within ${searchDir}, sorted by modification time (newest first):`,
        ...fileResults.map(r => r.filePath),
      ];

      return lines.join('\n');
    } catch (err) {
      return `Error in glob: ${err}`;
    }
  }

  @tool({
    name: 'search_file_content',
    description: 'Searches for a regex pattern within files, returning matches with line numbers.',
    parameters: {
      type: 'object',
      properties: {
        pattern: {
          type: 'string',
          description: 'Regex pattern to search for.',
        },
        path: {
          type: 'string',
          description: 'Directory to search in (default: base directory).',
        },
        include: {
          type: 'string',
          description: 'Glob pattern to filter which files to search.',
        },
      },
      required: ['pattern'],
    },
  })
  async searchFileContent(
    params: { pattern: string; path?: string; include?: string },
    _context: Context,
  ): Promise<string> {
    const searchDir = params.path
      ? this._resolvePath(params.path)
      : this.baseDir;

    if (!this._checkAccess(searchDir)) {
      return `Access denied: ${params.path} is outside allowed directories`;
    }

    try {
      const regex = new RegExp(params.pattern);

      let filesToSearch: string[];
      if (params.include) {
        filesToSearch = await this._globWalk(searchDir, params.include);
      } else {
        filesToSearch = await this._globWalk(searchDir, '**/*');
      }

      let totalCount = 0;
      const fileMatches: string[] = [];

      for (const filePath of filesToSearch) {
        try {
          const st = statSync(filePath);
          if (!st.isFile()) continue;
        } catch {
          continue;
        }

        if (this._isBinary(filePath)) continue;
        if (filePath.includes(`${path.sep}.git${path.sep}`)) continue;

        try {
          const content = readFileSync(filePath, 'utf-8');
          const lines = content.split('\n');
          const matchesInFile: string[] = [];

          for (let i = 0; i < lines.length; i++) {
            if (regex.test(lines[i])) {
              matchesInFile.push(`L${i + 1}: ${lines[i]}`);
              totalCount++;
            }
          }

          if (matchesInFile.length > 0) {
            const relPath = path.relative(searchDir, filePath);
            fileMatches.push(`File: ${relPath}\n${matchesInFile.join('\n')}`);
          }
        } catch {
          continue;
        }
      }

      if (fileMatches.length === 0) {
        return `Found 0 matches for pattern "${params.pattern}" in ${searchDir}`;
      }

      let header = `Found ${totalCount} matches for pattern "${params.pattern}" in ${searchDir}`;
      if (params.include) {
        header += ` (filter: "${params.include}")`;
      }
      header += ':';

      return `${header}\n---\n${fileMatches.join('\n---\n')}\n---`;
    } catch (err) {
      return `Error searching content: ${err}`;
    }
  }

  @tool({
    name: 'replace',
    description: 'Replaces exact text within a file. Pass empty old_string to create a new file with the given content.',
    parameters: {
      type: 'object',
      properties: {
        file_path: {
          type: 'string',
          description: 'Absolute path to the file.',
        },
        old_string: {
          type: 'string',
          description: 'Exact text to replace. Empty to create a new file.',
        },
        new_string: {
          type: 'string',
          description: 'Replacement text.',
        },
        expected_replacements: {
          type: 'integer',
          description: 'Number of occurrences to replace (default: 1).',
        },
      },
      required: ['file_path', 'old_string', 'new_string'],
    },
  })
  async replace(
    params: {
      file_path: string;
      old_string: string;
      new_string: string;
      expected_replacements?: number;
    },
    _context: Context,
  ): Promise<string> {
    const filePath = this._resolvePath(params.file_path);
    const expectedReplacements = params.expected_replacements ?? 1;

    if (!this._checkAccess(filePath)) {
      return `Access denied: ${params.file_path} is outside allowed directories`;
    }

    // Create new file when old_string is empty
    if (!params.old_string) {
      if (existsSync(filePath)) {
        return `Failed to edit: old_string is empty but file ${params.file_path} already exists.`;
      }

      try {
        await fs.mkdir(path.dirname(filePath), { recursive: true });
        await fs.writeFile(filePath, params.new_string, 'utf-8');
        return `Created new file: ${params.file_path} with provided content.`;
      } catch (err) {
        return `Error creating file: ${err}`;
      }
    }

    // Replace in existing file
    if (!existsSync(filePath)) {
      return `Failed to edit: file ${params.file_path} does not exist.`;
    }

    try {
      const content = await fs.readFile(filePath, 'utf-8');

      let count = 0;
      let idx = -1;
      while ((idx = content.indexOf(params.old_string, idx + 1)) !== -1) {
        count++;
      }

      if (count === 0) {
        return 'Failed to edit, 0 occurrences found of old_string. Please ensure exact match including whitespace.';
      }

      if (count !== expectedReplacements) {
        if (count < expectedReplacements) {
          return `Failed to edit, expected ${expectedReplacements} occurrences but found ${count}.`;
        }
        if (count > expectedReplacements && expectedReplacements === 1) {
          return `Failed to edit, expected 1 occurrence but found ${count}. Please provide more context to disambiguate.`;
        }
      }

      let newContent = content;
      let replaced = 0;
      let searchFrom = 0;
      while (replaced < expectedReplacements) {
        const pos = newContent.indexOf(params.old_string, searchFrom);
        if (pos === -1) break;
        newContent =
          newContent.slice(0, pos) +
          params.new_string +
          newContent.slice(pos + params.old_string.length);
        searchFrom = pos + params.new_string.length;
        replaced++;
      }

      await fs.writeFile(filePath, newContent, 'utf-8');
      return `Successfully modified file: ${params.file_path} (${expectedReplacements} replacements).`;
    } catch (err) {
      return `Error replacing text: ${err}`;
    }
  }

  // ===========================================================================
  // Recursive glob helper (Node 20+ fs.readdir recursive)
  // ===========================================================================

  private async _globWalk(
    dir: string,
    pattern: string,
  ): Promise<string[]> {
    const results: string[] = [];

    const isRecursive = pattern.includes('**');

    if (isRecursive) {
      const allEntries = await fs.readdir(dir, {
        recursive: true,
        withFileTypes: false,
      }) as unknown as string[];

      for (const entry of allEntries) {
        const fullPath = path.join(dir, entry);
        if (this._matchGlob(entry, pattern)) {
          results.push(fullPath);
        }
      }
    } else {
      const allEntries = await fs.readdir(dir, {
        recursive: pattern.includes('/'),
        withFileTypes: false,
      }) as unknown as string[];

      for (const entry of allEntries) {
        const fullPath = path.join(dir, entry);
        if (this._matchGlob(entry, pattern)) {
          results.push(fullPath);
        }
      }
    }

    return results;
  }

  /**
   * Minimal glob matcher supporting *, **, and ? wildcards.
   * Operates on forward-slash-normalized relative paths.
   */
  private _matchGlob(filePath: string, pattern: string): boolean {
    const normalizedPath = filePath.replace(/\\/g, '/');
    const regexStr = pattern
      .replace(/\\/g, '/')
      .replace(/[.+^${}()|[\]]/g, '\\$&')
      .replace(/\*\*\//g, '(.+/)?')
      .replace(/\*\*/g, '.*')
      .replace(/\*/g, '[^/]*')
      .replace(/\?/g, '[^/]');

    return new RegExp(`^${regexStr}$`).test(normalizedPath);
  }
}
