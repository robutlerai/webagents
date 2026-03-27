import { exec } from 'child_process';
import * as path from 'path';
import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context } from '../../core/types';

export interface ShellSkillConfig {
  baseDir?: string;
  allowedCommands?: string[];
  blockedCommands?: string[];
  sandboxEnabled?: boolean;
}

const DEFAULT_ALLOWED = [
  'ls', 'cat', 'grep', 'find', 'head', 'tail', 'wc', 'echo', 'date',
  'pwd', 'which', 'whereis', 'git', 'npm', 'pip', 'python', 'node',
  'uvx', 'curl', 'wget', 'rg', 'fd',
];

const DEFAULT_BLOCKED = [
  'rm', 'rmdir', 'dd', 'mkfs', 'fdisk', 'kill', 'killall', 'pkill',
  'shutdown', 'reboot', 'halt', 'su', 'sudo', 'chmod', 'chown',
];

const CHAIN_OPERATORS = new Set(['&&', '||', '|', ';']);

export class ShellSkill extends Skill {
  private workingDir: string;
  private sandboxEnabled: boolean;
  private allowedCommands: Set<string>;
  private blockedCommands: Set<string>;

  constructor(config: ShellSkillConfig = {}) {
    super({ name: 'ShellSkill' });
    this.workingDir = path.resolve(config.baseDir || process.cwd());
    this.sandboxEnabled = config.sandboxEnabled ?? !!config.baseDir;
    this.allowedCommands = new Set([...DEFAULT_ALLOWED, ...(config.allowedCommands || [])]);
    this.blockedCommands = new Set([...DEFAULT_BLOCKED, ...(config.blockedCommands || [])]);
  }

  // --------------------------------------------------------------------------
  // Tokenizer
  // --------------------------------------------------------------------------

  private _tokenize(command: string): string[] {
    const tokens: string[] = [];
    let current = '';
    let inSingle = false;
    let inDouble = false;

    for (let i = 0; i < command.length; i++) {
      const ch = command[i];

      if (ch === "'" && !inDouble) {
        inSingle = !inSingle;
        continue;
      }
      if (ch === '"' && !inSingle) {
        inDouble = !inDouble;
        continue;
      }

      if (!inSingle && !inDouble && (ch === ' ' || ch === '\t')) {
        if (current.length > 0) {
          tokens.push(current);
          current = '';
        }
        continue;
      }

      current += ch;
    }
    if (current.length > 0) tokens.push(current);
    return tokens;
  }

  // --------------------------------------------------------------------------
  // Command checking
  // --------------------------------------------------------------------------

  private _checkCommand(command: string): { allowed: boolean; reason: string } {
    const tokens = this._tokenize(command);
    if (tokens.length === 0) return { allowed: false, reason: 'Empty command' };

    const commandPositions = this._extractCommandPositions(tokens);

    for (const cmd of commandPositions) {
      const base = path.basename(cmd);

      if (this.blockedCommands.has(base)) {
        return { allowed: false, reason: `Command '${base}' is blocked` };
      }
      if (!this.allowedCommands.has(base)) {
        return { allowed: false, reason: `Command '${base}' is not in the allowlist` };
      }
    }

    if (this.sandboxEnabled) {
      const sandboxCheck = this._checkSandbox(tokens);
      if (!sandboxCheck.allowed) return sandboxCheck;
    }

    return { allowed: true, reason: '' };
  }

  private _extractCommandPositions(tokens: string[]): string[] {
    const commands: string[] = [];
    let expectCommand = true;

    for (const token of tokens) {
      if (expectCommand) {
        commands.push(token);
        expectCommand = false;
      }
      if (CHAIN_OPERATORS.has(token)) {
        expectCommand = true;
      }
    }

    return commands;
  }

  private _checkSandbox(tokens: string[]): { allowed: boolean; reason: string } {
    for (const token of tokens) {
      if (CHAIN_OPERATORS.has(token)) continue;

      if (token.includes('..')) {
        return { allowed: false, reason: `Path traversal ('..') is not allowed in sandbox mode` };
      }

      if (token.startsWith('~')) {
        return { allowed: false, reason: `Home directory expansion ('~') is not allowed in sandbox mode` };
      }

      if (path.isAbsolute(token)) {
        const resolved = path.resolve(token);
        if (!resolved.startsWith(this.workingDir)) {
          return {
            allowed: false,
            reason: `Absolute path '${token}' is outside the sandbox directory`,
          };
        }
      }
    }

    return { allowed: true, reason: '' };
  }

  // --------------------------------------------------------------------------
  // Tool
  // --------------------------------------------------------------------------

  @tool({
    description: 'Run a shell command (sandboxed)',
    parameters: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'Shell command to execute' },
        timeout: { type: 'number', description: 'Timeout in seconds (default: 30)' },
      },
      required: ['command'],
    },
  })
  async runCommand(
    params: { command: string; timeout?: number },
    _context: Context,
  ): Promise<string> {
    const { command, timeout = 30 } = params;

    const check = this._checkCommand(command);
    if (!check.allowed) return `Access denied: ${check.reason}`;

    return new Promise((resolve) => {
      exec(
        command,
        {
          cwd: this.workingDir,
          timeout: timeout * 1000,
          maxBuffer: 1024 * 1024,
        },
        (error, stdout, stderr) => {
          let output = stdout || '';
          if (stderr) output += `\nStderr: ${stderr}`;
          if (error) {
            if (error.killed) {
              resolve(`Command timed out after ${timeout}s`);
              return;
            }
            output += `\nExit code: ${error.code ?? 1}`;
          }
          resolve(output || '(No output)');
        },
      );
    });
  }
}
