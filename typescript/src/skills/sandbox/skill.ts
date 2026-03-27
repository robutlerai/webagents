/**
 * Sandbox Skill
 *
 * Executes code in isolated Docker containers. Provides a safe
 * environment for running untrusted code, data processing,
 * and computational tasks.
 *
 * Requires Docker to be available on the host.
 */

import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context } from '../../core/types';
import { execSync, spawn } from 'node:child_process';

export interface SandboxConfig {
  name?: string;
  enabled?: boolean;
  /** Docker image to use (default: python:3.12-slim) */
  defaultImage?: string;
  /** Maximum execution time in seconds (default: 30) */
  timeout?: number;
  /** Maximum memory in MB (default: 512) */
  maxMemory?: number;
  /** Maximum CPU shares (default: 1024) */
  maxCpu?: number;
  /** Network access: 'none' (default), 'bridge', 'host' */
  network?: 'none' | 'bridge' | 'host';
  /** Additional Docker run flags */
  extraFlags?: string[];
}

export class SandboxSkill extends Skill {
  private defaultImage: string;
  private timeout: number;
  private maxMemory: number;
  private maxCpu: number;
  private network: string;
  private extraFlags: string[];
  private dockerAvailable: boolean | null = null;

  constructor(config: SandboxConfig = {}) {
    super({ ...config, name: config.name || 'sandbox' });
    this.defaultImage = config.defaultImage ?? 'python:3.12-slim';
    this.timeout = config.timeout ?? 30;
    this.maxMemory = config.maxMemory ?? 512;
    this.maxCpu = config.maxCpu ?? 1024;
    this.network = config.network ?? 'none';
    this.extraFlags = config.extraFlags ?? [];
  }

  private checkDocker(): boolean {
    if (this.dockerAvailable !== null) return this.dockerAvailable;
    try {
      execSync('docker info', { stdio: 'ignore', timeout: 5000 });
      this.dockerAvailable = true;
    } catch {
      this.dockerAvailable = false;
    }
    return this.dockerAvailable;
  }

  private async runContainer(
    image: string,
    command: string[],
    stdin?: string,
  ): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    return new Promise((resolve) => {
      const args = [
        'run', '--rm',
        '--network', this.network,
        '--memory', `${this.maxMemory}m`,
        '--cpu-shares', String(this.maxCpu),
        '--pids-limit', '64',
        '--read-only',
        '--tmpfs', '/tmp:rw,noexec,nosuid,size=100m',
        '--security-opt', 'no-new-privileges',
        ...this.extraFlags,
        image,
        ...command,
      ];

      const proc = spawn('docker', args, {
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: this.timeout * 1000,
      });

      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (d) => { stdout += d.toString(); });
      proc.stderr.on('data', (d) => { stderr += d.toString(); });

      if (stdin) {
        proc.stdin.write(stdin);
        proc.stdin.end();
      }

      const timer = setTimeout(() => {
        proc.kill('SIGKILL');
        stderr += '\n[TIMEOUT] Execution exceeded time limit';
      }, this.timeout * 1000);

      proc.on('close', (code) => {
        clearTimeout(timer);
        const maxLen = 50_000;
        resolve({
          stdout: stdout.slice(0, maxLen),
          stderr: stderr.slice(0, maxLen),
          exitCode: code ?? 1,
        });
      });

      proc.on('error', (err) => {
        clearTimeout(timer);
        resolve({
          stdout: '',
          stderr: `Docker error: ${err.message}`,
          exitCode: 1,
        });
      });
    });
  }

  @tool({
    name: 'sandbox_run_python',
    description: 'Execute Python code in a sandboxed Docker container. Returns stdout, stderr, and exit code.',
    parameters: {
      type: 'object',
      properties: {
        code: { type: 'string', description: 'Python code to execute' },
        timeout: { type: 'number', description: 'Max execution time in seconds' },
      },
      required: ['code'],
    },
  })
  async sandboxRunPython(
    params: { code: string; timeout?: number },
    _context: Context,
  ): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    if (!this.checkDocker()) {
      return { stdout: '', stderr: 'Docker is not available on this system', exitCode: 1 };
    }

    const prevTimeout = this.timeout;
    if (params.timeout) this.timeout = params.timeout;

    try {
      return await this.runContainer(
        this.defaultImage,
        ['python3', '-c', params.code],
      );
    } finally {
      this.timeout = prevTimeout;
    }
  }

  @tool({
    name: 'sandbox_run_shell',
    description: 'Execute a shell command in a sandboxed Docker container.',
    parameters: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'Shell command to execute' },
        image: { type: 'string', description: 'Docker image (default: python:3.12-slim)' },
      },
      required: ['command'],
    },
  })
  async sandboxRunShell(
    params: { command: string; image?: string },
    _context: Context,
  ): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    if (!this.checkDocker()) {
      return { stdout: '', stderr: 'Docker is not available on this system', exitCode: 1 };
    }
    return this.runContainer(
      params.image ?? this.defaultImage,
      ['sh', '-c', params.command],
    );
  }

  @tool({
    name: 'sandbox_run_node',
    description: 'Execute JavaScript/TypeScript code in a sandboxed Docker container.',
    parameters: {
      type: 'object',
      properties: {
        code: { type: 'string', description: 'JavaScript code to execute' },
      },
      required: ['code'],
    },
  })
  async sandboxRunNode(
    params: { code: string },
    _context: Context,
  ): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    if (!this.checkDocker()) {
      return { stdout: '', stderr: 'Docker is not available on this system', exitCode: 1 };
    }
    return this.runContainer(
      'node:20-slim',
      ['node', '-e', params.code],
    );
  }
}
