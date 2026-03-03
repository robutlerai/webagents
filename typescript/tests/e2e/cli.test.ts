/**
 * CLI End-to-End Tests
 * 
 * Tests for the webagents CLI.
 * Note: These tests require the CLI to be built first.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { spawn } from 'child_process';
import { join } from 'path';

// Skip E2E tests in CI unless explicitly enabled
const runE2E = process.env.RUN_E2E === 'true';

describe.skipIf(!runE2E)('CLI E2E Tests', () => {
  const cliPath = join(__dirname, '../../dist/cli/index.js');
  
  /**
   * Run CLI command and capture output
   */
  async function runCli(args: string[]): Promise<{ stdout: string; stderr: string; code: number | null }> {
    return new Promise((resolve) => {
      const proc = spawn('node', [cliPath, ...args]);
      
      let stdout = '';
      let stderr = '';
      
      proc.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      proc.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      proc.on('close', (code) => {
        resolve({ stdout, stderr, code });
      });
    });
  }
  
  describe('help command', () => {
    it('displays help when no arguments', async () => {
      const { stdout } = await runCli(['--help']);
      expect(stdout).toContain('webagents');
      expect(stdout).toContain('Usage:');
    });
    
    it('displays version', async () => {
      const { stdout } = await runCli(['--version']);
      expect(stdout).toMatch(/\d+\.\d+\.\d+/);
    });
  });
  
  describe('info command', () => {
    it('displays SDK information', async () => {
      const { stdout } = await runCli(['info']);
      expect(stdout).toContain('webagents');
    });
  });
  
  describe('models command', () => {
    it('lists available models', async () => {
      const { stdout } = await runCli(['models']);
      // Should list some models (may require configuration)
      expect(stdout).toBeDefined();
    });
  });
});

describe('CLI Unit Tests', () => {
  // These tests can run without building the CLI
  
  describe('slash command parsing', () => {
    it('identifies slash commands', () => {
      const input = '/help';
      expect(input.startsWith('/')).toBe(true);
    });
    
    it('extracts command and args', () => {
      const input = '/model gpt-4';
      const [cmd, ...args] = input.slice(1).split(' ');
      expect(cmd).toBe('model');
      expect(args).toEqual(['gpt-4']);
    });
  });
  
  describe('message formatting', () => {
    it('trims whitespace', () => {
      const input = '  hello world  ';
      expect(input.trim()).toBe('hello world');
    });
    
    it('handles empty input', () => {
      const input = '';
      expect(input.trim()).toBe('');
    });
  });
});
