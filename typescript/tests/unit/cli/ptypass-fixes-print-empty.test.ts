/**
 * `-p` says so when the model returns nothing (the ptypass-fixes lane,
 * 2026-09-27, brief item 14; a chat-fixes leftover). The chat printed the
 * truthful line (`presentEmptyReply`), while `-p` printed an empty line and
 * exited 0, so a script could not tell an empty reply from an answer. The
 * line goes to stderr; stdout keeps the answer channel's shape. The Python
 * twin is `python/tests/cli/test_ptypass_fixes_print_empty.py`.
 *
 * Driven for real: the CLI in a child process (this repo's tsx) against a
 * local stand-in for the OpenAI API that returns a completion with no
 * content.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import * as path from 'node:path';

import { presentEmptyReply } from '../../../src/cli/failures';
import { CLI_ARGS, TSX_PROBLEM, tempDirs } from '../../helpers/cli';

const tempDir = tempDirs();
const withTsx = TSX_PROBLEM === null ? it : it.skip;
let server: http.Server;
let port = 0;

beforeAll(async () => {
  server = http.createServer((req, res) => {
    let raw = '';
    req.on('data', (chunk) => (raw += chunk));
    req.on('end', () => {
      const request = raw ? (JSON.parse(raw) as { stream?: boolean; model?: string }) : {};
      const base = { id: 'empty-1', object: 'chat.completion.chunk', created: 0, model: request.model ?? 'gpt-4o-mini' };
      if (request.stream) {
        res.writeHead(200, { 'Content-Type': 'text/event-stream' });
        res.write(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] })}\n\n`);
        res.write(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`);
        res.end('data: [DONE]\n\n');
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ...base, object: 'chat.completion', choices: [{ index: 0, message: { role: 'assistant', content: '' }, finish_reason: 'stop' }] }));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  port = (server.address() as AddressInfo).port;
});

afterAll(() => new Promise<void>((resolve) => server.close(() => resolve())));

describe('-p and an empty reply', () => {
  withTsx('prints the truthful line on stderr and nothing on stdout', async () => {
    const project = tempDir('wa-ptf-empty-project-');
    fs.writeFileSync(path.join(project, 'AGENT.md'), '---\nname: quiet\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nSay nothing.\n');
    const env: Record<string, string> = {};
    for (const [key, value] of Object.entries(process.env)) {
      if (value !== undefined && !key.endsWith('_API_KEY') && !['WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'WEBAGENTS_DEBUG'].includes(key)) env[key] = value;
    }
    Object.assign(env, {
      HOME: tempDir('wa-ptf-empty-home-'),
      WEBAGENTS_SECRETS_BACKEND: 'file',
      ROBUTLER_API_URL: 'http://127.0.0.1:9',
      OPENAI_API_KEY: 'sk-test-not-a-real-key',
      OPENAI_BASE_URL: `http://127.0.0.1:${port}/v1`,
    });
    // Not spawnSync: the stand-in model answers from this process's event loop.
    const done = await new Promise<{ status: number | null; stdout: string; stderr: string }>((resolve) => {
      const child = spawn(process.execPath, [...CLI_ARGS, '-p', 'say nothing'], { cwd: project, env });
      let stdout = '';
      let stderr = '';
      child.stdout.on('data', (chunk) => (stdout += chunk));
      child.stderr.on('data', (chunk) => (stderr += chunk));
      const timer = setTimeout(() => child.kill('SIGTERM'), 180_000);
      child.on('close', (status) => {
        clearTimeout(timer);
        resolve({ status, stdout, stderr });
      });
    });
    const expected = presentEmptyReply({});
    expect(done.status, done.stderr).toBe(0);
    expect(done.stdout.trim()).toBe('');
    expect(done.stderr).toContain(expected.headline);
    if (expected.hint) expect(done.stderr).toContain(expected.hint);
  }, 200_000);
});
