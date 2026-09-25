/**
 * `publish` updates the agent it created instead of minting another (2026-09-24).
 *
 * Every run POSTed a new agent, so publishing the same directory twice created
 * `<name>-2`, a second permanent handle (platform usernames cannot be renamed),
 * where the person meant an update. Measured against a local cluster. The link
 * now lives where the Python CLI keeps it, `link.agentId` in the directory's
 * `.webagents/config.json`, and a create is confirmed first, like `deploy`.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as http from 'node:http';
import * as path from 'node:path';
import { spawn } from 'node:child_process';
import { tempDirs, TSX_CLI } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

const CLI_SOURCE = path.resolve(__dirname, '../../../src/cli/index.ts');

interface Seen { method: string; url: string }

async function publish(project: string, home: string, extra: string[], seen: Seen[]) {
  const server = http.createServer((req, res) => {
    seen.push({ method: req.method ?? '', url: req.url ?? '' });
    req.resume();
    req.on('end', () => {
      res.setHeader('content-type', 'application/json');
      if (req.method === 'PATCH') {
        res.statusCode = 200;
        res.end(JSON.stringify({ agent: { id: 'a-1', username: 'owner.probe' } }));
      } else {
        res.statusCode = 201;
        res.end(JSON.stringify({ agent: { id: 'a-1', username: 'owner.probe' }, rawApiKey: 'rok_DUMMY' }));
      }
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as { port: number }).port;
  try {
    return await new Promise<{ code: number | null; stdout: string; stderr: string }>((resolve) => {
      const child = spawn(process.execPath, [TSX_CLI, CLI_SOURCE, 'publish', project, ...extra], {
        env: {
          ...process.env,
          HOME: home,
          WEBAGENTS_TOKEN: 'stub-token',
          ROBUTLER_API_URL: `http://127.0.0.1:${port}`,
          WEBAGENTS_PROFILE: '',
          WEBAGENTS_SECRETS_BACKEND: 'file',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stdout = '';
      let stderr = '';
      child.stdout.on('data', (d) => (stdout += d));
      child.stderr.on('data', (d) => (stderr += d));
      child.on('close', (code) => resolve({ code, stdout, stderr }));
    });
  } finally {
    server.close();
  }
}

function scaffold(): { project: string; home: string } {
  const home = tempDir('wa-link-home-');
  const project = tempDir('wa-link-proj-');
  fs.writeFileSync(
    path.join(project, 'AGENT.md'),
    '---\nname: probe\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\n\nYou help.\n',
  );
  return { project, home };
}

describe('publish and the directory link', () => {
  it('creates once, links, then updates the same agent', async () => {
    const { project, home } = scaffold();
    const seen: Seen[] = [];

    const first = await publish(project, home, ['--yes'], seen);
    expect(first.code).toBe(0);
    expect(seen).toEqual([{ method: 'POST', url: '/api/agents' }]);
    const link = JSON.parse(fs.readFileSync(path.join(project, '.webagents', 'config.json'), 'utf-8'));
    // The Python CLI's keys, so `webagents deploy` updates the same agent.
    expect(link).toEqual({ 'link.agentId': 'a-1', 'link.agentName': 'owner.probe' });

    const second = await publish(project, home, [], seen);
    expect(second.code).toBe(0);
    // THE BUG: this was a second POST, and the platform minted `probe-2`.
    expect(seen[1]).toEqual({ method: 'PATCH', url: '/api/agents/a-1' });
    expect(second.stdout).toContain('Updated owner.probe');
  }, 120_000);

  it('does not create without --yes when nobody can answer the prompt', async () => {
    const { project, home } = scaffold();
    const seen: Seen[] = [];
    const out = await publish(project, home, [], seen);
    expect(out.code).toBe(1);
    expect(out.stderr).toContain('--yes');
    expect(seen).toEqual([]);
  }, 60_000);
});
