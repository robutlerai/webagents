/**
 * The `web` skill (2026-09-25): the Python SDK's `web_fetch`, with the same
 * definition (`python/tests/fixtures/web_tool/definition.json`) and the same
 * address rules. Only public addresses are fetched, every redirect hop is
 * checked, and the connection goes to the address that was checked. The
 * Python cases are `python/tests/agents/skills/test_web_skill.py`.
 */

import { afterEach, describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { WebSkill, pageText } from '../../../src/skills/web/skill';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFINITION = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/web_tool/definition.json'), 'utf8'),
);

const PAGE =
  '<html><head><style>p{}</style><script>var secret = 1;</script></head>' +
  '<body><h1>Hello</h1><p>from the page</p></body></html>';

interface Site {
  port: number;
  hits: string[];
  close: () => Promise<void>;
}

const open: Site[] = [];

async function site(redirectTo?: () => string): Promise<Site> {
  const hits: string[] = [];
  const server = http.createServer((req, res) => {
    hits.push(req.url ?? '');
    if (req.url === '/hop' && redirectTo) {
      res.writeHead(302, { Location: redirectTo() });
      res.end();
      return;
    }
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(PAGE);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', () => resolve()));
  const s: Site = {
    port: (server.address() as AddressInfo).port,
    hits,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
  open.push(s);
  return s;
}

afterEach(async () => {
  await Promise.all(open.splice(0).map((s) => s.close()));
});

function fetchWith(skill: WebSkill, prompt: string): Promise<string> {
  return skill.webFetch({ prompt });
}

describe('the web skill', () => {
  it('offers the definition both SDKs share', () => {
    const agent = new BaseAgent({ name: 'w', instructions: 'x', skills: [new WebSkill()] });
    const def = agent.getToolDefinitions().find((d) => d.function.name === 'web_fetch');
    expect(def).toEqual(DEFINITION);
  });

  it('refuses a loopback address without connecting', async () => {
    const s = await site();
    const out = await fetchWith(new WebSkill(), `Read http://127.0.0.1:${s.port}/page`);
    expect(out).toContain('127.0.0.1 is not a public address, so it is not called.');
    expect(s.hits).toEqual([]);
  });

  it('refuses a name that resolves to loopback', async () => {
    const s = await site();
    const out = await fetchWith(new WebSkill(), `Read http://localhost:${s.port}/page`);
    expect(out).toContain('localhost resolves to');
    expect(out).toContain('which is not a public address');
    expect(s.hits).toEqual([]);
  });

  it.each(['http://169.254.169.254/latest/meta-data/', 'http://10.0.0.1/', 'http://[::1]/'])(
    'refuses %s',
    async (url) => {
      expect(await fetchWith(new WebSkill(), `Read ${url}`)).toContain('is not a public address, so it is not called.');
    },
  );

  it('fetches an allowed private address as text', async () => {
    const s = await site();
    const out = await fetchWith(new WebSkill({ allow_private: [`127.0.0.1:${s.port}`] }), `Read http://127.0.0.1:${s.port}/page`);
    expect(out).toContain('Hello from the page');
    expect(out).not.toContain('secret');
    expect(s.hits).toEqual(['/page']);
  });

  it('checks each redirect hop', async () => {
    const target = await site();
    const first = await site(() => `http://127.0.0.1:${target.port}/page`);
    // Only the first server is allowed; its redirect points elsewhere.
    const out = await fetchWith(new WebSkill({ allow_private: [`127.0.0.1:${first.port}`] }), `Read http://127.0.0.1:${first.port}/hop`);
    expect(out).toContain('is not a public address, so it is not called.');
    expect(first.hits).toEqual(['/hop']);
    expect(target.hits).toEqual([]);
  });

  it('stops the load on a malformed allow list', () => {
    expect(() => new WebSkill({ allow_private: ['not-an-address'] })).toThrow(/allow_private/);
  });

  it('keeps text, not scripts, styles or tags', () => {
    expect(pageText(new TextEncoder().encode('<p>a</p><script>b</script><style>c</style><div>d</div>'), 'text/html')).toBe('a d');
  });

  it('answers a prompt with no URL the way the Python skill does', async () => {
    expect(await fetchWith(new WebSkill(), 'nothing here')).toBe(
      'Error: No URLs found in prompt. The prompt must contain at least one URL starting with http:// or https://.',
    );
  });
});
