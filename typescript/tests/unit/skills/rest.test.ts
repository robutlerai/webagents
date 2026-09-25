/**
 * The REST tool (`rest` skill, `rest_request`), ADR-0045 section 6.
 *
 * Every scenario in `python/tests/fixtures/rest_tool/scenarios.json` runs here
 * against a local server with the routes that file lists, and must produce
 * exactly the result it pins; the Python suite runs the same file
 * (tests/agents/skills/test_rest_skill.py), so the two SDKs answer every one of
 * these requests identically. The tool definition is pinned by
 * `definition.json` the same way.
 */

import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import * as http from 'node:http';
import type { AddressInfo } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { RestSkill } from '../../../src/skills/rest/skill';
import { BaseAgent } from '../../../src/core/agent';
import type { SigningIdentity } from '../../../src/crypto/http-signature';
import { AgentIdentity } from '../../../src/crypto/identity';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures/rest_tool');
const SCENARIOS = JSON.parse(readFileSync(path.join(FIXTURES, 'scenarios.json'), 'utf8')).scenarios as Array<{
  name: string;
  args: Record<string, unknown>;
  expect: Record<string, unknown>;
  config?: Record<string, unknown>;
  env?: Record<string, string>;
}>;
const DEFINITION = JSON.parse(readFileSync(path.join(FIXTURES, 'definition.json'), 'utf8')).definition;

const JSON_BODY = '{"hello":"world","n":1}';

function send(
  res: http.ServerResponse,
  status: number,
  body: Buffer | string = '',
  contentType?: string,
  headers: Array<[string, string]> = [],
  head = false,
): void {
  const bytes = typeof body === 'string' ? Buffer.from(body) : body;
  const out: Array<[string, string]> = [];
  if (contentType) out.push(['Content-Type', contentType]);
  out.push(...headers, ['Content-Length', String(bytes.length)]);
  res.writeHead(status, out.flat());
  if (!head && bytes.length) res.write(bytes);
  res.end();
}

function route(req: http.IncomingMessage, res: http.ServerResponse, body: string): void {
  const pathname = (req.url ?? '/').split('?')[0];
  const head = req.method === 'HEAD';
  if ((req.method === 'POST' || req.method === 'PUT') && pathname === '/echo') {
    send(
      res,
      200,
      JSON.stringify({
        method: req.method,
        content_type: req.headers['content-type'] ?? null,
        content_length: req.headers['content-length'] ?? null,
        body,
        signed: req.headers['signature-input'] !== undefined,
      }),
      'application/json',
    );
    return;
  }
  switch (pathname) {
    case '/json':
      return send(res, 200, JSON_BODY, 'application/json', [], head);
    case '/text':
      return send(res, 404, 'not here', 'text/plain; charset=utf-8');
    case '/bin':
      return send(res, 200, Buffer.from(Array.from({ length: 256 }, (_, i) => i)), 'application/octet-stream');
    case '/r1':
      return send(res, 302, '', 'text/plain; charset=utf-8', [['Location', '/r2']]);
    case '/r2':
      return send(res, 301, '', 'text/plain; charset=utf-8', [['Location', '/json']]);
    case '/loop':
      return send(res, 302, '', 'text/plain; charset=utf-8', [['Location', '/loop']]);
    case '/to-private':
      return send(res, 302, '', 'text/plain; charset=utf-8', [['Location', 'http://10.0.0.1/']]);
    case '/headers':
      return send(res, 200, 'ok', 'text/plain; charset=utf-8', [
        ['Link', '<https://example.com/next>; rel="next"'],
        ['ETag', '"v1"'],
        ['X-RateLimit-Remaining', '41'],
        ['Set-Cookie', 'session=secret'],
      ]);
    case '/big':
      return send(res, 200, Buffer.alloc(2 * 1024 * 1024, 'a'), 'text/plain; charset=utf-8');
    case '/slow':
      setTimeout(() => {
        if (!res.destroyed) send(res, 200, 'late', 'text/plain; charset=utf-8');
      }, 3000);
      return;
    default:
      return send(res, 404, 'no route', 'text/plain; charset=utf-8');
  }
}

let server: http.Server;
let port = 0;

beforeAll(async () => {
  server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => route(req, res, Buffer.concat(chunks).toString('utf8')));
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  port = (server.address() as AddressInfo).port;
});

afterAll(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

const envBefore = { ...process.env };
afterEach(() => {
  for (const key of Object.keys(process.env)) if (!(key in envBefore)) delete process.env[key];
  for (const [key, value] of Object.entries(envBefore)) process.env[key] = value;
});

function expand(value: unknown): unknown {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record);
    if (keys.length === 2 && keys.includes('repeat') && keys.includes('count')) {
      return String(record.repeat).repeat(Number(record.count));
    }
    return Object.fromEntries(keys.map((k) => [k, expand(record[k])]));
  }
  if (Array.isArray(value)) return value.map(expand);
  if (typeof value === 'string') return value.replaceAll('PORT', String(port));
  return value;
}

function skill(config?: Record<string, unknown>, identity?: SigningIdentity): RestSkill {
  const rest = new RestSkill(config ?? { allow_private: ['127.0.0.1'] });
  rest.setAgent({ identity });
  return rest;
}

describe('the shared scenarios', () => {
  for (const scenario of SCENARIOS) {
    it(scenario.name, async () => {
      for (const [name, value] of Object.entries(scenario.env ?? {})) process.env[name] = value;
      const raw = await skill(scenario.config).call(expand(scenario.args) as Record<string, unknown>);
      const result = JSON.parse(raw) as Record<string, unknown>;
      delete result.elapsed_ms;
      expect(result).toEqual(expand(scenario.expect));
    });
  }
});

describe('the tool', () => {
  it('offers the shared definition, to the owner only', () => {
    const agent = new BaseAgent({ name: 'rest-probe', skills: [new RestSkill()] });
    const tool = (agent as unknown as { toolRegistry: Map<string, { name: string; description?: string; parameters?: unknown; scopes?: string[] }> })
      .toolRegistry.get('rest_request')!;
    expect({
      type: 'function',
      function: { name: tool.name, description: tool.description, parameters: tool.parameters },
    }).toEqual(DEFINITION);
    expect(tool.scopes).toEqual(['owner']);
  });

  it('keeps its key order', async () => {
    const raw = await skill().call({ method: 'GET', url: `http://127.0.0.1:${port}/json` });
    expect(Object.keys(JSON.parse(raw))).toEqual([
      'ok', 'status', 'url', 'redirects', 'signed', 'unsigned_reason',
      'content_type', 'headers', 'text', 'bytes', 'truncated', 'elapsed_ms',
    ]);
    expect(raw.startsWith('{"ok":true,"status":200,')).toBe(true);
  });

  it('refuses a bad config the way Python does', () => {
    expect(() => new RestSkill({ sign: 'sometimes' })).toThrow('rest: sign must be auto, always or never');
    expect(() => new RestSkill({ allow_private: ['localhost'] })).toThrow(/not an IP address or CIDR range/);
  });
});

async function identity(issuer: string): Promise<SigningIdentity> {
  const id = new AgentIdentity({ agentId: 'mini', issuer });
  await id.initialize();
  return id;
}

describe('signing', () => {
  it('signs when the agent has a public address', async () => {
    // Plain http to the test server is signed only where the platform's local
    // overlay would verify it, which this switch stands for.
    process.env.ROBUTLER_AGENT_URL_ALLOW_PRIVATE = '1';
    const rest = skill(undefined, await identity('https://agents.example.com/agents/mini'));
    const result = JSON.parse(await rest.call({ method: 'POST', url: `http://127.0.0.1:${port}/echo`, body: '{}' }));
    expect(result.signed).toBe(true);
    expect(result.signed_as).toBe('https://agents.example.com/agents/mini');
    expect(JSON.parse(result.text).signed).toBe(true);
  });

  it('does not sign plain http by default', async () => {
    delete process.env.ROBUTLER_AGENT_URL_ALLOW_PRIVATE;
    const rest = skill(undefined, await identity('https://agents.example.com/agents/mini'));
    const result = JSON.parse(await rest.call({ method: 'POST', url: `http://127.0.0.1:${port}/echo`, body: '{}' }));
    expect(result.signed).toBe(false);
    expect(result.unsigned_reason).toBe('requests to plain http addresses are not signed.');
  });

  it('a loopback identity cannot sign, and never means never', async () => {
    expect(skill(undefined, await identity('http://localhost:8080/agents/mini')).signingStatus().reason).toMatch(
      /^this agent has no public address/,
    );
    expect(skill({ sign: 'never' }, await identity('https://agents.example.com/agents/mini')).signingStatus()).toEqual({
      reason: 'signing is turned off for this tool (sign: never).',
    });
  });

  it('the prompt says whether requests are signed', async () => {
    expect(skill().restGuide()).toContain('Requests go out unsigned: this agent has no public address.');
    const signed = skill(undefined, await identity('https://agents.example.com/agents/mini')).restGuide();
    expect(signed).toContain('Requests are signed as https://agents.example.com/agents/mini (Web Bot Auth)');
    expect(signed).toContain('"signed":true');
  });
});
