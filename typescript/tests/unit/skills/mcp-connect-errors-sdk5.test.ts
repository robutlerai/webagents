/**
 * What the MCP client says when a remote server answers 401 or 403
 * (2026-09-29, the skills and MCP e2e), against the shared fixture
 * `python/tests/fixtures/mcp_tool/connect_errors.json`, which the Python
 * suite reads too (`tests/agents/skills/test_mcp_connect_errors_sdk5.py`).
 *
 * `webagents doctor` pointed at the portal's `/mcp` with no credential
 * printed the transport's own line, `Streamable HTTP error: Error POSTing to
 * endpoint: {..."Unauthorized"...}`, with no word about a credential, and its
 * fix line said to fix the server's entry (the Python doctor printed
 * `unhandled errors in a TaskGroup`). Pinned here:
 *
 *  - the words are the fixture's, in both SDKs;
 *  - an error group is unwrapped to its HTTP error, however deep;
 *  - a 401 and a 403 are said as the credential the server wants, with the
 *    `${secret:<SERVER>_TOKEN}` reference and the OAuth caveat; other errors
 *    keep their own message;
 *  - a real server answering 401 gives the report row that sentence with
 *    `needsCredential`, over `http` and over `auto`, and `doctor`'s fix line
 *    is the bearer recipe, never "Fix the server's entry".
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { cliCommand } from '../../../src/cli/config-store';
import { MCP_CHECK_WORDS, mcpCheck } from '../../../src/cli/doctor';
import {
  CREDENTIAL_STATUSES,
  NEEDS_CREDENTIAL,
  credentialSecretName,
  describeConnectError,
  httpStatusOf,
  needsCredentialSentence,
  rootCause,
} from '../../../src/skills/mcp/connect-errors';
import { MCPSkill, ownerReferenceSources } from '../../../src/skills/mcp/skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const WORDS = JSON.parse(readFileSync(path.join(FIXTURES, 'mcp_tool/connect_errors.json'), 'utf8')) as {
  needs_credential: string;
  credential_statuses: number[];
  cases: Array<{ server: string; status: number; name: string; says: string }>;
  not_a_credential: number[];
};
const SECRETS = JSON.parse(readFileSync(path.join(FIXTURES, 'cli/secrets.json'), 'utf8')) as { doctor: { words: Record<string, string> } };

/** The error the MCP SDK's Streamable HTTP transport throws for a non-2xx answer: the status is its `code`. */
class StreamableHTTPError extends Error {
  constructor(readonly code: number | undefined, message: string) {
    super(`Streamable HTTP error: ${message}`);
  }
}

function refused(status: number, text = '{"jsonrpc":"2.0","error":{"code":-32000,"message":"Unauthorized"},"id":null}'): StreamableHTTPError {
  return new StreamableHTTPError(status, `Error POSTing to endpoint: ${text}`);
}

describe('the words', () => {
  it('are the fixture\'s', () => {
    expect(NEEDS_CREDENTIAL).toBe(WORDS.needs_credential);
    expect([...CREDENTIAL_STATUSES]).toEqual(WORDS.credential_statuses);
    expect(MCP_CHECK_WORDS.fixCredential).toBe(SECRETS.doctor.words.fixCredential);
    expect(NEEDS_CREDENTIAL).not.toContain('—');
    expect(MCP_CHECK_WORDS.fixCredential).not.toContain('—');
  });

  it.each(WORDS.cases)('a $status from $server is said as the credential the server wants', (c) => {
    expect(credentialSecretName(c.server)).toBe(c.name);
    expect(needsCredentialSentence(c.server, c.status)).toBe(c.says);
    expect(describeConnectError(c.server, refused(c.status))).toEqual({ message: c.says, needsCredential: true });
    const group = new AggregateError([new AggregateError([refused(c.status)], 'inner')], 'outer');
    expect(describeConnectError(c.server, group)).toEqual({ message: c.says, needsCredential: true });
  });

  it('an error group is unwrapped to its HTTP error, and a plain error is itself', () => {
    const error = refused(401);
    expect(rootCause(new AggregateError([new AggregateError([error])]))).toBe(error);
    expect(rootCause(new AggregateError([new Error('closed'), error]))).toBe(error);
    const plain = new Error('boom');
    expect(rootCause(new AggregateError([plain]))).toBe(plain);
    expect(rootCause(plain)).toBe(plain);
    expect(describeConnectError('x', new AggregateError([plain]))).toEqual({ message: 'boom', needsCredential: false });
  });

  it('reads the status of the transports\' errors only under their own prefixes, never a JSON-RPC code', () => {
    expect(httpStatusOf(refused(401))).toBe(401);
    expect(httpStatusOf(Object.assign(new Error('SSE error: Non-200 status code (401)'), { code: 401 }))).toBe(401);
    expect(httpStatusOf(Object.assign(new Error('MCP error -32000: Connection closed'), { code: -32000 }))).toBeUndefined();
    expect(httpStatusOf(Object.assign(new Error('Unauthorized'), { code: 401 }))).toBeUndefined();
    expect(httpStatusOf(Object.assign(new Error('x'), { status: 403 }))).toBe(403);
    expect(httpStatusOf(new Error('x'))).toBeUndefined();
    expect(httpStatusOf(undefined)).toBeUndefined();
  });

  it.each(WORDS.not_a_credential)('a %i keeps the transport\'s own message', (status) => {
    const error = refused(status, 'nope');
    expect(describeConnectError('robutler', error)).toEqual({ message: error.message, needsCredential: false });
  });
});

describe('a real server that answers 401', () => {
  const ISOLATED = ['HOME', 'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_PROFILE', 'WEBAGENTS_SECRETS_DIR'];
  const saved: Record<string, string | undefined> = {};
  const tempDir = tempDirs();
  let server: Server | undefined;

  beforeEach(() => {
    for (const name of ISOLATED) {
      saved[name] = process.env[name];
      delete process.env[name];
    }
    process.env.HOME = tempDir('wa-mcp-401-home-');
    process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
    for (const method of ['log', 'warn', 'error', 'info', 'debug'] as const) vi.spyOn(console, method).mockImplementation(() => {});
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    for (const name of ISOLATED) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
    await new Promise<void>((resolve) => (server ? server.close(() => resolve()) : resolve()));
    server = undefined;
  });

  /** Every request refused with 401, as the portal's `/mcp` refuses a request with no credential. */
  async function refusingUrl(): Promise<string> {
    server = createServer((_req, res) => {
      res.writeHead(401, { 'Content-Type': 'application/json', 'WWW-Authenticate': 'Bearer realm="portal"' });
      res.end('{"jsonrpc":"2.0","error":{"code":-32000,"message":"Unauthorized"},"id":null}');
    });
    await new Promise<void>((resolve) => server!.listen(0, '127.0.0.1', () => resolve()));
    const { port } = server!.address() as { port: number };
    return `http://127.0.0.1:${port}/mcp`;
  }

  it.each(['http', 'auto'] as const)('over %s the report row says the sentence, and doctor\'s fix is the bearer recipe', async (transport) => {
    const url = await refusingUrl();
    const skill = new MCPSkill({ mcp: { robutler: { url, transport } } as never, references: ownerReferenceSources() });
    const agent = new BaseAgent({ name: 'client', instructions: 'x', skills: [skill] });
    await agent.initialize();
    try {
      const report = skill.serverReport();
      const expected = needsCredentialSentence('robutler', 401);
      expect(expected).toBe(WORDS.cases.find((c) => c.server === 'robutler')!.says);
      expect(report[0]).toMatchObject({ name: 'robutler', connected: false, error: expected, needsCredential: true });
      expect(JSON.stringify(report)).not.toContain('Streamable HTTP error');

      const check = mcpCheck(report);
      expect(check.status).toBe('fail');
      expect(check.detail).toBe(`robutler: ${expected}`);
      expect(check.fix).toBe(
        MCP_CHECK_WORDS.fixCredential
          .replace('{name}', 'ROBUTLER_TOKEN')
          .replace('{server}', 'robutler')
          .replace('{hint}', cliCommand('secrets set ROBUTLER_TOKEN')),
      );
      expect(check.fix).not.toContain(MCP_CHECK_WORDS.fixEntry);
    } finally {
      await agent.cleanup();
    }
  });
});
