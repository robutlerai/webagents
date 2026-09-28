/**
 * The ACP agent, in process (gap-closure plan item 1.6, 2026-09-26): what the
 * fixture `python/tests/fixtures/acp/acp_protocol.json` pins, the same in
 * both SDKs (Python: tests/test_acp_protocol.py): the constants, the tool
 * kinds, the transport entries an agent file may write, and the JSON-RPC
 * surface driven over an in-memory line pair. The S-249 pin lives here too:
 * the `acp` name is the Agent Client Protocol, with no tool that fetches a
 * caller-named URL and no HTTP or WebSocket route; and the client's own
 * `fs/*` and `terminal/*` methods are not served over stdio (S-269's shape).
 * The spawned-CLI transcripts are `tests/unit/cli/acp-stdio.test.ts`.
 */

import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { resolveSkillsByName } from '../../../src/skills/resolve';
import * as P from '../../../src/skills/transport/acp/protocol';
import { ACPTransportSkill, AcpSession, SessionStore } from '../../../src/skills/transport/acp/skill';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROTOCOL = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/acp/acp_protocol.json'), 'utf8')) as {
  protocol_version: number;
  agent_capabilities: unknown;
  auth_methods: unknown;
  agent_info_name: string;
  session_id_prefix: string;
  errors: Record<string, number>;
  client_methods_not_served: string[];
  stop_reasons: { end_turn: string; cancelled: string };
  permission: { kinds: string[]; options: unknown; rejected: string; cancelled: string };
  tool_kinds: { cases: { name: string; kind: string }[] };
  config_shapes: Record<string, { defaults: Record<string, unknown>; shapes: { name: string; config: Record<string, unknown>; settings: Record<string, unknown> }[] }>;
};

const fill = (template: string, tool: string) => template.replace('{tool}', tool);

describe('the constants are the fixture’s', () => {
  it('version, capabilities, auth methods, ids, codes, permission', () => {
    expect(P.PROTOCOL_VERSION).toBe(PROTOCOL.protocol_version);
    expect(P.AGENT_CAPABILITIES).toEqual(PROTOCOL.agent_capabilities);
    expect(P.AUTH_METHODS).toEqual(PROTOCOL.auth_methods);
    expect(P.AGENT_INFO_NAME).toBe(PROTOCOL.agent_info_name);
    expect(P.SESSION_ID_PREFIX).toBe(PROTOCOL.session_id_prefix);
    const e = PROTOCOL.errors;
    expect([P.PARSE_ERROR, P.INVALID_REQUEST, P.METHOD_NOT_FOUND, P.INVALID_PARAMS]).toEqual([e.parse, e.invalid_request, e.method_not_found, e.invalid_params]);
    expect([P.INTERNAL_ERROR, P.AUTH_REQUIRED, P.RESOURCE_NOT_FOUND, P.REQUEST_CANCELLED]).toEqual([e.internal, e.auth_required, e.resource_not_found, e.request_cancelled]);
    expect([...P.PERMISSION_KINDS].sort()).toEqual([...PROTOCOL.permission.kinds].sort());
    expect(P.PERMISSION_OPTIONS).toEqual(PROTOCOL.permission.options);
    expect(P.rejected('write_file')).toBe(fill(PROTOCOL.permission.rejected, 'write_file'));
    expect(P.cancelled('write_file')).toBe(fill(PROTOCOL.permission.cancelled, 'write_file'));
    expect([P.STOP_END_TURN, P.STOP_CANCELLED]).toEqual([PROTOCOL.stop_reasons.end_turn, PROTOCOL.stop_reasons.cancelled]);
  });

  it.each(PROTOCOL.tool_kinds.cases)('tool kind of $name is $kind', ({ name, kind }) => {
    expect(P.toolKind(name)).toBe(kind);
    expect(P.needsPermission(kind as P.ToolKind)).toBe(PROTOCOL.permission.kinds.includes(kind));
  });
});

describe('the transport entries an agent file may write', () => {
  it.each(Object.keys(PROTOCOL.config_shapes).filter((key) => key !== 'about'))('%s resolves as the fixture says', async (name) => {
    const shapes = PROTOCOL.config_shapes[name];
    for (const shape of shapes.shapes) {
      const { byName, unknown, failed } = await resolveSkillsByName([{ [name]: shape.config }], { agentDir: tmpdir() });
      expect(unknown, shape.name).toEqual([]);
      expect(failed, shape.name).toEqual([]);
      expect((byName.get(name) as unknown as { settings: unknown }).settings, shape.name).toEqual(shape.settings);
    }
    const bare = await resolveSkillsByName([name], { agentDir: tmpdir() });
    expect((bare.byName.get(name) as unknown as { settings: unknown }).settings).toEqual(shapes.defaults);
  });
});

describe('the acp skill is the Agent Client Protocol and nothing else (S-249, S-269)', () => {
  it('has no tools, no HTTP route and no WebSocket route', () => {
    const skill = new ACPTransportSkill();
    expect(skill.tools).toEqual([]);
    expect(skill.httpEndpoints).toEqual([]);
    expect(skill.wsEndpoints).toEqual([]);
    expect(skill.hooks.map((h) => h.lifecycle)).toEqual(['before_tool']);
    const source = readFileSync(path.resolve(HERE, '../../../src/skills/transport/acp/skill.ts'), 'utf8');
    expect(source).not.toMatch(/acp_get_catalog|acp_execute|acp_list_receipts|Agent Commerce Protocol Transport/);
  });
});

/** Serve `agent` over an in-memory line pair: every message in, every line out, parsed. */
async function drive(messages: unknown[], skill = new ACPTransportSkill(), agent?: BaseAgent): Promise<Array<Record<string, unknown>>> {
  const built = agent ?? new BaseAgent({ name: 't', instructions: '', skills: [skill] });
  const out: string[] = [];
  async function* lines(): AsyncGenerator<string> {
    for (const message of messages) {
      yield typeof message === 'string' ? message : JSON.stringify(message);
      // Let the request's task answer before the next line, as a client would wait.
      for (let i = 0; i < 20; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  await skill.serve(built as never, lines(), (line) => out.push(line));
  return out.map((line) => JSON.parse(line) as Record<string, unknown>);
}

describe('the JSON-RPC surface, driven in process', () => {
  it('initialize negotiates the version and advertises the fixture', async () => {
    const replies = await drive([{ jsonrpc: '2.0', id: 0, method: 'initialize', params: { protocolVersion: 99, clientCapabilities: { terminal: true } } }]);
    const result = replies[0].result as Record<string, unknown>;
    expect(replies[0]).toEqual({ jsonrpc: '2.0', id: 0, result });
    expect(result.protocolVersion).toBe(1);
    expect(result.agentCapabilities).toEqual(PROTOCOL.agent_capabilities);
    expect(result.authMethods).toEqual(PROTOCOL.auth_methods);
    expect(result.agentInfo).toEqual({ name: 'webagents', title: 't', version: expect.stringMatching(/^\d+\.\d+\.\d+/) });
  });

  it('tells requests and notifications apart by id, and refuses what is not a message', async () => {
    const replies = await drive([
      { jsonrpc: '2.0', id: 0, method: 'no/such' },
      { jsonrpc: '2.0', method: 'no/such' },
      { jsonrpc: '2.0', method: 'session/cancel', params: { sessionId: 'sess_none' } },
      { jsonrpc: '2.0', id: 0, result: {} },
      '{not json',
      [],
      { jsonrpc: '2.0', id: 7 },
    ]);
    expect(replies).toEqual([
      { jsonrpc: '2.0', id: 0, error: { code: -32601, message: 'Method not found: no/such' } },
      { jsonrpc: '2.0', id: null, error: { code: -32700, message: 'Parse error' } },
      { jsonrpc: '2.0', id: null, error: { code: -32600, message: 'Invalid request' } },
      { jsonrpc: '2.0', id: 7, error: { code: -32600, message: 'Invalid request' } },
    ]);
  });

  it.each(PROTOCOL.client_methods_not_served)('does not serve %s and runs nothing', async (method) => {
    const dir = await mkdtemp(path.join(tmpdir(), 'wa-acp-'));
    const marker = path.join(dir, 'pwned');
    const params = { sessionId: 'x', path: marker, content: 'x', command: 'touch', args: [marker], cwd: dir };
    const replies = await drive([{ jsonrpc: '2.0', id: 1, method, params }]);
    expect(replies).toEqual([{ jsonrpc: '2.0', id: 1, error: { code: -32601, message: `Method not found: ${method}` } }]);
    expect(existsSync(marker)).toBe(false);
  });

  it('session/new validates its params and persists the session', async () => {
    const dir = await mkdtemp(path.join(tmpdir(), 'wa-acp-'));
    const skill = new ACPTransportSkill({ sessionsDir: path.join(dir, 'sessions') });
    const replies = await drive(
      [
        { jsonrpc: '2.0', id: 1, method: 'session/new', params: { cwd: 'relative', mcpServers: [] } },
        { jsonrpc: '2.0', id: 2, method: 'session/new', params: { cwd: dir } },
        { jsonrpc: '2.0', id: 3, method: 'session/new', params: { cwd: dir, mcpServers: [] } },
        { jsonrpc: '2.0', id: 4, method: 'session/list', params: {} },
        { jsonrpc: '2.0', id: 5, method: 'session/prompt', params: { sessionId: 'sess_missing', prompt: [{ type: 'text', text: 'hi' }] } },
        { jsonrpc: '2.0', id: 6, method: 'session/load', params: { sessionId: '../../etc/passwd', cwd: dir, mcpServers: [] } },
        { jsonrpc: '2.0', id: 7, method: 'authenticate', params: { methodId: 'login' } },
        { jsonrpc: '2.0', id: 8, method: 'authenticate', params: { methodId: 'nope' } },
      ],
      skill,
    );
    expect((replies[0].error as { code: number }).code).toBe(-32602);
    expect((replies[1].error as { code: number }).code).toBe(-32602);
    const sessionId = (replies[2].result as { sessionId: string }).sessionId;
    expect(sessionId).toMatch(/^sess_/);
    const record = JSON.parse(await readFile(path.join(dir, 'sessions', `${sessionId}.json`), 'utf8')) as Record<string, unknown>;
    expect(record).toMatchObject({ sessionId, cwd: dir, agent: 't', messages: [] });
    expect((replies[3].result as { sessions: Array<{ sessionId: string; cwd: string }> }).sessions).toEqual([
      { sessionId, cwd: dir, updatedAt: expect.any(String) },
    ]);
    expect((replies[4].error as { code: number }).code).toBe(-32002);
    expect((replies[5].error as { code: number }).code).toBe(-32602);
    expect(replies[6]).toEqual({ jsonrpc: '2.0', id: 7, result: {} });
    expect((replies[7].error as { code: number }).code).toBe(-32602);
  });
});

describe('the session store', () => {
  it('never joins an arbitrary id into a path, and round-trips a session', async () => {
    const dir = await mkdtemp(path.join(tmpdir(), 'wa-acp-store-'));
    const store = new SessionStore(dir);
    expect(store.pathOf('../../etc/passwd')).toBeNull();
    expect(store.pathOf('sess_ok.1-2')).toBe(path.join(dir, 'sess_ok.1-2.json'));
    expect(await store.load('../x')).toBeNull();
    const session = new AcpSession('sess_a', '/p', 't', '2026-09-26T00:00:00Z', '2026-09-26T00:00:01Z');
    session.title = 'hi';
    session.messages = [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'yo' }];
    await store.save(session);
    const loaded = await store.load('sess_a');
    expect(loaded?.toRecord()).toEqual(session.toRecord());
    expect((await store.listAll()).map((s) => s.sessionId)).toEqual(['sess_a']);
  });
});

describe('the mappings', () => {
  it('prompt text, and the blocks it refuses', () => {
    expect(P.promptText([{ type: 'text', text: 'hi' }])).toBe('hi');
    expect(
      P.promptText([
        { type: 'text', text: 'see' },
        { type: 'resource', resource: { uri: 'file:///a.txt', text: 'A' } },
        { type: 'resource_link', uri: 'file:///b.txt' },
      ]),
    ).toBe('see\n\n[Resource: file:///a.txt]\nA\n\n[Resource link: file:///b.txt]');
    for (const bad of [[], 'hi', [{ type: 'image', data: '', mimeType: 'image/png' }], [{ type: 'nope' }], [1]]) {
      expect(() => P.promptText(bad)).toThrow(expect.objectContaining({ code: -32602 }));
    }
  });

  it('mcp servers from both entry shapes', () => {
    expect(
      P.mcpServersConfig([
        { name: 'demo', command: 'python', args: ['server.py'], env: [{ name: 'A', value: '1' }] },
        { type: 'http', name: 'remote', url: 'https://x.example/mcp', headers: [{ name: 'Authorization', value: 'Bearer t' }] },
        { type: 'sse', name: 'old', url: 'https://x.example/sse' },
        { name: 'broken' },
        'not an object',
      ]),
    ).toEqual({
      demo: { command: 'python', args: ['server.py'], env: { A: '1' } },
      remote: { url: 'https://x.example/mcp', headers: { Authorization: 'Bearer t' }, transport: 'http' },
      old: { url: 'https://x.example/sse', headers: {}, transport: 'sse' },
    });
  });

  it('plan entries from a todo list, and forgiving arguments', () => {
    expect(
      P.planEntries([
        { content: 'a', priority: 'critical', status: 'in_progress' },
        { content: 'b', priority: 'low', status: 'cancelled' },
        { content: 'c' },
      ]),
    ).toEqual([
      { content: 'a', priority: 'high', status: 'in_progress' },
      { content: 'c', priority: 'medium', status: 'pending' },
    ]);
    expect(P.planEntries(null)).toBeNull();
    expect(P.parseArguments('{"a": 1}')).toEqual({ a: 1 });
    expect(P.parseArguments('not json')).toEqual({});
    expect(P.parseArguments('[1]')).toEqual({});
    expect(P.parseArguments({ b: 2 })).toEqual({ b: 2 });
    expect(P.parseArguments(undefined)).toEqual({});
  });
});
