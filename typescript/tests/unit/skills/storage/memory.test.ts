/**
 * Unit tests for RobutlerMemorySkill (memory tool, file-system metaphor)
 *
 * The memory tool is registered via registerTool() with a handler,
 * so we invoke it through the skill's tools array.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { RobutlerMemorySkill } from '../../../../src/skills/storage/skill.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createMockContext(): any {
  return {
    auth: { authenticated: true, agentId: 'test-agent', userId: 'owner-123' },
    metadata: {},
    session: {},
    get: vi.fn(),
    set: vi.fn(),
    delete: vi.fn(),
    hasScope: vi.fn().mockReturnValue(true),
    hasScopes: vi.fn().mockReturnValue(true),
  };
}

function mockResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

/**
 * Invoke the 'memory' tool registered on the skill.
 */
function callMemory(skill: RobutlerMemorySkill, params: Record<string, unknown>, ctx: any): Promise<unknown> {
  const memoryTool = (skill as any).tools.find((t: any) => t.name === 'memory');
  if (!memoryTool?.handler) throw new Error('memory tool not found on skill');
  return memoryTool.handler(params, ctx);
}

// ---------------------------------------------------------------------------
// Global fetch mock
// ---------------------------------------------------------------------------

const originalFetch = globalThis.fetch;

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('RobutlerMemorySkill (file-system interface)', () => {
  const skill = new RobutlerMemorySkill({
    portalUrl: 'http://localhost:3000',
    apiKey: 'test-key',
    agentId: 'agent-1',
  });

  const ctx = createMockContext();

  describe('view command', () => {
    it('view directory lists entries', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { keys: ['topic-a', 'topic-b'] }),
      );

      const result = await callMemory(skill, { command: 'view', path: '/memories/' }, ctx);

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/storage/memory?'),
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
      expect(result).toEqual({ keys: ['topic-a', 'topic-b'] });
    });

    it('view file reads entry', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { value: 'stored content' }),
      );

      const result = await callMemory(skill, { command: 'view', path: '/memories/topic.md' }, ctx);

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/storage/memory/topic'),
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
      expect(result).toEqual({ value: 'stored content' });
    });

    it('view returns null for 404', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(404, {}),
      );

      const result = await callMemory(skill, { command: 'view', path: '/memories/missing.md' }, ctx);
      expect(result).toBeNull();
    });
  });

  describe('create command', () => {
    it('create calls PUT with content', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, {}),
      );

      const result = await callMemory(
        skill,
        { command: 'create', path: '/memories/notes.md', content: 'hello world' },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/storage/memory/notes'),
        expect.objectContaining({
          method: 'PUT',
          headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
        }),
      );
      const call = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
      const body = JSON.parse(call[1].body);
      expect(body.value).toBe('hello world');
      expect(body.agentId).toBe('agent-1');
      expect(result).toBe('Created /memories/notes.md');
    });

    it('create requires a key (or legacy path with filename) and content', async () => {
      // No path / no scope+key → _normalizeScopeKey turns the empty input into
      // '/memories/' which is the bucket root; create then complains that the
      // key is missing.
      const noKey = await callMemory(skill, { command: 'create', content: 'x' }, ctx);
      expect(noKey).toEqual({ error: 'path must include a filename for create' });

      const noContent = await callMemory(skill, { command: 'create', path: '/memories/x.md' }, ctx);
      expect(noContent).toEqual({ error: 'content is required for create' });

      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });

  describe('edit command', () => {
    it('edit performs str_replace on stored value and returns a mini-diff', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockResponse(200, { value: 'hello world' }))
        .mockResolvedValueOnce(mockResponse(200, {}));

      const result = await callMemory(
        skill,
        { command: 'edit', path: '/memories/notes.md', old_str: 'hello', new_str: 'goodbye' },
        ctx,
      );

      expect(typeof result).toBe('string');
      expect(result as string).toMatch(/^Edited \/memories\/notes\.md/);
      expect(result as string).toContain('[hello]');
      expect(result as string).toContain('[goodbye]');
      expect(globalThis.fetch).toHaveBeenCalledTimes(2);
    });

    it('edit requires path, old_str, new_str', async () => {
      const result = await callMemory(skill, { command: 'edit', path: '/memories/x.md' }, ctx);
      expect(result).toEqual({ error: 'old_str and new_str are required for edit' });
    });

    it('rejects editing non-string values (object) without issuing a PUT', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { value: { ok: true } }),
      );

      const result = await callMemory(
        skill,
        { command: 'edit', path: '/memories/json', old_str: 'foo', new_str: 'bar' },
        ctx,
      );
      expect(result).toEqual({
        error: expect.stringMatching(/Cannot edit \/memories\/json: value is object, not text/),
      });
      expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    });

    it('rejects null current values without issuing a PUT', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { value: null }),
      );

      const result = await callMemory(
        skill,
        { command: 'edit', path: '/memories/k', old_str: 'foo', new_str: 'bar' },
        ctx,
      );
      expect(result).toEqual({
        error: expect.stringMatching(/value is null, not text/),
      });
      expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    });

    it('rejects when old_str matches multiple times (no silent partial edit)', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { value: 'foo bar foo baz foo' }),
      );

      const result = await callMemory(
        skill,
        { command: 'edit', path: '/memories/k', old_str: 'foo', new_str: 'qux' },
        ctx,
      );
      expect(result).toEqual({
        error: expect.stringMatching(/old_str matches 3 times/),
      });
      expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    });

    it('replaces literally — `$&`, `$1`, `$$` survive verbatim in the PUT body', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockResponse(200, { value: 'price: TOKEN end' }))
        .mockResolvedValueOnce(mockResponse(200, {}));

      await callMemory(
        skill,
        { command: 'edit', path: '/memories/k', old_str: 'TOKEN', new_str: '$& and $1 and $$' },
        ctx,
      );

      const putCall = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[1];
      const body = JSON.parse(putCall[1].body);
      expect(body.value).toBe('price: $& and $1 and $$ end');
    });
  });

  describe('delete command', () => {
    it('delete calls DELETE', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, {}),
      );

      const result = await callMemory(
        skill,
        { command: 'delete', path: '/memories/old.md' },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/storage/memory/old'),
        expect.objectContaining({ method: 'DELETE' }),
      );
      expect(result).toBe('Deleted /memories/old.md');
    });

    it('delete requires a key (or legacy path with filename)', async () => {
      // After scope/key normalization, an empty input lands as `/memories/`
      // (bucket root, no key) — delete then refuses without hitting fetch.
      const result = await callMemory(skill, { command: 'delete' }, ctx);
      expect(result).toEqual({ error: 'path must include a filename for delete' });
    });
  });

  describe('rename command', () => {
    it('rename performs get + set + delete', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockResponse(200, { value: 'content' }))
        .mockResolvedValueOnce(mockResponse(200, {}))
        .mockResolvedValueOnce(mockResponse(200, {}));

      const result = await callMemory(
        skill,
        { command: 'rename', path: '/memories/old.md', new_str: '/memories/new.md' },
        ctx,
      );

      expect(result).toContain('Renamed');
      expect(globalThis.fetch).toHaveBeenCalledTimes(3);
    });

    it('rename requires path and new_str', async () => {
      const result = await callMemory(skill, { command: 'rename', path: '/memories/x.md' }, ctx);
      expect(result).toEqual({ error: 'path and new_str (new path) are required for rename' });
    });
  });

  describe('search command', () => {
    it('search calls GET with query params', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { results: [{ key: 'notes', value: 'found' }] }),
      );

      const result = await callMemory(skill, { command: 'search', query: 'hello' }, ctx);

      const url = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(url).toContain('action=search');
      expect(url).toContain('q=hello');
      expect(result).toEqual({ results: [{ key: 'notes', value: 'found' }] });
    });

    it('search requires query', async () => {
      const result = await callMemory(skill, { command: 'search' }, ctx);
      expect(result).toEqual({ error: 'query is required for search' });
    });
  });

  describe('stores command', () => {
    it('stores lists accessible stores', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { stores: ['store-1', 'store-2'] }),
      );

      const result = await callMemory(skill, { command: 'stores' }, ctx);
      expect(result).toEqual({ stores: ['store-1', 'store-2'] });
    });
  });

  describe('unknown command', () => {
    it('returns error', async () => {
      const result = await callMemory(skill, { command: 'invalid' }, ctx);
      expect(result).toEqual({ error: expect.stringContaining('Unknown command: invalid') });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });

  describe('tool parameters', () => {
    it('exposes the standard memory commands (share/unshare are intentionally hidden from the LLM)', () => {
      // share/unshare were removed from the LLM-facing schema — see the
      // long comment in skill.ts above the registerTool call. The handler
      // still implements them (the share/unshare describe blocks below
      // exercise that path directly), but the LLM tool surface no longer
      // advertises them. Re-enable here when sharing is re-introduced.
      const memoryTool = (skill as any).tools.find((t: any) => t.name === 'memory');
      expect(memoryTool).toBeDefined();
      const cmds: string[] = memoryTool.parameters.properties.command.enum;
      expect(cmds).toEqual(['view', 'create', 'edit', 'delete', 'rename', 'search', 'stores']);
      expect(cmds).not.toContain('share');
      expect(cmds).not.toContain('unshare');
      // share-only properties are also gone from the schema.
      expect(memoryTool.parameters.properties).not.toHaveProperty('agent');
      expect(memoryTool.parameters.properties).not.toHaveProperty('level');
    });

    it('schema is strict: additionalProperties:false, path constraints, scope enum, oneOf per-command required', () => {
      // Mirrors tests/unit/llm/platform-tools.test.ts — the SDK and the
      // platform tool definition must stay in lockstep so SDK-built agents
      // see the same validation surface as platform agents.
      const memoryTool = (skill as any).tools.find((t: any) => t.name === 'memory');
      const params = memoryTool.parameters;

      expect(params.additionalProperties).toBe(false);
      // path is the LEGACY back-compat parameter — preferred addressing is
      // scope+key, but the constraint stays so old callers still type-check.
      expect(params.properties.path.pattern).toBe('^/memories(/.*)?$');
      expect(params.properties.path.maxLength).toBe(256);

      // scope is the new addressing, exposing exactly the three built-in
      // buckets the visibility/memory plan calls out.
      expect(params.properties.scope.enum).toEqual(['agent', 'user', 'chat']);

      const cmdDesc: string = params.properties.command.description ?? '';
      for (const cmd of params.properties.command.enum) {
        expect(cmdDesc).toMatch(new RegExp(`\\b${cmd}\\b`));
      }

      const oneOfMap = new Map<string, string[]>(
        params.oneOf.map((b: any) => [b.properties.command.const, b.required ?? []]),
      );
      expect(new Set(oneOfMap.keys())).toEqual(new Set(params.properties.command.enum));
      // After the scope/key refactor, `path` is no longer part of the
      // per-command required set — the handler resolves it from
      // (scope|store, key) instead.
      expect(oneOfMap.get('create')).toEqual(expect.arrayContaining(['command', 'content']));
      expect(oneOfMap.get('edit')).toEqual(
        expect.arrayContaining(['command', 'old_str', 'new_str']),
      );
      // delete carries no per-command required fields beyond the top-level
      // `command` (handler validates key resolution at runtime).
      expect(oneOfMap.get('delete')).toEqual([]);
      expect(oneOfMap.get('rename')).toEqual(expect.arrayContaining(['command', 'new_str']));
      expect(oneOfMap.get('search')).toEqual(expect.arrayContaining(['command', 'query']));
      expect(oneOfMap.get('view')).toEqual([]);
      expect(oneOfMap.get('stores')).toEqual([]);
    });
  });

  describe('share command', () => {
    it('POSTs to /api/storage/memory?action=share with the supplied level', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(200, {}));

      const result = await callMemory(
        skill,
        { command: 'share', agent: 'agent-2', level: 'readwrite' },
        ctx,
      );

      expect(result).toBe('OK');
      const [url, init] = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
      expect(url).toContain('/api/storage/memory?action=share');
      expect(init.method).toBe('POST');
      const body = JSON.parse(init.body);
      expect(body).toMatchObject({
        agentId: 'agent-1',
        store: 'agent-1',
        grantee: 'agent-2',
        level: 'readwrite',
      });
    });

    it('defaults level to "read" when not provided', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(200, {}));

      await callMemory(skill, { command: 'share', agent: 'agent-2' }, ctx);

      const init = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1];
      const body = JSON.parse(init.body);
      expect(body.level).toBe('read');
    });

    it('forwards the new "search" enum value (FTS-only grant) verbatim', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(200, {}));

      await callMemory(
        skill,
        { command: 'share', agent: 'agent-2', level: 'search' },
        ctx,
      );

      const body = JSON.parse((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body);
      expect(body.level).toBe('search');
    });

    it('extracts store id from /memories/shared/<id>/... when path is supplied', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(200, {}));

      await callMemory(
        skill,
        { command: 'share', path: '/memories/shared/agent-9/notes.md', agent: 'agent-2' },
        ctx,
      );

      const body = JSON.parse((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body);
      expect(body.store).toBe('agent-9');
    });

    it('requires the agent argument', async () => {
      const result = await callMemory(skill, { command: 'share' }, ctx);
      expect(result).toEqual({ error: 'agent is required for share' });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });

    it('maps 403 to a write-access error', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(403, {}));
      const result = await callMemory(
        skill,
        { command: 'share', agent: 'agent-2' },
        ctx,
      );
      expect(result).toEqual({ error: 'You need write access to share this store' });
    });
  });

  describe('unshare command', () => {
    it('DELETEs ?action=share with store + grantee in the query string', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(200, {}));

      const result = await callMemory(
        skill,
        { command: 'unshare', agent: 'agent-2' },
        ctx,
      );

      expect(result).toBe('OK');
      const [url, init] = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
      expect(init.method).toBe('DELETE');
      expect(url).toContain('action=share');
      expect(url).toContain('store=agent-1');
      expect(url).toContain('grantee=agent-2');
    });

    it('uses the path-derived store when /memories/shared/<id>/ is supplied', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(200, {}));

      await callMemory(
        skill,
        { command: 'unshare', path: '/memories/shared/agent-9/', agent: 'agent-2' },
        ctx,
      );

      const url = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(url).toContain('store=agent-9');
    });

    it('requires the agent argument', async () => {
      const result = await callMemory(skill, { command: 'unshare' }, ctx);
      expect(result).toEqual({ error: 'agent is required for unshare' });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });

    it('maps 403 to a write-access error', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(403, {}));
      const result = await callMemory(
        skill,
        { command: 'unshare', agent: 'agent-2' },
        ctx,
      );
      expect(result).toEqual({ error: 'You need write access to unshare this store' });
    });
  });
});
