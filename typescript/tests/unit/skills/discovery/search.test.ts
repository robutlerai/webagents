/**
 * Unit tests for PortalDiscoverySkill.search tool
 *
 * Every skill here holds a platform key. These tests are about the search
 * plumbing (which route, which query string, which body), not about the
 * credential, and since 2026-09-23 a skill with neither a signing identity
 * nor a key refuses before it dials anything. The credential rule itself is
 * pinned in credentials.test.ts.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BaseAgent } from '../../../../src/core/agent.js';
import { setAgentTrace } from '../../../../src/core/trace.js';
import { NO_DISCOVERY_CREDENTIAL, NO_DISCOVERY_SIGN_IN, PortalDiscoverySkill } from '../../../../src/skills/discovery/skill.js';

/**
 * What the CLI's `resolvePlatformUrl` answers, per test. The real one reads
 * `~/.webagents`, which a test must not depend on (the developer's own
 * `platform.url` would leak in) and cannot move (`os.homedir()` ignores a
 * worker's `process.env.HOME`). Unset: the real function.
 */
const cliPlatform = vi.hoisted(() => ({ answer: undefined as undefined | (() => [string, string]) }));
vi.mock('../../../../src/cli/config-store.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../../src/cli/config-store.js')>();
  return {
    ...actual,
    resolvePlatformUrl: (...args: Parameters<typeof actual.resolvePlatformUrl>) =>
      cliPlatform.answer ? cliPlatform.answer() : actual.resolvePlatformUrl(...args),
  };
});

const HERE = path.dirname(fileURLToPath(import.meta.url));
/** The definition both SDKs offer (`python/tests/agents/skills/test_discovery_search.py` checks the same file). */
const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../../python/tests/fixtures/discovery_tool/definition.json'), 'utf8'),
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createMockContext(): any {
  return {
    auth: { authenticated: true, agentId: 'test-agent' },
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
    text: () => Promise.resolve(typeof body === 'string' ? body : JSON.stringify(body)),
  } as Response;
}

const PORTAL_URL = 'https://portal.test';
const originalFetch = globalThis.fetch;

/**
 * Route incoming fetch calls to the correct mock response based on URL pattern.
 */
function routedFetch(routes: Record<string, Response>): typeof globalThis.fetch {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString();
    for (const [pattern, response] of Object.entries(routes)) {
      if (url.includes(pattern)) return response;
    }
    return mockResponse(404, {});
  }) as any;
}

// ---------------------------------------------------------------------------
// Global fetch mock
// ---------------------------------------------------------------------------

beforeEach(() => {
  globalThis.fetch = vi.fn().mockResolvedValue(mockResponse(404, {}));
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('PortalDiscoverySkill.search', () => {
  it('calls POST /api/intents/search for intents and GET /api/discovery/agents for agents', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'test-key', timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/intents/search': mockResponse(200, {
        results: [{ intent: 'generate images', agentId: 'agent-1', score: 0.9 }],
      }),
      '/api/discovery/agents': mockResponse(200, {
        agents: [{ id: 'agent-1', username: 'image-gen', displayName: 'Image Generator' }],
      }),
    });

    const result = await skill.search(
      { query: 'generate images', types: ['intents', 'agents'], limit: 5 },
      ctx,
    );

    const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls;
    const intentCall = calls.find((c: any) => c[0].includes('/api/intents/search'));
    expect(intentCall).toBeDefined();
    expect(intentCall![1].method).toBe('POST');
    expect(intentCall![1].body).toContain('"query":"generate images"');
    expect(intentCall![1].body).toContain('"limit":5');

    const agentCall = calls.find((c: any) => c[0].includes('/api/discovery/agents'));
    expect(agentCall).toBeDefined();
    expect(agentCall![0]).toContain('search=generate+images');
    expect(agentCall![0]).toContain('type=agent');

    expect(result.intents).toEqual([{ intent: 'generate images', agentId: 'agent-1', score: 0.9 }]);
    expect(result.agents).toEqual([{
      username: 'image-gen',
      display_name: 'Image Generator',
      bio: undefined,
      reputation: 0,
      trust_level: 'standard',
      tier: undefined,
      is_online: undefined,
    }]);
  });

  it('calls correct per-type discovery endpoints for posts/channels/users/tags', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'key', timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/discovery/posts': mockResponse(200, { posts: [{ id: 'p1', title: 'AI post' }] }),
    });

    const result = await skill.search(
      { query: 'artificial intelligence', types: ['posts'], limit: 20 },
      ctx,
    );

    const postCall = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls.find(
      (c: any) => c[0].includes('/api/discovery/posts'),
    );
    expect(postCall).toBeDefined();
    expect(postCall![0]).toContain('q=artificial+intelligence');
    expect(postCall![0]).toContain('limit=20');
    // Cut to what the tool promises (`formatPost`), never the whole post.
    expect(result.posts).toEqual([{ id: 'p1', title: 'AI post', likes: 0 }]);
  });

  it('handles empty results from intent search', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'test-key', timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/intents/search': mockResponse(200, { results: [] }),
      '/api/discovery/agents': mockResponse(200, { agents: [] }),
    });

    const result = await skill.search({ query: 'nonexistent', types: ['intents', 'agents'] }, ctx);

    expect(result.intents).toEqual([]);
    expect(result.agents).toEqual([]);
  });

  it('handles empty results from content search', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'test-key', timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/discovery/channels': mockResponse(200, {}),
    });

    const result = await skill.search({ query: 'empty', types: ['channels'] }, ctx);

    expect(result.channels).toEqual([]);
  });

  it('handles non-ok status from intent search without throwing', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'test-key', timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/intents/search': mockResponse(500, { error: 'Internal Server Error' }),
    });

    const result = await skill.search({ query: 'fail', types: ['intents'] }, ctx);

    // Nothing came back and a call failed: the answer says which, where an
    // empty object read to the model as "nothing found".
    expect(result.intents).toBeUndefined();
    expect(result).toEqual({ error: 'Search failed: intents 500.' });
  });

  it('handles non-ok status from content search without throwing', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'test-key', timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/intents/search': mockResponse(200, { results: [] }),
      '/api/discovery/posts': mockResponse(403, { error: 'Forbidden' }),
    });

    const result = await skill.search({ query: 'mixed', types: ['intents', 'posts'] }, ctx);

    expect(result.intents).toEqual([]);
    expect(result.posts).toBeUndefined();
  });

  it('returns results from /api/discovery/agents directly (not from intent dedup)', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'test-key', timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/intents/search': mockResponse(200, {
        results: [
          { intent: 'draw', agentId: 'agent-a', score: 0.95 },
          { intent: 'paint', agentId: 'agent-a', score: 0.8 },
        ],
      }),
      '/api/discovery/agents': mockResponse(200, {
        agents: [
          { id: 'agent-a', username: 'artist', displayName: 'Artist' },
          { id: 'agent-b', username: 'painter', displayName: 'Painter' },
        ],
      }),
    });

    const result = await skill.search({ query: 'art', types: ['intents', 'agents'] }, ctx);

    expect(result.intents).toHaveLength(2);
    expect(result.agents).toHaveLength(2);
  });

  it('resolves data[type] key from content search response', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'test-key', timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/discovery/users': mockResponse(200, { users: [{ id: 'u1', name: 'Alice' }] }),
    });

    const result = await skill.search({ query: 'alice', types: ['users'] }, ctx);

    expect(result.users).toEqual([{ id: 'u1', name: 'Alice' }]);
  });

  it('defaults types to ["intents","agents"] when not provided', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'test-key', timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/intents/search': mockResponse(200, { results: [{ intent: 'default', agentId: 'x', score: 1 }] }),
      '/api/discovery/agents': mockResponse(200, { agents: [{ id: 'x', username: 'agent-x' }] }),
    });

    const result = await skill.search({ query: 'test' }, ctx);

    const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls;
    expect(calls.some((c: any) => c[0].includes('/api/intents/search'))).toBe(true);
    expect(calls.some((c: any) => c[0].includes('/api/discovery/agents'))).toBe(true);
    expect(result.intents).toHaveLength(1);
    expect(result.agents).toHaveLength(1);
  });

  it('uses custom limit parameter for intent and content search', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'test-key', timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/intents/search': mockResponse(200, { results: [] }),
      '/api/discovery/posts': mockResponse(200, { posts: [] }),
    });

    await skill.search({ query: 'custom limit', types: ['intents', 'posts'], limit: 25 }, ctx);

    const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls;
    const intentCall = calls.find((c: any) => c[0].includes('/api/intents/search'));
    expect(intentCall![1].body).toContain('"limit":25');

    const postCall = calls.find((c: any) => c[0].includes('/api/discovery/posts'));
    expect(postCall![0]).toContain('limit=25');
  });

  it('defaults limit to 10 when not provided', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'test-key', timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/intents/search': mockResponse(200, { results: [] }),
    });

    await skill.search({ query: 'no limit', types: ['intents'] }, ctx);

    const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls;
    const intentCall = calls.find((c: any) => c[0].includes('/api/intents/search'));
    expect(intentCall![1].body).toContain('"limit":10');
  });

  it('runs all type fetches in parallel', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'test-key', timeout: 5000 });
    const ctx = createMockContext();

    const order: string[] = [];
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.includes('/api/intents/search')) {
        order.push('intents_start');
        await new Promise((r) => setTimeout(r, 10));
        order.push('intents_end');
        return mockResponse(200, { results: [] });
      }
      if (url.includes('/api/discovery/agents')) {
        order.push('agents_start');
        await new Promise((r) => setTimeout(r, 10));
        order.push('agents_end');
        return mockResponse(200, { agents: [] });
      }
      if (url.includes('/api/discovery/posts')) {
        order.push('posts_start');
        await new Promise((r) => setTimeout(r, 10));
        order.push('posts_end');
        return mockResponse(200, { posts: [] });
      }
      return mockResponse(404, {});
    }) as any;

    await skill.search({ query: 'parallel', types: ['intents', 'agents', 'posts'] }, ctx);

    // All should start before any ends (parallel execution)
    const allStarts = order.filter((e) => e.endsWith('_start'));
    const firstEnd = order.findIndex((e) => e.endsWith('_end'));
    expect(allStarts.length).toBe(3);
    expect(firstEnd).toBeGreaterThanOrEqual(allStarts.length);
  });
});

describe('the search tool both SDKs share (2026-09-25)', () => {
  it('offers the definition in the shared fixture', () => {
    const agent = new BaseAgent({
      name: 'd', instructions: 'x',
      skills: [new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'k' }) as any],
    });
    const def = agent.getToolDefinitions().find((d) => d.function.name === 'search');
    expect(def).toEqual(FIXTURE.definition);
  });

  it('refuses without a credential in the shared sentence', () => {
    expect(NO_DISCOVERY_CREDENTIAL).toBe(FIXTURE.no_credential);
    expect(NO_DISCOVERY_SIGN_IN).toBe(FIXTURE.no_sign_in);
  });

  it('answers in the order the types were asked for, whatever order the platform answers in', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'k' });
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      // The first type asked for answers last.
      if (url.includes('/api/discovery/channels')) {
        await new Promise((r) => setTimeout(r, 15));
        return mockResponse(200, { channels: [{ slug: 'c' }] });
      }
      if (url.includes('/api/discovery/tags')) return mockResponse(200, { tags: [{ name: 't' }] });
      return mockResponse(404, {});
    }) as any;
    const result = await skill.search({ query: 'x', types: ['channels', 'tags'] });
    expect(Object.keys(result)).toEqual(['channels', 'tags']);
  });

  it('says which calls failed when nothing came back', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'k' });
    globalThis.fetch = routedFetch({
      '/api/intents/search': mockResponse(401, { error: 'Unauthorized' }),
      '/api/discovery/agents': mockResponse(401, { error: 'Unauthorized' }),
    });
    expect(await skill.search({ query: 'x', types: ['intents', 'agents'] })).toEqual({
      error: 'Search failed: intents 401, agents 401.',
    });
  });

  it('cuts a post to an excerpt with its author, channel and likes', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'k' });
    globalThis.fetch = routedFetch({
      '/api/discovery/posts': mockResponse(200, {
        posts: [{
          id: 'p1', title: 'T', content: 'x'.repeat(1000), humanLikes: 2, agentLikes: 3,
          author: { username: 'alice', avatarUrl: 'https://img' }, channel: { slug: 'news', name: 'News' },
        }],
      }),
    });
    const result = await skill.search({ query: 'x', types: ['posts'] });
    expect(result.posts).toEqual([{ id: 'p1', title: 'T', content: 'x'.repeat(300), author: 'alice', channel: 'news', likes: 5 }]);
  });

  it("gives an agent result the agent's URL", async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'k' });
    globalThis.fetch = routedFetch({
      '/api/discovery/agents': mockResponse(200, {
        agents: [{ username: 'jurist', displayName: 'Jurist', agentUrl: 'https://legal.example.com/jurist', reputationScore: 12 }],
      }),
    });
    const result = await skill.search({ query: 'x', types: ['agents'] });
    expect(result.agents).toEqual([{
      username: 'jurist', display_name: 'Jurist', url: 'https://legal.example.com/jurist', reputation: 12, trust_level: 'standard',
    }]);
  });

  it('prints nothing of its own: progress goes to the agent trace, without the query', async () => {
    const lines: string[] = [];
    setAgentTrace({ enabled: true, sink: (line) => lines.push(line) });
    const log = vi.spyOn(console, 'log').mockImplementation(() => {});
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, apiKey: 'k' });
      globalThis.fetch = routedFetch({ '/api/intents/search': mockResponse(500, {}) });
      await skill.search({ query: 'private words', types: ['intents', 'agents'] });
      expect(log).not.toHaveBeenCalled();
      expect(error).not.toHaveBeenCalled();
      expect(lines.some((line) => line.startsWith('[search] intents 500 in '))).toBe(true);
      expect(lines.join('\n')).not.toContain('private words');
    } finally {
      setAgentTrace({ enabled: true, sink: (line) => console.log(line) });
      log.mockRestore();
      error.mockRestore();
    }
  });
});

describe('the platform the search goes to (2026-09-25)', () => {
  const saved = { ...process.env };
  afterEach(() => {
    process.env = { ...saved };
    cliPlatform.answer = undefined;
  });

  function clearPlatformEnv(): void {
    for (const name of ['ROBUTLER_API_URL', 'ROBUTLER_INTERNAL_API_URL', 'WEBAGENTS_PROFILE']) delete process.env[name];
  }

  it('is the configured URL first, then ROBUTLER_API_URL, then ROBUTLER_INTERNAL_API_URL', async () => {
    clearPlatformEnv();
    process.env.ROBUTLER_API_URL = 'https://api.example.com/';
    process.env.ROBUTLER_INTERNAL_API_URL = 'http://internal:3000';
    expect(await new PortalDiscoverySkill({ portalUrl: 'https://mine.example.com/' }).platformUrl()).toBe('https://mine.example.com');
    expect(await new PortalDiscoverySkill().platformUrl()).toBe('https://api.example.com');
    delete process.env.ROBUTLER_API_URL;
    expect(await new PortalDiscoverySkill().platformUrl()).toBe('http://internal:3000');
  });

  it('is the portal the CLI is pointed at (`platform.url`, `webagents login`), then https://robutler.ai', async () => {
    clearPlatformEnv();
    cliPlatform.answer = () => ['https://macbook.example.ts.net/', 'global'];
    expect(await new PortalDiscoverySkill().platformUrl()).toBe('https://macbook.example.ts.net');
    cliPlatform.answer = () => {
      throw new Error('no CLI configuration here');
    };
    expect(await new PortalDiscoverySkill().platformUrl()).toBe('https://robutler.ai');
  });
});
