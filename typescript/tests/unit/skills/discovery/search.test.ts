/**
 * Unit tests for PortalDiscoverySkill.search tool
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { PortalDiscoverySkill } from '../../../../src/skills/discovery/skill.js';

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
    expect(result.posts).toEqual([{ id: 'p1', title: 'AI post' }]);
  });

  it('handles empty results from intent search', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, timeout: 5000 });
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
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/discovery/channels': mockResponse(200, {}),
    });

    const result = await skill.search({ query: 'empty', types: ['channels'] }, ctx);

    expect(result.channels).toEqual([]);
  });

  it('handles non-ok status from intent search without throwing', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/intents/search': mockResponse(500, { error: 'Internal Server Error' }),
    });

    const result = await skill.search({ query: 'fail', types: ['intents'] }, ctx);

    expect(result.intents).toBeUndefined();
    expect(result).toEqual({});
  });

  it('handles non-ok status from content search without throwing', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, timeout: 5000 });
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
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, timeout: 5000 });
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
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, timeout: 5000 });
    const ctx = createMockContext();

    globalThis.fetch = routedFetch({
      '/api/discovery/users': mockResponse(200, { users: [{ id: 'u1', name: 'Alice' }] }),
    });

    const result = await skill.search({ query: 'alice', types: ['users'] }, ctx);

    expect(result.users).toEqual([{ id: 'u1', name: 'Alice' }]);
  });

  it('defaults types to ["intents","agents"] when not provided', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, timeout: 5000 });
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
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, timeout: 5000 });
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
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, timeout: 5000 });
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
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL_URL, timeout: 5000 });
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
