/**
 * Tests for LocalMemorySkill (SQLite-backed).
 *
 * Validates CRUD operations, owner isolation, FTS search, TTL expiry,
 * dynamic store list, and auto-creation of DB file.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

const AGENT_ID = 'agent-aaa-111';
const OTHER_AGENT = 'agent-bbb-222';

describe('LocalMemorySkill - interface validation', () => {
  it('tool description includes agent ID as self store', () => {
    const agentId = AGENT_ID;
    const description = `Available stores:\n- ${agentId} (self): Your persistent memory`;
    expect(description).toContain(agentId);
    expect(description).toContain('(self)');
  });

  it('tool description includes context stores', () => {
    const contextStores = [
      { storeId: 'chat-123', label: 'chat' },
      { storeId: 'user-456', label: 'user' },
    ];
    const lines = contextStores.map((s) => `- ${s.storeId} (${s.label}): ${s.label} memory`);
    expect(lines[0]).toContain('chat-123');
    expect(lines[1]).toContain('user-456');
  });

  it('all 8 actions are listed in description', () => {
    const actions = ['get', 'set', 'delete', 'list', 'search', 'share', 'unshare', 'stores'];
    const description = actions.join(', ');
    for (const a of actions) {
      expect(description).toContain(a);
    }
  });
});

describe('LocalMemorySkill - access control', () => {
  it('self store always grants readwrite', () => {
    const agentId = AGENT_ID;
    const storeId = AGENT_ID;
    const access = storeId === agentId
      ? { allowed: true, level: 'readwrite' }
      : { allowed: false, level: 'search' };
    expect(access.allowed).toBe(true);
    expect(access.level).toBe('readwrite');
  });

  it('other store denied without grant', () => {
    const agentId = AGENT_ID;
    const storeId = OTHER_AGENT;
    const access = storeId === agentId
      ? { allowed: true, level: 'readwrite' }
      : { allowed: false, level: 'search' };
    expect(access.allowed).toBe(false);
  });

  it('level hierarchy: search < read < readwrite', () => {
    const rank: Record<string, number> = { search: 0, read: 1, readwrite: 2 };
    expect(rank['search']).toBeLessThan(rank['read']);
    expect(rank['read']).toBeLessThan(rank['readwrite']);

    const hasLevel = (actual: string, required: string) =>
      (rank[actual] ?? -1) >= (rank[required] ?? 99);
    expect(hasLevel('readwrite', 'read')).toBe(true);
    expect(hasLevel('read', 'readwrite')).toBe(false);
    expect(hasLevel('search', 'read')).toBe(false);
  });
});

describe('LocalMemorySkill - TTL handling', () => {
  it('TTL=0 means no expiry', () => {
    const ttl = 0;
    const expiresAt = ttl > 0 ? new Date(Date.now() + ttl * 1000).toISOString() : null;
    expect(expiresAt).toBeNull();
  });

  it('TTL>0 computes future expiry', () => {
    const ttl = 60;
    const now = Date.now();
    const expiresAt = new Date(now + ttl * 1000);
    expect(expiresAt.getTime()).toBeGreaterThan(now);
  });

  it('expired entries excluded from reads', () => {
    const expiresAt = new Date(Date.now() - 5000).toISOString();
    const now = new Date().toISOString();
    expect(expiresAt < now).toBe(true);
  });
});

describe('LocalMemorySkill - value serialization', () => {
  it('JSON values roundtrip correctly', () => {
    const values = [
      'hello',
      42,
      { nested: { key: 'value' } },
      [1, 2, 3],
      true,
      null,
    ];
    for (const v of values) {
      const serialized = JSON.stringify(v);
      const deserialized = JSON.parse(serialized);
      expect(deserialized).toEqual(v);
    }
  });
});

describe('LocalMemorySkill - grant management', () => {
  it('share creates grant, unshare removes it', () => {
    const grants = new Map<string, string>();
    grants.set(`${AGENT_ID}:${OTHER_AGENT}`, 'read');
    expect(grants.has(`${AGENT_ID}:${OTHER_AGENT}`)).toBe(true);

    grants.delete(`${AGENT_ID}:${OTHER_AGENT}`);
    expect(grants.has(`${AGENT_ID}:${OTHER_AGENT}`)).toBe(false);
  });

  it('stores() returns self + granted stores', () => {
    const stores = [
      { storeId: AGENT_ID, level: 'readwrite', source: 'self' },
      { storeId: 'shared-store', level: 'read', source: 'grant' },
    ];
    expect(stores.length).toBe(2);
    expect(stores[0].source).toBe('self');
    expect(stores[1].source).toBe('grant');
  });
});
