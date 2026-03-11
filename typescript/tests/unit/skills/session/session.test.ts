import { describe, it, expect, beforeEach } from 'vitest';
import { SessionSkill } from '../../../../src/skills/session/skill.js';
import { ContextImpl } from '../../../../src/core/context.js';

describe('SessionSkill', () => {
  let skill: SessionSkill;
  let ctx: ContextImpl;

  beforeEach(() => {
    skill = new SessionSkill({ maxEntries: 10 });
    ctx = new ContextImpl();
    ctx.metadata = { sessionId: 'test-session' };
  });

  it('should set and get a value', async () => {
    await (skill as any).sessionSet({ key: 'name', value: 'Alice' }, ctx);
    const result = await (skill as any).sessionGet({ key: 'name' }, ctx);
    expect(result).toBe('Alice');
  });

  it('should return null for missing keys', async () => {
    const result = await (skill as any).sessionGet({ key: 'missing' }, ctx);
    expect(result).toBeNull();
  });

  it('should delete a key', async () => {
    await (skill as any).sessionSet({ key: 'k', value: 1 }, ctx);
    await (skill as any).sessionDelete({ key: 'k' }, ctx);
    const result = await (skill as any).sessionGet({ key: 'k' }, ctx);
    expect(result).toBeNull();
  });

  it('should list all keys', async () => {
    await (skill as any).sessionSet({ key: 'a', value: 1 }, ctx);
    await (skill as any).sessionSet({ key: 'b', value: 2 }, ctx);
    const keys = await (skill as any).sessionList({}, ctx);
    expect(keys).toContain('a');
    expect(keys).toContain('b');
  });

  it('should clear all entries', async () => {
    await (skill as any).sessionSet({ key: 'x', value: 1 }, ctx);
    await (skill as any).sessionClear({}, ctx);
    const keys = await (skill as any).sessionList({}, ctx);
    expect(keys).toHaveLength(0);
  });

  it('should evict oldest entry when max exceeded', async () => {
    for (let i = 0; i < 12; i++) {
      await (skill as any).sessionSet({ key: `k${i}`, value: i }, ctx);
    }
    const keys = await (skill as any).sessionList({}, ctx);
    expect(keys.length).toBeLessThanOrEqual(10);
  });

  it('should isolate sessions by ID', async () => {
    await (skill as any).sessionSet({ key: 'val', value: 'session1' }, ctx);

    const ctx2 = new ContextImpl();
    ctx2.metadata = { sessionId: 'other-session' };
    const result = await (skill as any).sessionGet({ key: 'val' }, ctx2);
    expect(result).toBeNull();
  });
});
