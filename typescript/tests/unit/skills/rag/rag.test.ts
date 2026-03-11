import { describe, it, expect, beforeEach } from 'vitest';
import { RAGSkill } from '../../../../src/skills/rag/skill.js';
import { ContextImpl } from '../../../../src/core/context.js';

describe('RAGSkill (in-memory backend)', () => {
  let skill: RAGSkill;
  let ctx: ContextImpl;

  beforeEach(() => {
    skill = new RAGSkill({ backend: 'memory', chunkSize: 50, chunkOverlap: 10, topK: 3 });
    ctx = new ContextImpl();
  });

  it('should ingest and chunk text', async () => {
    const result = await (skill as any).ragIngest({
      text: 'A'.repeat(120),
      source: 'test.txt',
    }, ctx);
    expect(result.chunks).toBeGreaterThan(1);
  });

  it('should search and return results', async () => {
    await (skill as any).ragIngest({
      text: 'The quick brown fox jumps over the lazy dog',
      source: 'animals.txt',
    }, ctx);
    await (skill as any).ragIngest({
      text: 'TypeScript is a typed superset of JavaScript',
      source: 'programming.txt',
    }, ctx);

    const results = await (skill as any).ragSearch({ query: 'fox animal' }, ctx);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]).toHaveProperty('text');
    expect(results[0]).toHaveProperty('score');
  });

  it('should delete by source', async () => {
    await (skill as any).ragIngest({ text: 'hello world', source: 'greet.txt' }, ctx);
    const before = await (skill as any).ragStats({}, ctx);
    expect(before.totalChunks).toBeGreaterThan(0);

    await (skill as any).ragDelete({ source: 'greet.txt' }, ctx);
    const after = await (skill as any).ragStats({}, ctx);
    expect(after.totalChunks).toBe(0);
  });

  it('should report stats', async () => {
    await (skill as any).ragIngest({ text: 'data1', source: 'a.txt' }, ctx);
    await (skill as any).ragIngest({ text: 'data2', source: 'b.txt' }, ctx);
    const stats = await (skill as any).ragStats({}, ctx);
    expect(stats.sources).toContain('a.txt');
    expect(stats.sources).toContain('b.txt');
  });
});
