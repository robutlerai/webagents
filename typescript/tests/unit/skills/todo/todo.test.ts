import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { TodoSkill } from '../../../../src/skills/todo/skill.js';
import { ContextImpl } from '../../../../src/core/context.js';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

describe('TodoSkill', () => {
  let skill: TodoSkill;
  let ctx: ContextImpl;
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'todo-test-'));
    skill = new TodoSkill({ filePath: path.join(tmpDir, 'todos.json') });
    ctx = new ContextImpl();
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('should add a todo', async () => {
    const item = await (skill as any).todoAdd({ content: 'Test task' }, ctx);
    expect(item.content).toBe('Test task');
    expect(item.status).toBe('pending');
    expect(item.id).toMatch(/^todo-/);
  });

  it('should list todos', async () => {
    await (skill as any).todoAdd({ content: 'Task 1' }, ctx);
    await (skill as any).todoAdd({ content: 'Task 2' }, ctx);
    const list = await (skill as any).todoList({}, ctx);
    expect(list).toHaveLength(2);
  });

  it('should filter by status', async () => {
    await (skill as any).todoAdd({ content: 'A' }, ctx);
    const item = await (skill as any).todoAdd({ content: 'B' }, ctx);
    await (skill as any).todoUpdate({ id: item.id, status: 'completed' }, ctx);

    const pending = await (skill as any).todoList({ status: 'pending' }, ctx);
    expect(pending).toHaveLength(1);
    expect(pending[0].content).toBe('A');
  });

  it('should update a todo', async () => {
    const item = await (skill as any).todoAdd({ content: 'Original' }, ctx);
    const updated = await (skill as any).todoUpdate({
      id: item.id,
      content: 'Updated',
      priority: 'high',
    }, ctx);
    expect(updated.content).toBe('Updated');
    expect(updated.priority).toBe('high');
  });

  it('should delete a todo', async () => {
    const item = await (skill as any).todoAdd({ content: 'Delete me' }, ctx);
    await (skill as any).todoDelete({ id: item.id }, ctx);
    const list = await (skill as any).todoList({}, ctx);
    expect(list).toHaveLength(0);
  });

  it('should persist across instances', async () => {
    const filePath = path.join(tmpDir, 'persist.json');
    const skill1 = new TodoSkill({ filePath });
    await (skill1 as any).todoAdd({ content: 'Persistent' }, ctx);

    const skill2 = new TodoSkill({ filePath });
    const list = await (skill2 as any).todoList({}, ctx);
    expect(list).toHaveLength(1);
    expect(list[0].content).toBe('Persistent');
  });
});
