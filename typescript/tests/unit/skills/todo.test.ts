/**
 * The todo skill (2026-09-25): the definitions and guide the Python skill
 * shares (`python/tests/fixtures/todo_tool/definitions.json`; Python:
 * `python/tests/agents/skills/test_todo_skill.py`), and the same behavior.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { TodoSkill } from '../../../src/skills/todo/skill';
import { resolveSkillsByName } from '../../../src/skills/resolve';
import { tempDirs } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/todo_tool/definitions.json'), 'utf8'),
) as { definitions: Array<{ function: { name: string } }>; prompt: string };

describe('the todo skill', () => {
  it('offers the definitions and guide both SDKs share', () => {
    const agent = new BaseAgent({ name: 't', instructions: 'x', skills: [new TodoSkill()] });
    const defs = agent.getToolDefinitions().filter((d) => d.function.name.startsWith('todo_'));
    expect(defs).toEqual(FIXTURE.definitions);
    const guide = (new TodoSkill() as unknown as { prompts: Array<{ handler: (c: unknown) => string }> }).prompts[0];
    expect(guide.handler({})).toBe(FIXTURE.prompt);
  });

  it('is named `todo` in an agent file, and keeps its list in the agent folder', async () => {
    const dir = tempDir('todo-');
    const { skills, unknown } = await resolveSkillsByName(['todo'], { agentDir: dir });
    expect(unknown).toEqual([]);
    const skill = skills[0] as unknown as TodoSkill;
    const first = await skill.todoAdd({ content: 'Write the parser', priority: 'high', tags: ['core'] }, {} as never);
    const second = await skill.todoAdd({ content: 'Test it' }, {} as never);
    expect([first.id, second.id]).toEqual(['todo-1', 'todo-2']);
    expect(Object.keys(first)).toEqual(['id', 'content', 'status', 'priority', 'tags', 'dependsOn', 'createdAt', 'updatedAt']);
    const done = await skill.todoUpdate({ id: 'todo-1', status: 'completed' }, {} as never);
    expect(typeof done === 'object' && done.completedAt === done.updatedAt).toBe(true);
    expect(await skill.todoDelete({ id: 'todo-9' }, {} as never)).toBe('Todo todo-9 not found');
    const saved = JSON.parse(readFileSync(path.join(dir, '.webagents', 'todos.json'), 'utf8')) as Array<{ id: string }>;
    expect(saved.map((i) => i.id)).toEqual(['todo-1', 'todo-2']);
  });
});
