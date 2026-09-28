/**
 * S-294 (2026-09-26): the todo list is the CALLER's. It was one file for
 * everyone, so a stranger with an arbitrary bearer over `mcp serve --http`
 * listed the todo the owner had made over stdio. Whose list a turn sees now
 * follows the memory skill's namespace rule, pinned by `caller_scope` in
 * `python/tests/fixtures/todo_tool/definitions.json`, which the Python suite
 * runs too (`tests/agents/skills/test_todo_caller_scope_s294_e2efix.py`):
 * two verified callers and the owner each keep their own list, a caller
 * nothing verified gets the refusal from every tool, and the ACP plan reads
 * the owner's list.
 */

import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { TODO_NO_CALLER, TodoSkill } from '../../../src/skills/todo/skill';
import { callerKey, namespaceOf } from '../../../src/skills/memory/namespace';
import type { AuthInfo, Context } from '../../../src/core/types';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SCOPE = (JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/todo_tool/definitions.json'), 'utf8')) as { caller_scope: Scope }).caller_scope;

interface Scope {
  owner_file: string;
  caller_file: string;
  refusal: string;
  callers: { case: string; auth: Record<string, unknown> | null; namespace: string | null; caller_key?: string }[];
}

const tempDir = tempDirs();
const turn = (auth: Record<string, unknown> | null): Context => ({ auth: auth ?? undefined } as unknown as Context);
const byCase = (name: string) => SCOPE.callers.find((c) => c.case === name)!;

describe('S-294: the todo list is the caller’s', () => {
  it('the refusal and the namespace of every fixture caller match', () => {
    expect(TODO_NO_CALLER).toBe(SCOPE.refusal);
    for (const c of SCOPE.callers) {
      expect(namespaceOf(c.auth as Partial<AuthInfo> | null), c.case).toBe(c.namespace);
      if (c.caller_key) expect(callerKey(c.namespace!.slice('caller:'.length))).toBe(c.caller_key);
    }
  });

  it('two verified callers and the owner each keep their own list, in their own file', async () => {
    const dir = tempDir('todo-scope-');
    const skill = new TodoSkill({ filePath: path.join(dir, SCOPE.owner_file) });
    const owner = turn(byCase('the owner').auth);
    const alice = byCase("a platform credential's user");
    const bob = byCase('another platform user');

    await skill.todoAdd({ content: 'owner: pay the rent' }, owner);
    const aliceFirst = (await skill.todoAdd({ content: 'alice: buy milk' }, turn(alice.auth))) as { id: string };
    await skill.todoAdd({ content: 'bob: call mum' }, turn(bob.auth));

    const contents = async (who: Context) => ((await skill.todoList({}, who)) as { content: string }[]).map((i) => i.content);
    expect(await contents(owner)).toEqual(['owner: pay the rent']);
    expect(await contents(turn(alice.auth))).toEqual(['alice: buy milk']);
    expect(await contents(turn(bob.auth))).toEqual(['bob: call mum']);
    // Ids are per list: alice's first is todo-1 too.
    expect(aliceFirst.id).toBe('todo-1');
    // Bob cannot reach alice's item by id, and cannot touch the owner's.
    expect(await skill.todoDelete({ id: 'todo-1' }, turn(bob.auth))).toBe('OK');
    expect(await contents(turn(alice.auth))).toEqual(['alice: buy milk']);
    expect(await contents(owner)).toEqual(['owner: pay the rent']);

    expect(existsSync(path.join(dir, SCOPE.owner_file))).toBe(true);
    const aliceFile = path.join(dir, SCOPE.caller_file.replace('{caller_key}', alice.caller_key!));
    expect(JSON.parse(readFileSync(aliceFile, 'utf8')).map((i: { content: string }) => i.content)).toEqual(['alice: buy milk']);
    expect(JSON.parse(readFileSync(path.join(dir, SCOPE.owner_file), 'utf8'))).toHaveLength(1);
  });

  it('a caller nothing verified gets the refusal from every tool, and nothing is written', async () => {
    const dir = tempDir('todo-scope-');
    const skill = new TodoSkill({ filePath: path.join(dir, SCOPE.owner_file) });
    for (const name of ['a bearer nothing verified', 'anonymous']) {
      const who = turn(byCase(name).auth);
      expect(await skill.todoAdd({ content: 'x' }, who)).toEqual({ error: SCOPE.refusal });
      expect(await skill.todoList({}, who)).toEqual({ error: SCOPE.refusal });
      expect(await skill.todoUpdate({ id: 'todo-1', status: 'completed' }, who)).toEqual({ error: SCOPE.refusal });
      expect(await skill.todoDelete({ id: 'todo-1' }, who)).toEqual({ error: SCOPE.refusal });
    }
    expect(existsSync(path.join(dir, '.webagents'))).toBe(false);
  });

  it('the plan a transport shows is the owner’s list', async () => {
    const dir = tempDir('todo-scope-');
    const skill = new TodoSkill({ filePath: path.join(dir, SCOPE.owner_file) });
    await skill.todoAdd({ content: 'alice: buy milk' }, turn(byCase("a platform credential's user").auth));
    await skill.todoAdd({ content: 'owner: pay the rent' }, turn(byCase('the owner').auth));
    expect(skill.getItems().map((i) => i.content)).toEqual(['owner: pay the rent']);
    // A fresh skill over the same folder reads the owner's file for the plan.
    expect(new TodoSkill({ filePath: path.join(dir, SCOPE.owner_file) }).getItems().map((i) => i.content)).toEqual(['owner: pay the rent']);
  });
});
