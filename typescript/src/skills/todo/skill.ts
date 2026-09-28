/**
 * Todo Skill
 *
 * Task management for agents. Provides structured task tracking
 * with status, priority, and dependencies. Persists to a JSON
 * file in the working directory.
 *
 * THE LIST IS THE CALLER'S (S-294, 2026-09-26). It was one file for
 * everyone: every credentialed caller of a served agent read and changed
 * the owner's todos, and in the e2e run a stranger with an arbitrary bearer
 * over `webagents mcp serve --http` listed the todo the owner had made over
 * stdio. Whose list a turn sees is now decided the way the memory skill
 * decides whose notes it reads (`memory/namespace.ts`, `namespaceOf`): the
 * owner keeps `.webagents/todos.json`; a verified caller (`user:`, `agent:`,
 * `key:`, `channel:`) has its own file under `.webagents/todos/callers/`,
 * named by the same caller key the memory and session skills use; a caller
 * nothing verified has no list, and every todo tool tells it so. Chosen over
 * owner-only tools because the list is a caller's working plan for the
 * multi-step work it asked for, which is as useful to a verified caller as
 * to the owner, and because one identity rule for memory, sessions and
 * todos is one rule to get right. The Python skill applies the same rule
 * (`local/todo/skill.py`), pinned by `todo_tool/definitions.json`
 * (`caller_scope`).
 */

import { Skill } from '../../core/skill';
import { tool, prompt } from '../../core/decorators';
import type { AuthInfo, Context } from '../../core/types';
import { OWNER_NAMESPACE, localDirOf, namespaceOf } from '../memory/namespace';
import * as fs from 'node:fs/promises';
import * as fsSync from 'node:fs';
import * as path from 'node:path';

/** The refusal every todo tool answers a caller nothing verified (the fixture's `caller_scope.refusal`). */
export const TODO_NO_CALLER = 'todo: nothing is kept for a caller nothing verified.';

/** A list file's text, or null when there is none. */
function readOrNull(file: string): string | null {
  try {
    return fsSync.readFileSync(file, 'utf-8');
  } catch {
    return null;
  }
}

/** The items a list file holds; a missing or unreadable file is an empty list, as it always was. */
function parseList(raw: string | null): TodoItem[] {
  if (raw === null) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as TodoItem[]) : [];
  } catch {
    return [];
  }
}

export interface TodoConfig {
  name?: string;
  enabled?: boolean;
  /** Path to the todo file (default: .webagents/todos.json) */
  filePath?: string;
}

export type TodoStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled';
export type TodoPriority = 'low' | 'medium' | 'high' | 'critical';

export interface TodoItem {
  id: string;
  content: string;
  status: TodoStatus;
  priority: TodoPriority;
  tags: string[];
  dependsOn: string[];
  createdAt: string;
  updatedAt: string;
  completedAt?: string;
}

export class TodoSkill extends Skill {
  /** Allowed in a `restricted` turn (S-030): a per-run scratch list, nothing persists past the turn. */
  static restrictedPostureDefault = 'allow' as const;

  /** The owner's file; a caller's sits beside it under `todos/callers/`. */
  private filePath: string;
  /** Each list that has been read, by namespace (file comment). */
  private lists: Map<string, TodoItem[]> = new Map();

  constructor(config: TodoConfig = {}) {
    super({ ...config, name: config.name || 'todo' });
    this.filePath = config.filePath ?? path.join(process.cwd(), '.webagents', 'todos.json');
  }

  /**
   * The OWNER's list as it stands, for a transport that shows it as a plan
   * (the ACP `plan` update, 2026-09-26; the editor's user is the owner). A
   * copy: the list is this skill's to change.
   */
  getItems(): TodoItem[] {
    let items = this.lists.get(OWNER_NAMESPACE);
    if (!items) {
      items = parseList(readOrNull(this.filePath));
      this.lists.set(OWNER_NAMESPACE, items);
    }
    return items.map((item) => ({ ...item, tags: [...item.tags], dependsOn: [...item.dependsOn] }));
  }

  /** Whose list this turn is: `owner`, `caller:<principal>`, or null for a caller nothing verified. */
  private caller(context: Context | undefined): string | null {
    return namespaceOf(context?.auth as Partial<AuthInfo> | undefined);
  }

  /** Where a namespace's list lives: the owner's file, or `todos/callers/<key>.json` beside it. */
  private fileFor(namespace: string): string {
    if (namespace === OWNER_NAMESPACE) return this.filePath;
    return path.join(path.dirname(this.filePath), 'todos', `${localDirOf(namespace)}.json`);
  }

  private async load(namespace: string): Promise<TodoItem[]> {
    const cached = this.lists.get(namespace);
    if (cached) return cached;
    let raw: string | null = null;
    try {
      raw = await fs.readFile(this.fileFor(namespace), 'utf-8');
    } catch {
      raw = null;
    }
    const items = parseList(raw);
    this.lists.set(namespace, items);
    return items;
  }

  private async save(namespace: string, items: TodoItem[]): Promise<void> {
    const file = this.fileFor(namespace);
    await fs.mkdir(path.dirname(file), { recursive: true });
    await fs.writeFile(file, JSON.stringify(items, null, 2));
  }

  private nextId(items: TodoItem[]): string {
    const max = items.reduce((m, i) => {
      const n = parseInt(i.id.replace('todo-', ''), 10);
      return isNaN(n) ? m : Math.max(m, n);
    }, 0);
    return `todo-${max + 1}`;
  }

  @prompt({ priority: 50, name: 'todoGuide', scope: 'all' })
  todoGuide(_ctx: Context): string {
    return [
      '## Todo skill',
      '',
      'Use the `todo_*` tools for **multi-step work where tracking progress visibly helps the user** — refactors touching many files, multi-feature implementation plans, long-running investigations, anything where you\'d otherwise lose context across iterations or tool calls.',
      '',
      '### When to use',
      '- 3+ distinct steps with dependencies between them.',
      '- Work that spans many tool calls and the user benefits from seeing what\'s done vs pending.',
      '- After receiving new instructions mid-task — capture the new requirements as todos so nothing is dropped.',
      '- When the task is complex enough that you might forget a step.',
      '',
      '### When NOT to use',
      '- Single-step tasks (just do them).',
      '- Trivial tasks completable in 1-2 obvious tool calls.',
      '- Purely conversational requests.',
      '- Don\'t add a "test the change" todo unless the user explicitly asked — it shifts focus toward testing over implementation.',
      '',
      '### Discipline',
      '- Update status in real time. Mark `in_progress` when you start, `completed` IMMEDIATELY after finishing — not in batches.',
      '- Only ONE task `in_progress` at a time. Finish it before starting another.',
      '- Break complex tasks into specific, actionable items. Vague todos like "improve performance" are noise.',
      '- Persisted to `.webagents/todos.json` in the working directory; survives across runs.',
    ].join('\n');
  }

  @tool({
    name: 'todo_add',
    description: 'Add a new todo item.',
    parameters: {
      type: 'object',
      properties: {
        content: { type: 'string', description: 'Task description' },
        priority: { type: 'string', enum: ['low', 'medium', 'high', 'critical'], description: 'Priority (default: medium)' },
        tags: { type: 'array', items: { type: 'string' }, description: 'Optional tags' },
        depends_on: { type: 'array', items: { type: 'string' }, description: 'IDs of tasks this depends on' },
      },
      required: ['content'],
    },
  })
  async todoAdd(
    params: { content: string; priority?: TodoPriority; tags?: string[]; depends_on?: string[] },
    context: Context,
  ): Promise<TodoItem | { error: string }> {
    const namespace = this.caller(context);
    if (!namespace) return { error: TODO_NO_CALLER };
    const items = await this.load(namespace);
    const now = new Date().toISOString();
    const item: TodoItem = {
      id: this.nextId(items),
      content: params.content,
      status: 'pending',
      priority: params.priority ?? 'medium',
      tags: params.tags ?? [],
      dependsOn: params.depends_on ?? [],
      createdAt: now,
      updatedAt: now,
    };
    items.push(item);
    await this.save(namespace, items);
    return item;
  }

  @tool({
    name: 'todo_list',
    description: 'List todo items, optionally filtered by status or tag.',
    parameters: {
      type: 'object',
      properties: {
        status: { type: 'string', enum: ['pending', 'in_progress', 'completed', 'cancelled'] },
        tag: { type: 'string', description: 'Filter by tag' },
      },
    },
  })
  async todoList(
    params: { status?: TodoStatus; tag?: string },
    context: Context,
  ): Promise<TodoItem[] | { error: string }> {
    const namespace = this.caller(context);
    if (!namespace) return { error: TODO_NO_CALLER };
    let result = [...(await this.load(namespace))];
    if (params.status) result = result.filter((i) => i.status === params.status);
    if (params.tag) result = result.filter((i) => i.tags.includes(params.tag!));
    return result;
  }

  @tool({
    name: 'todo_update',
    description: 'Update a todo item (status, content, priority, tags).',
    parameters: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Todo ID' },
        status: { type: 'string', enum: ['pending', 'in_progress', 'completed', 'cancelled'] },
        content: { type: 'string' },
        priority: { type: 'string', enum: ['low', 'medium', 'high', 'critical'] },
        tags: { type: 'array', items: { type: 'string' } },
      },
      required: ['id'],
    },
  })
  async todoUpdate(
    params: { id: string; status?: TodoStatus; content?: string; priority?: TodoPriority; tags?: string[] },
    context: Context,
  ): Promise<TodoItem | string | { error: string }> {
    const namespace = this.caller(context);
    if (!namespace) return { error: TODO_NO_CALLER };
    const items = await this.load(namespace);
    const item = items.find((i) => i.id === params.id);
    if (!item) return `Todo ${params.id} not found`;

    if (params.status) item.status = params.status;
    if (params.content) item.content = params.content;
    if (params.priority) item.priority = params.priority;
    if (params.tags) item.tags = params.tags;
    item.updatedAt = new Date().toISOString();
    if (params.status === 'completed') item.completedAt = item.updatedAt;

    await this.save(namespace, items);
    return item;
  }

  @tool({
    name: 'todo_delete',
    description: 'Delete a todo item.',
    parameters: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Todo ID to delete' },
      },
      required: ['id'],
    },
  })
  async todoDelete(params: { id: string }, context: Context): Promise<string | { error: string }> {
    const namespace = this.caller(context);
    if (!namespace) return { error: TODO_NO_CALLER };
    const items = await this.load(namespace);
    const idx = items.findIndex((i) => i.id === params.id);
    if (idx === -1) return `Todo ${params.id} not found`;
    items.splice(idx, 1);
    await this.save(namespace, items);
    return 'OK';
  }
}
