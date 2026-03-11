/**
 * Todo Skill
 *
 * Task management for agents. Provides structured task tracking
 * with status, priority, and dependencies. Persists to a JSON
 * file in the working directory.
 */

import { Skill } from '../../core/skill.js';
import { tool } from '../../core/decorators.js';
import type { Context } from '../../core/types.js';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

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
  private filePath: string;
  private items: TodoItem[] = [];
  private loaded = false;

  constructor(config: TodoConfig = {}) {
    super({ ...config, name: config.name || 'todo' });
    this.filePath = config.filePath ?? path.join(process.cwd(), '.webagents', 'todos.json');
  }

  private async load(): Promise<void> {
    if (this.loaded) return;
    try {
      const raw = await fs.readFile(this.filePath, 'utf-8');
      this.items = JSON.parse(raw);
    } catch {
      this.items = [];
    }
    this.loaded = true;
  }

  private async save(): Promise<void> {
    await fs.mkdir(path.dirname(this.filePath), { recursive: true });
    await fs.writeFile(this.filePath, JSON.stringify(this.items, null, 2));
  }

  private nextId(): string {
    const max = this.items.reduce((m, i) => {
      const n = parseInt(i.id.replace('todo-', ''), 10);
      return isNaN(n) ? m : Math.max(m, n);
    }, 0);
    return `todo-${max + 1}`;
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
    _context: Context,
  ): Promise<TodoItem> {
    await this.load();
    const now = new Date().toISOString();
    const item: TodoItem = {
      id: this.nextId(),
      content: params.content,
      status: 'pending',
      priority: params.priority ?? 'medium',
      tags: params.tags ?? [],
      dependsOn: params.depends_on ?? [],
      createdAt: now,
      updatedAt: now,
    };
    this.items.push(item);
    await this.save();
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
    _context: Context,
  ): Promise<TodoItem[]> {
    await this.load();
    let result = [...this.items];
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
    _context: Context,
  ): Promise<TodoItem | string> {
    await this.load();
    const item = this.items.find((i) => i.id === params.id);
    if (!item) return `Todo ${params.id} not found`;

    if (params.status) item.status = params.status;
    if (params.content) item.content = params.content;
    if (params.priority) item.priority = params.priority;
    if (params.tags) item.tags = params.tags;
    item.updatedAt = new Date().toISOString();
    if (params.status === 'completed') item.completedAt = item.updatedAt;

    await this.save();
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
  async todoDelete(params: { id: string }, _context: Context): Promise<string> {
    await this.load();
    const idx = this.items.findIndex((i) => i.id === params.id);
    if (idx === -1) return `Todo ${params.id} not found`;
    this.items.splice(idx, 1);
    await this.save();
    return 'OK';
  }
}
