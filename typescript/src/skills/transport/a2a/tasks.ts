/**
 * The A2A task store: every task a served agent has answered or is working
 * on, per caller, with a TTL (plan item 1.3, 2026-09-26).
 *
 * PER CALLER, AS A RULE OF THE STORE. A caller reads, lists, subscribes to and
 * cancels only the tasks it created. The old store was one map keyed by task
 * id, so any caller who learned an id (ids are UUIDs, but they travel in logs
 * and in the reply of the caller that made them) could read another caller's
 * conversation with this agent, or cancel it. The owner key is part of every
 * lookup, and a miss for the wrong owner is a miss, never a 403: a 403 would
 * confirm the id exists.
 *
 * Events are kept per task so `SubscribeToTask` can replay what a streaming
 * caller missed before attaching, and so a client that disconnected can
 * resume; they go with the task when it expires.
 */

import { isSettled, type Artifact, type Message, type StreamResponse, type Task, type TaskState, type TaskStatus } from './types';

export interface TaskRecord {
  owner: string;
  task: Task;
  /** Every stream event so far, in order. */
  events: StreamResponse[];
  listeners: Set<(event: StreamResponse) => void>;
  createdAt: number;
  expiresAt: number;
  /** Cancels the run in flight, when there is one. */
  abort?: () => void;
  /** Resolved when the task reaches a settled state. */
  settled: Promise<void>;
  markSettled: () => void;
}

export interface ListFilter {
  contextId?: string;
  status?: TaskState;
  pageSize?: number;
  pageToken?: string;
  historyLength?: number;
  includeArtifacts?: boolean;
}

export const DEFAULT_PAGE_SIZE = 50;
export const MAX_PAGE_SIZE = 200;

export class TaskStore {
  private readonly records = new Map<string, TaskRecord>();

  constructor(private readonly ttlMs: number, private readonly now: () => number = Date.now) {}

  get size(): number {
    this.sweep();
    return this.records.size;
  }

  create(owner: string, task: Task): TaskRecord {
    this.sweep();
    let markSettled!: () => void;
    const settled = new Promise<void>((resolve) => {
      markSettled = resolve;
    });
    const record: TaskRecord = {
      owner,
      task,
      events: [],
      listeners: new Set(),
      createdAt: this.now(),
      expiresAt: this.now() + this.ttlMs,
      settled,
      markSettled,
    };
    this.records.set(task.id, record);
    return record;
  }

  /** The record for `id` when `owner` created it; undefined otherwise. */
  get(owner: string, id: string): TaskRecord | undefined {
    this.sweep();
    const record = this.records.get(id);
    return record && record.owner === owner ? record : undefined;
  }

  /** `owner`'s tasks, newest first, filtered and paged. */
  list(owner: string, filter: ListFilter = {}): { tasks: Task[]; nextPageToken: string; pageSize: number; totalSize: number } {
    this.sweep();
    const pageSize = Math.max(1, Math.min(filter.pageSize ?? DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE));
    const matching = [...this.records.values()]
      .filter((r) => r.owner === owner)
      .filter((r) => !filter.contextId || r.task.contextId === filter.contextId)
      .filter((r) => !filter.status || r.task.status.state === filter.status)
      .sort((a, b) => b.createdAt - a.createdAt);
    const offset = filter.pageToken ? Number.parseInt(filter.pageToken, 10) || 0 : 0;
    const page = matching.slice(offset, offset + pageSize);
    const next = offset + pageSize < matching.length ? String(offset + pageSize) : '';
    return {
      tasks: page.map((r) => taskView(r.task, { historyLength: filter.historyLength, includeArtifacts: filter.includeArtifacts ?? true })),
      nextPageToken: next,
      pageSize,
      totalSize: matching.length,
    };
  }

  /** Update the status, record the event, wake listeners; settles the record at a terminal or interrupted state. */
  setStatus(record: TaskRecord, status: TaskStatus, metadata?: Record<string, unknown>): void {
    record.task.status = status;
    if (metadata) record.task.metadata = { ...(record.task.metadata ?? {}), ...metadata };
    this.emit(record, { statusUpdate: { taskId: record.task.id, contextId: record.task.contextId, status } });
    if (isSettled(status.state)) {
      record.abort = undefined;
      record.markSettled();
    }
  }

  /** Append an artifact chunk and record the event. */
  addArtifactChunk(record: TaskRecord, artifact: Artifact, append: boolean, lastChunk: boolean): void {
    const existing = record.task.artifacts.find((a) => a.artifactId === artifact.artifactId);
    if (existing && append) existing.parts.push(...artifact.parts);
    else if (existing) existing.parts = [...artifact.parts];
    else record.task.artifacts.push({ ...artifact, parts: [...artifact.parts] });
    this.emit(record, {
      artifactUpdate: { taskId: record.task.id, contextId: record.task.contextId, artifact, append, lastChunk },
    });
  }

  addHistory(record: TaskRecord, message: Message): void {
    record.task.history.push(message);
  }

  emit(record: TaskRecord, event: StreamResponse): void {
    record.events.push(event);
    for (const listener of record.listeners) {
      try {
        listener(event);
      } catch {
        // a listener that throws does not stop the others
      }
    }
  }

  /** Drop every record past its TTL. */
  sweep(): void {
    const now = this.now();
    for (const [id, record] of this.records) {
      if (record.expiresAt <= now) {
        record.listeners.clear();
        this.records.delete(id);
      }
    }
  }

  clear(): void {
    for (const record of this.records.values()) record.listeners.clear();
    this.records.clear();
  }
}

/** A task as a response carries it: history trimmed to `historyLength` when asked, artifacts optional. */
export function taskView(task: Task, options: { historyLength?: number; includeArtifacts?: boolean } = {}): Task {
  const history =
    options.historyLength !== undefined && options.historyLength >= 0 ? task.history.slice(-options.historyLength || task.history.length) : task.history;
  return {
    id: task.id,
    contextId: task.contextId,
    status: task.status,
    artifacts: options.includeArtifacts === false ? [] : task.artifacts,
    history: options.historyLength === 0 ? [] : history,
    ...(task.metadata ? { metadata: task.metadata } : {}),
  };
}
