/**
 * Model failover (2026-09-26, gap-closure plan item 2.8): the agent's model,
 * then the `fallback_models:` of its file, in order, when a call fails with
 * a provider error.
 *
 * One LLM skill standing in for a chain of them. Each turn asks the first
 * member; when it answers a 5xx, a 429, a 408 or never answers (no
 * connection, no host, a timeout), the next member is asked the same
 * question, and the transcript gets a note saying so BEFORE the next reply
 * (`{failed} did not answer ({reason}); trying {next}`). Any other failure
 * (a refused key, an unknown model, a bad request, no credits) is reported
 * as it is: those are not outages, and asking another model would hide the
 * thing to fix. A member that had already started answering is never
 * retried, because the reply would repeat.
 *
 * The note travels as a UAMP `progress` event (`stage: 'failover'`), which
 * `runStreaming` turns into a `note` chunk for the chat and for
 * `-p --output-format stream-json`; served agents see the event. The
 * Python SDK's `FailoverLLMSkill` (`llm/failover.py`) does the same with a
 * note chunk, and the words, the retryable set and the cases are pinned by
 * `python/tests/fixtures/w2ops/models.json` (`failover`).
 */

import { Skill } from '../../../core/skill';
import { handoff } from '../../../core/decorators';
import type { Context, Handoff, ISkill } from '../../../core/types';
import type { ClientEvent, ServerEvent } from '../../../uamp/events';
import { generateEventId } from '../../../uamp/events';

export const FAILOVER_SKILL_NAME = 'failover';
export const FAILOVER_STAGE = 'failover';

/** The HTTP statuses that mean the provider, not the request, failed. */
export const RETRYABLE_STATUSES: ReadonlySet<number> = new Set([408, 429, 500, 502, 503, 504]);

/** One model in the chain: its skill and its `provider/model` label. */
export interface FailoverMember {
  skill: ISkill;
  model: string;
}

export interface FailoverLLMSkillConfig {
  chain: FailoverMember[];
}

/** The note the transcript gets when the chain moves on (fixture `failover.note`). */
export function failoverNote(failed: string, reason: string, next: string): string {
  return `${failed} did not answer (${reason}); trying ${next}`;
}

/**
 * Whether a model call's failure is the provider's (retryable), and how the
 * note names it: `HTTP 503` for a status, `could not reach <origin>` for a
 * request that got no answer. `undefined` for a failure that is the
 * request's own.
 */
export function providerFailure(error: { code?: string; message?: string } | undefined): { reason: string } | undefined {
  const message = error?.message ?? '';
  const status = /\b(?:returned|status|HTTP)\s+(\d{3})\b/i.exec(message);
  if (status) {
    const code = Number(status[1]);
    return RETRYABLE_STATUSES.has(code) ? { reason: `HTTP ${code}` } : undefined;
  }
  const unreachable = /could not reach (https?:\/\/[^\s/:]+(?::\d+)?|[^\s:]+)/i.exec(message);
  if (unreachable) return { reason: `could not reach ${unreachable[1]}` };
  if (/fetch failed|ECONNREFUSED|ENOTFOUND|ETIMEDOUT|ECONNRESET|EAI_AGAIN|EPIPE|socket hang up|network error|timed out/i.test(message)) {
    const host = /(https?:\/\/[^\s/]+)/i.exec(message);
    return { reason: host ? `could not reach ${host[1]}` : 'could not reach the provider' };
  }
  return undefined;
}

export class FailoverLLMSkill extends Skill {
  readonly chain: FailoverMember[];
  /** The model that answered the last call (`provider/model`), for the chat's cost line. */
  answeredModel: string | undefined;
  /** The notes said during the last call, in order. */
  notes: string[] = [];

  constructor(config: FailoverLLMSkillConfig) {
    super({ name: FAILOVER_SKILL_NAME });
    if (!config.chain.length) throw new Error('A failover chain needs at least one model');
    this.chain = config.chain;
  }

  /** The chain's models, primary first. */
  get models(): string[] {
    return this.chain.map((m) => m.model);
  }

  override async initialize(): Promise<void> {
    for (const member of this.chain) await member.skill.initialize?.();
  }

  override async cleanup(): Promise<void> {
    for (const member of this.chain) {
      try {
        await member.skill.cleanup?.();
      } catch {
        // A member that cannot close must not stop the others.
      }
    }
  }

  private handoffOf(member: FailoverMember): Handoff {
    const found = member.skill.handoffs?.[0];
    if (!found) throw new Error(`${member.model}: the skill has no model handoff`);
    return found;
  }

  // Above every member's own priority, so this is the agent's handoff.
  @handoff({ name: FAILOVER_SKILL_NAME, priority: 100 })
  async *processUAMP(events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent, void, unknown> {
    this.notes = [];
    this.answeredModel = undefined;
    // One response id across attempts: the first `response.created` is the
    // one the client binds to, and every later event carries its id.
    let responseId: string | undefined;
    for (const [index, member] of this.chain.entries()) {
      const last = index === this.chain.length - 1;
      let started = false;
      let failed = false;
      let moveOn: string | undefined;
      for await (const event of this.handoffOf(member).handler(events, context)) {
        const carried = event as ServerEvent & { response_id?: string };
        if (event.type === 'response.created') {
          if (responseId) continue;
          responseId = carried.response_id;
          yield event;
          continue;
        }
        const out = (responseId && carried.response_id && carried.response_id !== responseId
          ? { ...carried, response_id: responseId }
          : event) as ServerEvent;
        if (event.type === 'response.error' && !started && !last) {
          const failure = providerFailure((event as { error?: { code?: string; message?: string } }).error);
          if (failure) {
            moveOn = failure.reason;
            break;
          }
        }
        if (event.type === 'response.delta' || event.type === 'response.done') started = true;
        if (event.type === 'response.error') failed = true;
        yield out;
      }
      if (moveOn === undefined) {
        // Answered, or failed for a reason that is the request's own: either
        // way the chain stops here, and only an answer names a model.
        if (!failed) this.answeredModel = member.model;
        return;
      }
      const next = this.chain[index + 1];
      const note = failoverNote(member.model, moveOn, next.model);
      this.notes.push(note);
      console.warn(`[failover] ${note}`);
      yield {
        type: 'progress',
        event_id: generateEventId(),
        target: 'response',
        ...(responseId ? { target_id: responseId } : {}),
        stage: FAILOVER_STAGE,
        message: note,
      } as ServerEvent;
    }
  }
}
