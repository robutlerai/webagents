/**
 * A refused internal tool call reaches no hook.
 *
 * THE DEFECT THIS PINS. Both agentic loops ran `before_tool` /
 * `before_toolcall` and only then let the executor decide that the call was
 * refused (a tool hidden at this turn's posture, a scope the caller lacks, a
 * name the registry does not hold). The executor answered with a plain string,
 * which the loops did not treat as an error, so `after_tool` / `after_toolcall`
 * then ran with a "successful" result. Any hook that reserves something for a
 * call and settles it afterwards (a payment lock, a quota, a rate budget)
 * charged for a tool that never ran.
 *
 * The rule under test: the refusal is decided before any hook runs, no hook
 * sees the call, the model is told the refusal, and the `tool_result` on the
 * wire carries `is_error: true`.
 */

import { describe, it, expect } from 'vitest';
import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff, hook, tool } from '../../../src/core/decorators';
import type { Context, HookData } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDeltaEvent, createResponseDoneEvent, generateEventId } from '../../../src/uamp/events';
import type { ContentItem } from '../../../src/uamp/types';

/** Unclassified, so denied in a restricted turn; `peek` opts in. */
class SpendSkill extends Skill {
  calls = 0;
  @tool({ name: 'spend', description: 'spends money' })
  async spend(): Promise<string> { this.calls++; return 'spent'; }
  @tool({ name: 'peek', description: 'read-only', restrictedPosture: 'allow' })
  async peek(): Promise<string> { return 'peeked'; }
  @tool({ name: 'admin_only', description: 'needs a scope', scopes: ['admin'], restrictedPosture: 'allow' })
  async adminOnly(): Promise<string> { return 'admin'; }
}

/** Records every hook it sees, with the tool name. */
class LedgerSkill extends Skill {
  static restrictedPostureDefault = 'allow' as const;
  seen: string[] = [];
  @hook({ lifecycle: 'before_tool' })
  async before(d: HookData): Promise<void> { this.seen.push(`before_tool:${d.tool_name}`); }
  @hook({ lifecycle: 'before_toolcall' })
  async beforeCall(d: HookData): Promise<void> { this.seen.push(`before_toolcall:${d.tool_name}`); }
  @hook({ lifecycle: 'after_tool' })
  async after(d: HookData): Promise<void> { this.seen.push(`after_tool:${d.tool_name}`); }
  @hook({ lifecycle: 'after_toolcall' })
  async afterCall(d: HookData): Promise<void> { this.seen.push(`after_toolcall:${d.tool_name}`); }
}

type Scripted = { text?: string; toolCalls?: Array<{ id: string; name: string; arguments: string }> };

/** Returns the scripted tool calls once, then text; records the tool results it is fed. */
class ScriptedLLM extends Skill {
  static restrictedPostureDefault = 'allow' as const;
  toolResults: Array<{ name?: string; content: unknown }> = [];
  private i = 0;
  constructor(private readonly script: Scripted[]) { super(); }
  @handoff({ name: 'scripted-llm' })
  async *processUAMP(_events: ClientEvent[], ctx: Context): AsyncGenerator<ServerEvent> {
    const msgs = ctx.get<Array<{ role: string; name?: string; content: unknown }>>('_agentic_messages') ?? [];
    for (const m of msgs) if (m.role === 'tool') this.toolResults.push({ name: m.name, content: m.content });
    const step = this.script[this.i++] ?? { text: 'done' };
    const rid = generateEventId();
    yield { type: 'response.created', event_id: generateEventId(), response_id: rid } as ServerEvent;
    const output: ContentItem[] = [];
    if (step.text) output.push({ type: 'text', text: step.text });
    for (const tc of step.toolCalls ?? []) {
      yield createResponseDeltaEvent(rid, { type: 'tool_call', tool_call: tc });
      output.push({ type: 'tool_call', tool_call: tc });
    }
    yield createResponseDoneEvent(rid, output);
  }
}

const hi = [{ role: 'user' as const, content: 'hi' }];
const at = (posture: unknown) => ({ metadata: { turn: { id: null, posture } } });

function build(script: Scripted[]) {
  const spend = new SpendSkill();
  const ledger = new LedgerSkill();
  const llm = new ScriptedLLM(script);
  const agent = new BaseAgent({ name: 'probe', skills: [spend, ledger, llm] });
  return { agent, spend, ledger, llm };
}

async function collect<T>(gen: AsyncGenerator<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const e of gen) out.push(e);
  return out;
}

/** `runStreaming` yields flattened chunks: `{ type: 'tool_result', tool_result }`. */
type Chunk = { type?: string; tool_result?: { call_id: string; result: string; is_error?: boolean } };
function toolResultDeltas(chunks: unknown[]) {
  return (chunks as Chunk[]).filter((c) => c?.type === 'tool_result').map((c) => c.tool_result!);
}

describe('a refused internal tool call reaches no hook', () => {
  it('hidden at restricted posture: no hook, handler never runs, result is an error', async () => {
    const { agent, spend, ledger, llm } = build([
      { toolCalls: [{ id: 'c1', name: 'spend', arguments: '{}' }] },
      { text: 'ok' },
    ]);
    const events = await collect(agent.runStreaming(hi, at('restricted')));

    expect(spend.calls).toBe(0);
    expect(ledger.seen).toEqual([]);
    const results = toolResultDeltas(events);
    expect(results).toHaveLength(1);
    expect(results[0]).toMatchObject({ call_id: 'c1', is_error: true });
    expect(results[0].result).toMatch(/not available in this turn: spend/);
    expect(llm.toolResults).toEqual([{ name: 'spend', content: expect.stringMatching(/not available in this turn/) }]);
  });

  it('out of scope: no hook, result is an error', async () => {
    const { agent, ledger } = build([
      { toolCalls: [{ id: 'c1', name: 'admin_only', arguments: '{}' }] },
      { text: 'ok' },
    ]);
    const events = await collect(agent.runStreaming(hi));
    expect(ledger.seen).toEqual([]);
    expect(toolResultDeltas(events)[0]).toMatchObject({ call_id: 'c1', is_error: true });
    expect(toolResultDeltas(events)[0].result).toMatch(/Insufficient permissions/);
  });

  it('unparseable arguments: no hook, result is an error', async () => {
    const { agent, ledger } = build([
      { toolCalls: [{ id: 'c1', name: 'peek', arguments: '{not json' }] },
      { text: 'ok' },
    ]);
    const events = await collect(agent.runStreaming(hi));
    expect(ledger.seen).toEqual([]);
    expect(toolResultDeltas(events)[0]).toMatchObject({ call_id: 'c1', is_error: true });
  });

  it('the mixed loop (an external tool in the same turn) refuses the same way', async () => {
    const { agent, spend, ledger } = build([
      { toolCalls: [
        { id: 'c1', name: 'spend', arguments: '{}' },
        { id: 'c2', name: 'client_side_tool', arguments: '{}' },
      ] },
      { text: 'ok' },
    ]);
    const events = await collect(agent.runStreaming(hi, at('restricted')));
    expect(spend.calls).toBe(0);
    expect(ledger.seen).toEqual([]);
    const refused = toolResultDeltas(events).find((r) => r.call_id === 'c1');
    expect(refused).toMatchObject({ is_error: true });
    expect(refused?.result).toMatch(/not available in this turn: spend/);
  });

  it('an allowed call still runs every hook, in order', async () => {
    const { agent, ledger } = build([
      { toolCalls: [{ id: 'c1', name: 'peek', arguments: '{}' }] },
      { text: 'ok' },
    ]);
    const events = await collect(agent.runStreaming(hi, at('restricted')));
    expect(ledger.seen).toEqual([
      'before_tool:peek', 'before_toolcall:peek', 'after_tool:peek', 'after_toolcall:peek',
    ]);
    expect(toolResultDeltas(events)[0]).toMatchObject({ call_id: 'c1', result: 'peeked' });
    expect(toolResultDeltas(events)[0].is_error).toBeUndefined();
  });
});
