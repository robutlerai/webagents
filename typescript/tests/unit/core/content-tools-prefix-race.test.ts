/**
 * The built-in content tools (`present`, `read_content`, `save_content`) on
 * one agent instance serving overlapping runs.
 *
 * The tools used to be registered at the start of every run with handlers
 * that closed over THAT run's conversation and collected items, and removed
 * when the run ended. One instance serves many runs at once, so the later
 * run's handlers replaced the earlier run's in the shared registry: the
 * earlier run's model then resolved ids against the later run's
 * conversation, and the first run to finish took the tools away from the
 * others mid-turn. The tools are now registered once and read the run's
 * state off the run context.
 *
 * Pinned here, with two runs provably in flight together:
 *   - each run's `present` sees its own conversation's ids and none of the
 *     other run's, and each run's output carries its own item only;
 *   - the tool list is the same on every model call of a run, with the
 *     built-ins in it, and `listTools()` outside a run does not list them;
 *   - `save_content` is listed only on a run that has a media saver, and
 *     that decision holds for the whole run.
 */

import { describe, it, expect } from 'vitest';
import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff } from '../../../src/core/decorators';
import type { Context } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDoneEvent, generateEventId } from '../../../src/uamp/events';
import type { ContentItem } from '../../../src/uamp/types';

const ID_A = '11111111-1111-4111-8111-111111111111';
const ID_B = '22222222-2222-4222-8222-222222222222';

/** Resolves for everyone once `n` callers have arrived, so runs are provably in flight together. */
function meeting(n: number): () => Promise<void> {
  let arrived = 0;
  let open!: () => void;
  const all = new Promise<void>((resolve) => { open = resolve; });
  return async () => {
    arrived += 1;
    if (arrived >= n) open();
    await all;
  };
}

/**
 * Stands in for the model. On its first call of a run it waits at the
 * meeting point, then presents the run's own id and the other run's id; on
 * its second call it records the tool results it was given and answers.
 */
class PresentsBoth extends Skill {
  readonly toolLists = new Map<string, string[]>();
  readonly results = new Map<string, string[]>();
  meet?: () => Promise<void>;

  @handoff({ name: 'presents-both' })
  async *processUAMP(_events: ClientEvent[], ctx: Context): AsyncGenerator<ServerEvent> {
    const who = String(ctx.auth?.user_id ?? 'anonymous');
    const messages = (ctx.get('_agentic_messages') as Array<{ role: string; content?: unknown }>) ?? [];
    const tools = (ctx.get('_agentic_tools') as Array<{ function: { name: string } }>) ?? [];
    const lists = this.toolLists.get(who) ?? [];
    lists.push(JSON.stringify(tools));
    this.toolLists.set(who, lists);
    const responseId = generateEventId();
    const toolRows = messages.filter((m) => m.role === 'tool').map((m) => String(m.content ?? ''));
    if (toolRows.length === 0) {
      if (this.meet) await this.meet();
      const mine = who === 'ada' ? ID_A : ID_B;
      const theirs = who === 'ada' ? ID_B : ID_A;
      const output: ContentItem[] = [
        { type: 'tool_call', tool_call: { id: `${who}-1`, name: 'present', arguments: JSON.stringify({ content_id: mine }) } },
        { type: 'tool_call', tool_call: { id: `${who}-2`, name: 'present', arguments: JSON.stringify({ content_id: theirs }) } },
      ];
      yield createResponseDoneEvent(responseId, output);
      return;
    }
    this.results.set(who, toolRows);
    yield createResponseDoneEvent(responseId, [{ type: 'text', text: 'done' }]);
  }
}

/** Puts a media saver on the run context the way a media skill does, on request. */
class SaverOnRequest extends Skill {
  constructor(private readonly saverFor: Set<string>) {
    super({ name: 'saver-on-request' });
    this.registerHook({
      lifecycle: 'on_connection',
      priority: 1,
      enabled: true,
      handler: async (_data: unknown, context: Context) => {
        if (this.saverFor.has(String(context.auth?.user_id ?? ''))) {
          context.set('_media_saver', { save: async () => ({ url: 'u', content_id: ID_A }) });
        }
      },
    });
  }
}

const picture = (id: string): ContentItem => ({ type: 'image', image: { url: `https://example.test/${id}.png` }, content_id: id } as ContentItem);

const conversationOf = (id: string) => [
  { role: 'user' as const, content: 'show it', content_items: [picture(id)] },
];

function names(list: string): string[] {
  return (JSON.parse(list) as Array<{ function: { name: string } }>).map((t) => t.function.name);
}

describe('built-in content tools across overlapping runs of one instance', () => {
  it('each run presents from its own conversation and gets its own item back', async () => {
    const model = new PresentsBoth();
    model.meet = meeting(2);
    const agent = new BaseAgent({ name: 'probe', skills: [model] });
    const rich = { client_capabilities: { supports_rich_display: true } };
    const [a, b] = await Promise.all([
      agent.run(conversationOf(ID_A), { userId: 'ada', auth: { scope: 'user' }, ...rich }),
      agent.run(conversationOf(ID_B), { userId: 'bob', auth: { scope: 'user' }, ...rich }),
    ]);

    const idsOf = (r: { content_items?: ContentItem[] }) =>
      (r.content_items ?? []).map((ci) => (ci as { content_id?: string }).content_id).filter((x) => x && x !== undefined);
    expect(idsOf(a as { content_items?: ContentItem[] })).toEqual([ID_A]);
    expect(idsOf(b as { content_items?: ContentItem[] })).toEqual([ID_B]);

    const ada = model.results.get('ada')!;
    const bob = model.results.get('bob')!;
    expect(ada[0]).toMatch(/^Displayed/);
    expect(ada[0]).toContain(ID_A);
    expect(ada[1]).toMatch(/^Content not found/);
    expect(ada[1]).toContain(`Available content_ids: ${ID_A}.`);
    expect(ada[1]).not.toContain(`Available content_ids: ${ID_B}`);
    expect(bob[0]).toMatch(/^Displayed/);
    expect(bob[0]).toContain(ID_B);
    expect(bob[1]).toMatch(/^Content not found/);
    expect(bob[1]).toContain(`Available content_ids: ${ID_B}.`);
  });

  it('the tool list is the same on every call of a run, with the built-ins in it, and empty of them outside a run', async () => {
    const model = new PresentsBoth();
    const agent = new BaseAgent({ name: 'probe', skills: [model] });
    await agent.run(conversationOf(ID_A), { userId: 'ada', auth: { scope: 'user' } });
    const lists = model.toolLists.get('ada')!;
    expect(lists).toHaveLength(2);
    expect(lists[0]).toBe(lists[1]);
    expect(names(lists[0])).toEqual(expect.arrayContaining(['present', 'read_content']));
    expect(names(lists[0])).not.toContain('save_content');

    const outside = (await agent.listTools({ userId: 'ada', auth: { scope: 'user' } })).map((t) => (t as { function: { name: string } }).function.name);
    expect(outside).not.toContain('present');
    expect(outside).not.toContain('read_content');
    expect(outside).not.toContain('save_content');
  });

  it('save_content is listed only on the run with a media saver, and on every call of it', async () => {
    const model = new PresentsBoth();
    model.meet = meeting(2);
    const agent = new BaseAgent({ name: 'probe', skills: [new SaverOnRequest(new Set(['ada'])), model] });
    await Promise.all([
      agent.run(conversationOf(ID_A), { userId: 'ada', auth: { scope: 'user' } }),
      agent.run(conversationOf(ID_B), { userId: 'bob', auth: { scope: 'user' } }),
    ]);
    const ada = model.toolLists.get('ada')!;
    const bob = model.toolLists.get('bob')!;
    expect(ada.map(names).every((n) => n.includes('save_content'))).toBe(true);
    expect(bob.map(names).some((n) => n.includes('save_content'))).toBe(false);
    expect(ada[0]).toBe(ada[1]);
    expect(bob[0]).toBe(bob[1]);
  });

  it('a built-in called outside a run says so instead of touching another run\'s state', async () => {
    const model = new PresentsBoth();
    const agent = new BaseAgent({ name: 'probe', skills: [model] });
    await agent.run(conversationOf(ID_A), { userId: 'ada', auth: { scope: 'user' } });
    const out = await agent.executeTool('present', { content_id: ID_A });
    expect(String(out)).toMatch(/only available while a run is in progress/);
  });
});
