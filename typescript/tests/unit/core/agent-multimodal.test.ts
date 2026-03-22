/**
 * BaseAgent Multimodal Content Pipeline Tests
 *
 * Tests _buildConversationFromEvents handling of all input event types,
 * consecutive user-event merging, content_items preservation through the
 * agentic loop, and structured tool results.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { tool, handoff } from '../../../src/core/decorators.js';
import type { Context, StructuredToolResult, AgenticMessage } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';
import {
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
  createResponseDeltaEvent,
  createResponseDoneEvent,
  generateEventId,
} from '../../../src/uamp/events.js';
import type { ContentItem, ImageContent } from '../../../src/uamp/types.js';

class CaptureLLM extends Skill {
  captured: AgenticMessage[] = [];

  @handoff({ name: 'capture-llm' })
  async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
    this.captured = context.get<AgenticMessage[]>('_agentic_messages') || [];
    yield createResponseDeltaEvent('r1', { type: 'text', text: 'ok' });
    yield createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }]);
  }
}

describe('_buildConversationFromEvents', () => {
  let agent: BaseAgent;
  let captureLLM: CaptureLLM;

  beforeEach(() => {
    captureLLM = new CaptureLLM();
    agent = new BaseAgent({ skills: [captureLLM] });
  });

  it('handles input.text events', async () => {
    const events: ClientEvent[] = [
      createSessionCreateEvent({ modalities: ['text'] }),
      createInputTextEvent('Hello'),
      createResponseCreateEvent(),
    ];

    for await (const _e of agent.processUAMP(events)) { /* drain */ }

    const userMsg = captureLLM.captured.find(m => m.role === 'user');
    expect(userMsg).toBeDefined();
    expect(userMsg!.content_items).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'text', text: 'Hello' })]),
    );
  });

  it('handles input.image events', async () => {
    const events: ClientEvent[] = [
      createSessionCreateEvent({ modalities: ['text', 'image'] }),
      { type: 'input.image', event_id: generateEventId(), image: { url: '/api/content/img-1' }, detail: 'high' } as ClientEvent,
      createResponseCreateEvent(),
    ];

    for await (const _e of agent.processUAMP(events)) { /* drain */ }

    const userMsg = captureLLM.captured.find(m => m.role === 'user');
    expect(userMsg).toBeDefined();
    expect(userMsg!.content_items).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'image' })]),
    );
  });

  it('handles input.audio events', async () => {
    const events: ClientEvent[] = [
      createSessionCreateEvent({ modalities: ['text', 'audio'] }),
      { type: 'input.audio', event_id: generateEventId(), audio: 'base64data', format: 'mp3' } as ClientEvent,
      createResponseCreateEvent(),
    ];

    for await (const _e of agent.processUAMP(events)) { /* drain */ }

    const userMsg = captureLLM.captured.find(m => m.role === 'user');
    expect(userMsg!.content_items).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'audio', audio: 'base64data' })]),
    );
  });

  it('handles input.video events', async () => {
    const events: ClientEvent[] = [
      createSessionCreateEvent({ modalities: ['text', 'video'] }),
      { type: 'input.video', event_id: generateEventId(), video: { url: 'https://cdn/v.mp4' }, format: 'mp4' } as ClientEvent,
      createResponseCreateEvent(),
    ];

    for await (const _e of agent.processUAMP(events)) { /* drain */ }

    const userMsg = captureLLM.captured.find(m => m.role === 'user');
    expect(userMsg!.content_items).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'video' })]),
    );
  });

  it('handles input.file events', async () => {
    const events: ClientEvent[] = [
      createSessionCreateEvent({ modalities: ['text', 'file'] }),
      { type: 'input.file', event_id: generateEventId(), file: { url: 'https://s3/doc.pdf' }, filename: 'doc.pdf', mime_type: 'application/pdf' } as ClientEvent,
      createResponseCreateEvent(),
    ];

    for await (const _e of agent.processUAMP(events)) { /* drain */ }

    const userMsg = captureLLM.captured.find(m => m.role === 'user');
    expect(userMsg!.content_items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'file', filename: 'doc.pdf', mime_type: 'application/pdf' }),
      ]),
    );
  });

  it('merges consecutive user events into single message', async () => {
    const events: ClientEvent[] = [
      createSessionCreateEvent({ modalities: ['text', 'image'] }),
      createInputTextEvent('Look at this:'),
      { type: 'input.image', event_id: generateEventId(), image: 'base64img' } as ClientEvent,
      createInputTextEvent('What do you think?'),
      createResponseCreateEvent(),
    ];

    for await (const _e of agent.processUAMP(events)) { /* drain */ }

    const userMessages = captureLLM.captured.filter(m => m.role === 'user');
    expect(userMessages).toHaveLength(1);
    expect(userMessages[0].content_items).toHaveLength(3);
    expect(userMessages[0].content_items![0]).toMatchObject({ type: 'text', text: 'Look at this:' });
    expect(userMessages[0].content_items![1]).toMatchObject({ type: 'image' });
    expect(userMessages[0].content_items![2]).toMatchObject({ type: 'text', text: 'What do you think?' });
  });

  it('system input.text does NOT merge with user messages', async () => {
    const events: ClientEvent[] = [
      createSessionCreateEvent({ modalities: ['text'] }),
      createInputTextEvent('Be brief', 'system'),
      createInputTextEvent('Hi'),
      createResponseCreateEvent(),
    ];

    for await (const _e of agent.processUAMP(events)) { /* drain */ }

    const systemMsgs = captureLLM.captured.filter(m => m.role === 'system');
    const userMsgs = captureLLM.captured.filter(m => m.role === 'user');
    expect(systemMsgs.length).toBeGreaterThanOrEqual(1);
    expect(userMsgs.length).toBeGreaterThanOrEqual(1);
  });
});

describe('content_items through run()', () => {
  it('preserves content_items from initial messages in the agentic context', async () => {
    const captureLLM = new CaptureLLM();
    const agent = new BaseAgent({ skills: [captureLLM] });

    const imageItem: ContentItem = { type: 'image', image: { url: '/api/content/img-1' } };
    await agent.run([
      {
        role: 'user',
        content: 'Describe this image',
        content_items: [
          { type: 'text', text: 'Describe this image' },
          imageItem,
        ],
      },
    ]);

    const userMsg = captureLLM.captured.find(m => m.role === 'user');
    expect(userMsg).toBeDefined();
    expect(userMsg!.content_items).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'image' })]),
    );
  });
});

describe('_content_registry population', () => {
  let agent: BaseAgent;
  let captureLLM: CaptureLLM;

  beforeEach(() => {
    captureLLM = new CaptureLLM();
    agent = new BaseAgent({ skills: [captureLLM] });
  });

  it('registers input.image content in _content_registry', async () => {
    const events: ClientEvent[] = [
      createSessionCreateEvent({ modalities: ['text', 'image'] }),
      { type: 'input.image', event_id: generateEventId(), image: { url: '/api/content/a1b2c3d4-e5f6-7890-abcd-ef1234567890' }, detail: 'high' } as ClientEvent,
      createResponseCreateEvent(),
    ];

    for await (const _e of agent.processUAMP(events)) { /* drain */ }

    const registry = (agent as any).context.get('_content_registry') as Map<string, ContentItem>;
    expect(registry).toBeDefined();
    expect(registry.size).toBeGreaterThanOrEqual(1);
    const entry = registry.get('a1b2c3d4-e5f6-7890-abcd-ef1234567890');
    expect(entry).toBeDefined();
    expect(entry!.type).toBe('image');
  });

  it('preserves content_id from input event (not regenerated)', async () => {
    const events: ClientEvent[] = [
      createSessionCreateEvent({ modalities: ['text', 'image'] }),
      { type: 'input.image', event_id: generateEventId(), image: 'base64data', content_id: 'my-custom-uuid' } as ClientEvent,
      createResponseCreateEvent(),
    ];

    for await (const _e of agent.processUAMP(events)) { /* drain */ }

    const userMsg = captureLLM.captured.find(m => m.role === 'user');
    const imgItem = userMsg!.content_items!.find(ci => ci.type === 'image') as ImageContent;
    expect(imgItem.content_id).toBe('my-custom-uuid');

    const registry = (agent as any).context.get('_content_registry') as Map<string, ContentItem>;
    expect(registry.get('my-custom-uuid')).toBeDefined();
  });

  it('assigns new UUID when input event has no content_id', async () => {
    const events: ClientEvent[] = [
      createSessionCreateEvent({ modalities: ['text', 'image'] }),
      { type: 'input.image', event_id: generateEventId(), image: 'base64data' } as ClientEvent,
      createResponseCreateEvent(),
    ];

    for await (const _e of agent.processUAMP(events)) { /* drain */ }

    const userMsg = captureLLM.captured.find(m => m.role === 'user');
    const imgItem = userMsg!.content_items!.find(ci => ci.type === 'image') as ImageContent;
    expect(imgItem.content_id).toBeDefined();
    expect(imgItem.content_id).toMatch(/^[0-9a-f-]{36}$/);
  });
});

describe('structured tool results in conversation', () => {
  it('stores StructuredToolResult content_items in conversation', async () => {
    let secondCallConv: AgenticMessage[] = [];

    class MediaTool extends Skill {
      @tool({ name: 'gen_image', description: 'Generate an image' })
      async genImage(_p: Record<string, unknown>, _c: Context): Promise<StructuredToolResult> {
        return {
          text: 'Generated image',
          content_items: [{ type: 'image', image: { url: '/api/content/gen-1' } } as ImageContent],
        };
      }
    }

    let callCount = 0;
    class InspectLLM extends Skill {
      @handoff({ name: 'inspect-llm' })
      async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
        callCount++;
        if (callCount === 1) {
          yield createResponseDoneEvent('r1', [
            { type: 'tool_call', tool_call: { id: 'tc_1', name: 'gen_image', arguments: '{}' } },
          ]);
        } else {
          secondCallConv = [...(context.get<AgenticMessage[]>('_agentic_messages') || [])];
          yield createResponseDeltaEvent('r2', { type: 'text', text: 'Here it is' });
          yield createResponseDoneEvent('r2', [{ type: 'text', text: 'Here it is' }]);
        }
      }
    }

    const agent = new BaseAgent({ skills: [new MediaTool(), new InspectLLM()] });
    await agent.run([{ role: 'user', content: 'make an image' }]);

    const toolMsg = secondCallConv.find(m => m.role === 'tool' && m.name === 'gen_image');
    expect(toolMsg).toBeDefined();
    expect(toolMsg!.content).toBe('Generated image');
    expect(toolMsg!.content_items).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'image' })]),
    );
  });

  it('registers StructuredToolResult content_items in _content_registry', async () => {
    let secondCallRegistry: Map<string, ContentItem> | undefined;

    class MediaTool extends Skill {
      @tool({ name: 'gen_image', description: 'Generate an image' })
      async genImage(_p: Record<string, unknown>, _c: Context): Promise<StructuredToolResult> {
        return {
          text: 'Generated image',
          content_items: [{ type: 'image', image: { url: '/api/content/a1b2c3d4-e5f6-7890-abcd-ef1234567890' } } as ImageContent],
        };
      }
    }

    let callCount2 = 0;
    class InspectLLM2 extends Skill {
      @handoff({ name: 'inspect-llm' })
      async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
        callCount2++;
        if (callCount2 === 1) {
          yield createResponseDoneEvent('r1', [
            { type: 'tool_call', tool_call: { id: 'tc_1', name: 'gen_image', arguments: '{}' } },
          ]);
        } else {
          secondCallRegistry = context.get<Map<string, ContentItem>>('_content_registry');
          yield createResponseDeltaEvent('r2', { type: 'text', text: 'Here' });
          yield createResponseDoneEvent('r2', [{ type: 'text', text: 'Here' }]);
        }
      }
    }

    const agent2 = new BaseAgent({ skills: [new MediaTool(), new InspectLLM2()] });
    await agent2.run([{ role: 'user', content: 'make an image' }]);

    expect(secondCallRegistry).toBeDefined();
    expect(secondCallRegistry!.size).toBeGreaterThanOrEqual(1);
    const entry = secondCallRegistry!.get('a1b2c3d4-e5f6-7890-abcd-ef1234567890');
    expect(entry).toBeDefined();
    expect(entry!.type).toBe('image');
  });
});
