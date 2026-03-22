/**
 * Transport Multimodal Tests
 *
 * Tests Completions and A2A transport skills' handling of
 * multimodal content items — format conversion and non-text data part routing.
 */

import { describe, it, expect, vi } from 'vitest';
import { CompletionsTransportSkill } from '../../../src/skills/transport/completions/skill.js';
import { A2ATransportSkill } from '../../../src/skills/transport/a2a/skill.js';
import { generateEventId } from '../../../src/uamp/events.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';

describe('CompletionsTransportSkill.toUAMP', () => {
  const skill = new CompletionsTransportSkill();

  it('converts text-only messages', () => {
    const events = skill.toUAMP({
      model: 'test',
      messages: [{ role: 'user', content: 'Hello' }],
    });

    const inputText = events.find(e => e.type === 'input.text');
    expect(inputText).toBeDefined();
    expect((inputText as any).text).toBe('Hello');
  });

  it('converts multimodal array with image_url to input.image event', () => {
    const events = skill.toUAMP({
      model: 'test',
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: 'What is in this image?' },
          { type: 'image_url', image_url: { url: 'https://example.com/cat.png', detail: 'high' } },
        ],
      }],
    });

    const textEvent = events.find(e => e.type === 'input.text');
    expect(textEvent).toBeDefined();
    expect((textEvent as any).text).toBe('What is in this image?');

    const imageEvent = events.find(e => e.type === 'input.image');
    expect(imageEvent).toBeDefined();
    expect((imageEvent as any).image).toBe('https://example.com/cat.png');
  });

  it('converts multimodal array with input_audio to input.audio event', () => {
    const events = skill.toUAMP({
      model: 'test',
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: 'Transcribe this audio' },
          { type: 'input_audio', input_audio: { data: 'base64audio==', format: 'wav' } },
        ],
      }],
    });

    const audioEvent = events.find(e => e.type === 'input.audio');
    expect(audioEvent).toBeDefined();
    expect((audioEvent as any).audio).toBe('base64audio==');
    expect((audioEvent as any).format).toBe('wav');
  });

  it('includes session.create and response.create bookend events', () => {
    const events = skill.toUAMP({
      model: 'test',
      messages: [{ role: 'user', content: 'hi' }],
    });

    expect(events[0].type).toBe('session.create');
    expect(events[events.length - 1].type).toBe('response.create');
  });

  it('passes payment token in session extensions', () => {
    const events = skill.toUAMP(
      { model: 'test', messages: [{ role: 'user', content: 'hi' }] },
      'pay-token-123',
    );

    const session = events[0] as any;
    expect(session.session.extensions['X-Payment-Token']).toBe('pay-token-123');
  });
});

describe('A2ATransportSkill multimodal parts', () => {
  function createMockAgent(capturedEvents: ClientEvent[]) {
    return {
      name: 'test',
      description: 'test',
      getCapabilities: () => ({
        id: 'test', provider: 'webagents', modalities: ['text'],
        supports_streaming: true, supports_thinking: false, supports_caching: false,
      }),
      async *processUAMP(events: ClientEvent[]): AsyncGenerator<ServerEvent> {
        capturedEvents.push(...events);
        yield {
          type: 'response.done', event_id: generateEventId(), response_id: 'r1',
          response: { id: 'r1', status: 'completed', output: [{ type: 'text', text: 'ok' }] },
        } as unknown as ServerEvent;
      },
      run: vi.fn(), runStreaming: vi.fn(),
    };
  }

  function makeCtx() {
    return {
      session: { id: 's', created_at: 0, last_activity: 0, data: {} },
      auth: { authenticated: false }, payment: { valid: false }, metadata: {},
      get: () => undefined, set: () => {}, delete: () => {},
      hasScope: () => false, hasScopes: () => false,
    } as any;
  }

  it('converts image data parts to input.image events', async () => {
    const captured: ClientEvent[] = [];
    const skill = new A2ATransportSkill();
    skill.setAgent(createMockAgent(captured) as any);

    const req = new Request('http://localhost/a2a', {
      method: 'POST',
      body: JSON.stringify({
        id: 'r1', method: 'tasks/send',
        params: { message: { role: 'user', parts: [
          { type: 'text', text: 'Analyze' },
          { type: 'data', data: 'imgbase64', mimeType: 'image/png' },
        ] } },
      }),
    });

    await skill.handleA2ARequest(req, makeCtx());
    expect(captured.some(e => e.type === 'input.image')).toBe(true);
  });

  it('converts audio data parts to input.audio events', async () => {
    const captured: ClientEvent[] = [];
    const skill = new A2ATransportSkill();
    skill.setAgent(createMockAgent(captured) as any);

    const req = new Request('http://localhost/a2a', {
      method: 'POST',
      body: JSON.stringify({
        id: 'r2', method: 'tasks/send',
        params: { message: { role: 'user', parts: [
          { type: 'data', data: 'audiobase64', mimeType: 'audio/wav' },
        ] } },
      }),
    });

    await skill.handleA2ARequest(req, makeCtx());
    const audioEvent = captured.find(e => e.type === 'input.audio');
    expect(audioEvent).toBeDefined();
    expect((audioEvent as any).audio).toBe('audiobase64');
  });

  it('converts file parts to input.file events', async () => {
    const captured: ClientEvent[] = [];
    const skill = new A2ATransportSkill();
    skill.setAgent(createMockAgent(captured) as any);

    const req = new Request('http://localhost/a2a', {
      method: 'POST',
      body: JSON.stringify({
        id: 'r3', method: 'tasks/send',
        params: { message: { role: 'user', parts: [
          { type: 'file', file: { uri: 'https://s3/doc.pdf', name: 'doc.pdf', mimeType: 'application/pdf' } },
        ] } },
      }),
    });

    await skill.handleA2ARequest(req, makeCtx());
    const fileEvent = captured.find(e => e.type === 'input.file');
    expect(fileEvent).toBeDefined();
    expect((fileEvent as any).filename).toBe('doc.pdf');
  });
});
