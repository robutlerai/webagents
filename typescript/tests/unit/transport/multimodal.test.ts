/**
 * Transport Multimodal Tests
 *
 * Tests the Completions transport skill's handling of multimodal content
 * items — format conversion and non-text data part routing. The A2A part
 * mapping moved to `a2a-v1.test.ts` with the v1.0 transport (2026-09-26).
 */

import { describe, it, expect } from 'vitest';
import { CompletionsTransportSkill } from '../../../src/skills/transport/completions/skill.js';

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
