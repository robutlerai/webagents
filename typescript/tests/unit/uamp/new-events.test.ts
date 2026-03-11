import { describe, it, expect } from 'vitest';
import {
  createSessionEndEvent,
  createSessionErrorEvent,
  createResponseCancelledEvent,
  createAudioDoneEvent,
  createRateLimitEvent,
  isClientEvent,
  isServerEvent,
} from '../../../src/uamp/events.js';

describe('New UAMP Events', () => {
  it('should create session.end event', () => {
    const event = createSessionEndEvent('user_closed');
    expect(event.type).toBe('session.end');
    expect(event.reason).toBe('user_closed');
    expect(event.event_id).toBeTruthy();
    expect(isClientEvent(event)).toBe(true);
  });

  it('should create session.error event', () => {
    const event = createSessionErrorEvent('auth_failed', 'Token expired');
    expect(event.type).toBe('session.error');
    expect(event.error.code).toBe('auth_failed');
    expect(event.error.message).toBe('Token expired');
    expect(isServerEvent(event)).toBe(true);
  });

  it('should create response.cancelled event', () => {
    const event = createResponseCancelledEvent('resp-123');
    expect(event.type).toBe('response.cancelled');
    expect(event.response_id).toBe('resp-123');
    expect(isServerEvent(event)).toBe(true);
  });

  it('should create audio.done event', () => {
    const event = createAudioDoneEvent('resp-456', 5000);
    expect(event.type).toBe('audio.done');
    expect(event.response_id).toBe('resp-456');
    expect(event.duration_ms).toBe(5000);
    expect(isServerEvent(event)).toBe(true);
  });

  it('should create rate_limit event', () => {
    const event = createRateLimitEvent(100, 5, 1700000000);
    expect(event.type).toBe('rate_limit');
    expect(event.limit).toBe(100);
    expect(event.remaining).toBe(5);
    expect(event.reset_at).toBe(1700000000);
    expect(isServerEvent(event)).toBe(true);
  });

  it('should correctly classify conversation.item events as client', () => {
    const event = { type: 'conversation.item.create', event_id: 'test' } as any;
    expect(isClientEvent(event)).toBe(true);
  });

  it('should correctly classify input.audio_committed as client', () => {
    const event = { type: 'input.audio_committed', event_id: 'test' } as any;
    expect(isClientEvent(event)).toBe(true);
  });
});
