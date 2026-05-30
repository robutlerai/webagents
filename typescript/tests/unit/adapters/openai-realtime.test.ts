/**
 * OpenAI Realtime adapter unit tests.
 *
 * Drives `openOpenAIRealtimeSession` with an injected fake `ws`-style socket so
 * we can assert the wire framing + token-usage metering without a network.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  openOpenAIRealtimeSession,
  OPENAI_REALTIME_OUTPUT_RATE,
} from '../../../src/adapters/openai-realtime.js';

/** Minimal stand-in for the `ws` library's EventEmitter-style socket. */
class FakeWS {
  static last: FakeWS | null = null;
  url: string;
  sent: string[] = [];
  private listeners: Record<string, Array<(...args: unknown[]) => void>> = {};
  constructor(url: string, _opts?: unknown) {
    this.url = url;
    FakeWS.last = this;
  }
  on(type: string, cb: (...args: unknown[]) => void) {
    (this.listeners[type] ||= []).push(cb);
  }
  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.emit('close', 1000, Buffer.from(''));
  }
  emit(type: string, ...args: unknown[]) {
    (this.listeners[type] || []).forEach((cb) => cb(...args));
  }
  open() {
    this.emit('open');
  }
  message(obj: unknown) {
    this.emit('message', JSON.stringify(obj));
  }
  sentJson(): Array<Record<string, unknown>> {
    return this.sent.map((s) => JSON.parse(s));
  }
}

function newSession(overrides: Record<string, unknown> = {}) {
  const onAudioChunk = vi.fn();
  const onTurnComplete = vi.fn();
  const onReady = vi.fn();
  const onUsage = vi.fn();
  const session = openOpenAIRealtimeSession({
    apiKey: 'k',
    model: 'gpt-realtime',
    voiceId: 'marin',
    systemPrompt: 'be brief',
    onAudioChunk,
    onTurnComplete,
    onReady,
    onUsage,
    webSocketImpl: FakeWS as unknown as never,
    ...overrides,
  });
  return { session, ws: FakeWS.last!, onAudioChunk, onTurnComplete, onReady, onUsage };
}

describe('openOpenAIRealtimeSession — setup + framing', () => {
  it('sends session.update on open with manual turn detection', () => {
    const { ws, onReady } = newSession();
    ws.open();
    const upd = ws.sentJson()[0];
    expect(upd.type).toBe('session.update');
    const sess = upd.session as Record<string, unknown>;
    expect(sess.voice).toBe('marin');
    expect(sess.turn_detection).toBeNull();
    expect(onReady).toHaveBeenCalledOnce();
  });

  it('passes through model audio deltas as raw 24k PCM base64', () => {
    const { ws, onAudioChunk } = newSession();
    ws.open();
    const pcm = Buffer.from(new Uint8Array([1, 2, 3])).toString('base64');
    ws.message({ type: 'response.audio.delta', delta: pcm });
    expect(onAudioChunk).toHaveBeenCalledOnce();
    expect(onAudioChunk.mock.calls[0]).toEqual([pcm, OPENAI_REALTIME_OUTPUT_RATE]);
  });
});

describe('openOpenAIRealtimeSession — usage metering', () => {
  it('parses response.done usage modality details', () => {
    const { ws, onUsage } = newSession();
    ws.open();
    ws.message({
      type: 'response.done',
      response: {
        usage: {
          input_tokens: 130,
          output_tokens: 240,
          input_token_details: { audio_tokens: 100, text_tokens: 30 },
          output_token_details: { audio_tokens: 200, text_tokens: 40 },
          cached_tokens_details: { audio_tokens: 25, text_tokens: 5 },
        },
      },
    });
    expect(onUsage).toHaveBeenCalledOnce();
    expect(onUsage.mock.calls[0][0]).toEqual({
      audioInputTokens: 100,
      audioOutputTokens: 200,
      textInputTokens: 30,
      textOutputTokens: 40,
      cachedAudioInputTokens: 25,
      cachedTextInputTokens: 5,
    });
  });

  it('SUMS usage across responses into a cumulative total', () => {
    const { ws, onUsage } = newSession();
    ws.open();
    const turn = () =>
      ws.message({
        type: 'response.done',
        response: {
          usage: {
            input_token_details: { audio_tokens: 100, text_tokens: 10 },
            output_token_details: { audio_tokens: 200, text_tokens: 20 },
          },
        },
      });
    turn();
    turn();
    expect(onUsage).toHaveBeenCalledTimes(2);
    // OpenAI bills per response, so two turns accumulate (unlike Gemini's
    // running-total snapshots which the relay would keep the last of).
    expect(onUsage.mock.calls.at(-1)![0]).toMatchObject({
      audioInputTokens: 200,
      audioOutputTokens: 400,
      textInputTokens: 20,
      textOutputTokens: 40,
    });
  });

  it('falls back to whole input/output tokens as audio when no modality split', () => {
    const { ws, onUsage } = newSession();
    ws.open();
    ws.message({
      type: 'response.done',
      response: { usage: { input_tokens: 60, output_tokens: 90 } },
    });
    expect(onUsage.mock.calls.at(-1)![0]).toMatchObject({
      audioInputTokens: 60,
      audioOutputTokens: 90,
      textInputTokens: 0,
      textOutputTokens: 0,
    });
  });
});
