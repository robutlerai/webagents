/**
 * Gemini Live adapter unit tests.
 *
 * Drives `openGeminiLiveSession` with an injected fake WebSocket so we can
 * assert the BidiGenerateContent wire framing without a network.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  openGeminiLiveSession,
  GEMINI_LIVE_OUTPUT_RATE,
} from '../../../src/adapters/google-live.js';

class FakeWS {
  static last: FakeWS | null = null;
  url: string;
  binaryType = 'blob';
  sent: string[] = [];
  private listeners: Record<string, Array<(ev: unknown) => void>> = {};
  constructor(url: string) {
    this.url = url;
    FakeWS.last = this;
  }
  addEventListener(type: string, cb: (ev: unknown) => void) {
    (this.listeners[type] ||= []).push(cb);
  }
  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.emit('close', {});
  }
  emit(type: string, ev: unknown) {
    (this.listeners[type] || []).forEach((cb) => cb(ev));
  }
  open() {
    this.emit('open', {});
  }
  message(obj: unknown) {
    this.emit('message', { data: JSON.stringify(obj) });
  }
  sentJson(): Array<Record<string, unknown>> {
    return this.sent.map((s) => JSON.parse(s));
  }
}

function newSession(overrides: Record<string, unknown> = {}) {
  const onAudioChunk = vi.fn();
  const onTurnComplete = vi.fn();
  const onInterrupted = vi.fn();
  const onReady = vi.fn();
  const onUsage = vi.fn();
  const session = openGeminiLiveSession({
    apiKey: 'k',
    model: 'gemini-2.0-flash-exp',
    voiceId: 'Aoede',
    systemPrompt: 'be brief',
    onAudioChunk,
    onTurnComplete,
    onInterrupted,
    onReady,
    onUsage,
    webSocketImpl: FakeWS as unknown as typeof WebSocket,
    ...overrides,
  });
  return { session, ws: FakeWS.last!, onAudioChunk, onTurnComplete, onInterrupted, onReady, onUsage };
}

describe('openGeminiLiveSession — setup', () => {
  it('sends a setup frame on open with audio modality, voice, and manual VAD', () => {
    const { ws } = newSession();
    ws.open();
    const setup = ws.sentJson()[0].setup as Record<string, unknown>;
    expect(setup.model).toBe('models/gemini-2.0-flash-exp');
    const gen = setup.generationConfig as Record<string, unknown>;
    expect(gen.responseModalities).toEqual(['AUDIO']);
    expect(JSON.stringify(gen.speechConfig)).toContain('Aoede');
    expect(JSON.stringify(setup.systemInstruction)).toContain('be brief');
    expect(JSON.stringify(setup.realtimeInputConfig)).toContain('disabled');
  });
});

describe('openGeminiLiveSession — readiness gating', () => {
  it('queues audio until setupComplete, then flushes', () => {
    const { session, ws, onReady } = newSession();
    ws.open();
    session.sendAudio(new Uint8Array([0, 0]));
    // Only the setup frame so far — audio is queued pre-ready.
    expect(ws.sent.length).toBe(1);
    ws.message({ setupComplete: {} });
    expect(onReady).toHaveBeenCalledOnce();
    expect(session.isReady()).toBe(true);
    // Queued audio frame flushed.
    const frames = ws.sentJson();
    expect(frames.some((f) => f.realtimeInput)).toBe(true);
  });
});

describe('openGeminiLiveSession — outbound framing', () => {
  it('sendAudio → realtimeInput.audio Blob with 16k pcm mime', () => {
    const { session, ws } = newSession();
    ws.open();
    ws.message({ setupComplete: {} });
    session.sendAudio(new Uint8Array([1, 2, 3, 4]));
    const last = ws.sentJson().at(-1)!.realtimeInput as Record<string, unknown>;
    // Gemini 3.1 uses a single `audio` Blob, not the deprecated mediaChunks array.
    expect(last.mediaChunks).toBeUndefined();
    const audio = last.audio as Record<string, unknown>;
    expect(audio.mimeType).toBe('audio/pcm;rate=16000');
    expect(typeof audio.data).toBe('string');
  });

  it('sendActivityStart / End and interrupt emit the right frames', () => {
    const { session, ws } = newSession();
    ws.open();
    ws.message({ setupComplete: {} });
    session.sendActivityStart();
    session.sendActivityEnd();
    session.interrupt();
    const frames = ws.sentJson().map((f) => f.realtimeInput).filter(Boolean) as Array<Record<string, unknown>>;
    expect(frames.some((f) => 'activityStart' in f)).toBe(true);
    expect(frames.some((f) => 'activityEnd' in f)).toBe(true);
  });
});

describe('openGeminiLiveSession — inbound', () => {
  it('serverContent inlineData → raw PCM base64 passed through unchanged', () => {
    const { ws, onAudioChunk } = newSession();
    ws.open();
    ws.message({ setupComplete: {} });
    const pcmB64 = Buffer.from(new Uint8Array([5, 6, 7, 8])).toString('base64');
    ws.message({
      serverContent: { modelTurn: { parts: [{ inlineData: { mimeType: 'audio/pcm;rate=24000', data: pcmB64 } }] } },
    });
    expect(onAudioChunk).toHaveBeenCalledOnce();
    const [audioB64, rate] = onAudioChunk.mock.calls[0];
    expect(rate).toBe(GEMINI_LIVE_OUTPUT_RATE);
    // Raw provider PCM16 passes straight through (no WAV container).
    expect(audioB64).toBe(pcmB64);
  });

  it('turnComplete and interrupted fire their callbacks', () => {
    const { ws, onTurnComplete, onInterrupted } = newSession();
    ws.open();
    ws.message({ setupComplete: {} });
    ws.message({ serverContent: { turnComplete: true } });
    ws.message({ serverContent: { interrupted: true } });
    expect(onTurnComplete).toHaveBeenCalledOnce();
    expect(onInterrupted).toHaveBeenCalledOnce();
  });
});

describe('openGeminiLiveSession — usage metering', () => {
  it('parses usageMetadata modality details into cumulative RealtimeUsage', () => {
    const { ws, onUsage } = newSession();
    ws.open();
    ws.message({ setupComplete: {} });
    ws.message({
      usageMetadata: {
        promptTokenCount: 130,
        responseTokenCount: 240,
        promptTokensDetails: [
          { modality: 'AUDIO', tokenCount: 100 },
          { modality: 'TEXT', tokenCount: 30 },
        ],
        responseTokensDetails: [
          { modality: 'AUDIO', tokenCount: 200 },
          { modality: 'TEXT', tokenCount: 40 },
        ],
        cacheTokensDetails: [{ modality: 'AUDIO', tokenCount: 25 }],
      },
      serverContent: { turnComplete: true },
    });
    expect(onUsage).toHaveBeenCalledOnce();
    expect(onUsage.mock.calls[0][0]).toEqual({
      audioInputTokens: 100,
      audioOutputTokens: 200,
      textInputTokens: 30,
      textOutputTokens: 40,
      cachedAudioInputTokens: 25,
      cachedTextInputTokens: 0,
    });
  });

  it('attributes unsplit token counts to audio (voice session)', () => {
    const { ws, onUsage } = newSession();
    ws.open();
    ws.message({ setupComplete: {} });
    ws.message({ usageMetadata: { promptTokenCount: 50, responseTokenCount: 80 } });
    expect(onUsage.mock.calls.at(-1)![0]).toMatchObject({
      audioInputTokens: 50,
      audioOutputTokens: 80,
      textInputTokens: 0,
      textOutputTokens: 0,
    });
  });

  it('emits the LATEST cumulative snapshot (Gemini reports running totals)', () => {
    const { ws, onUsage } = newSession();
    ws.open();
    ws.message({ setupComplete: {} });
    ws.message({ usageMetadata: { promptTokensDetails: [{ modality: 'AUDIO', tokenCount: 100 }] } });
    ws.message({ usageMetadata: { promptTokensDetails: [{ modality: 'AUDIO', tokenCount: 220 }] } });
    // Two snapshots, each a running total — the relay keeps the last (220), it
    // does NOT sum to 320.
    expect(onUsage).toHaveBeenCalledTimes(2);
    expect(onUsage.mock.calls.at(-1)![0].audioInputTokens).toBe(220);
  });

  it('does not warn or callback for a usage-only message without serverContent', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { ws, onUsage } = newSession();
    ws.open();
    ws.message({ setupComplete: {} });
    ws.message({ usageMetadata: { promptTokenCount: 10 } });
    expect(onUsage).toHaveBeenCalledOnce();
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });
});
