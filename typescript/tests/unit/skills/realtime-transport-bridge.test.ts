/**
 * RealtimeTransportSkill ↔ Gemini Live bridge tests.
 *
 * Wires a fake widget socket into `handleRealtimeConnection` and a fake
 * provider socket into the adapter (via `provider.webSocketImpl`), then
 * asserts the bidirectional UAMP ↔ BidiGenerateContent translation:
 *   input.audio          → realtimeInput.activityStart + mediaChunks
 *   input.audio_committed → realtimeInput.activityEnd
 *   serverContent inline  → response.audio.delta
 *   response.cancel       → realtimeInput.activityStart (interrupt)
 */

import { describe, it, expect, vi } from 'vitest';
import { RealtimeTransportSkill } from '../../../src/skills/transport/realtime/index.js';

// Provider-side fake (the adapter constructs this via webSocketImpl).
class GeminiFakeWS {
  static last: GeminiFakeWS | null = null;
  binaryType = 'blob';
  sent: string[] = [];
  private l: Record<string, Array<(ev: unknown) => void>> = {};
  constructor(_url: string) {
    GeminiFakeWS.last = this;
  }
  addEventListener(t: string, cb: (ev: unknown) => void) {
    (this.l[t] ||= []).push(cb);
  }
  send(d: string) {
    this.sent.push(d);
  }
  close() {
    this.emit('close', {});
  }
  emit(t: string, ev: unknown) {
    (this.l[t] || []).forEach((cb) => cb(ev));
  }
  open() {
    this.emit('open', {});
  }
  message(o: unknown) {
    this.emit('message', { data: JSON.stringify(o) });
  }
  json() {
    return this.sent.map((s) => JSON.parse(s));
  }
}

// Widget-side fake (passed into handleRealtimeConnection).
class WidgetFakeWS {
  sent: string[] = [];
  private l: Record<string, Array<(ev: unknown) => void>> = {};
  addEventListener(t: string, cb: (ev: unknown) => void) {
    (this.l[t] ||= []).push(cb);
  }
  send(d: string) {
    this.sent.push(d);
  }
  close() {
    this.emit('close', {});
  }
  emit(t: string, ev: unknown) {
    (this.l[t] || []).forEach((cb) => cb(ev));
  }
  fromWidget(o: unknown) {
    this.emit('message', { data: JSON.stringify(o) });
  }
  json() {
    return this.sent.map((s) => JSON.parse(s));
  }
  types() {
    return this.json().map((m) => m.type);
  }
}

async function connect() {
  const skill = new RealtimeTransportSkill({
    provider: {
      provider: 'gemini',
      model: 'gemini-2.0-flash-exp',
      voiceId: 'Aoede',
      systemPrompt: 'be brief',
      apiKey: 'k',
      webSocketImpl: GeminiFakeWS as unknown as typeof WebSocket,
    },
  });
  const widget = new WidgetFakeWS();
  await skill.handleRealtimeConnection(
    { ws: widget, metadata: { transport: 'realtime' } } as never,
    {} as never,
  );
  const gemini = GeminiFakeWS.last!;
  gemini.open();
  gemini.message({ setupComplete: {} });
  return { skill, widget, gemini };
}

const b64 = (bytes: number[]) => Buffer.from(new Uint8Array(bytes)).toString('base64');

describe('RealtimeTransportSkill bridge', () => {
  it('emits session.created to the widget on connect', async () => {
    const { widget } = await connect();
    expect(widget.types()).toContain('session.created');
  });

  it('input.audio → provider activityStart then audio Blob', async () => {
    const { widget, gemini } = await connect();
    widget.fromWidget({ type: 'input.audio', audio: b64([1, 2, 3, 4]) });
    const ri = gemini.json().map((f) => f.realtimeInput).filter(Boolean) as Array<Record<string, unknown>>;
    expect(ri.some((f) => 'activityStart' in f)).toBe(true);
    expect(ri.some((f) => 'audio' in f)).toBe(true);
  });

  it('input.audio_committed → provider activityEnd', async () => {
    const { widget, gemini } = await connect();
    widget.fromWidget({ type: 'input.audio', audio: b64([1, 2]) });
    widget.fromWidget({ type: 'input.audio_committed' });
    const ri = gemini.json().map((f) => f.realtimeInput).filter(Boolean) as Array<Record<string, unknown>>;
    expect(ri.some((f) => 'activityEnd' in f)).toBe(true);
  });

  it('provider serverContent audio → widget response.audio.delta', async () => {
    const { widget, gemini } = await connect();
    gemini.message({
      serverContent: { modelTurn: { parts: [{ inlineData: { mimeType: 'audio/pcm;rate=24000', data: b64([9, 9]) } }] } },
    });
    const delta = widget.json().find((m) => m.type === 'response.audio.delta');
    expect(delta).toBeTruthy();
    expect(typeof delta.audio).toBe('string');
  });

  it('response.cancel → provider interrupt (activityStart)', async () => {
    const { widget, gemini } = await connect();
    // Open a turn first so a fresh activityStart is unambiguous.
    widget.fromWidget({ type: 'input.audio', audio: b64([1]) });
    const before = gemini.json().filter((f) => f.realtimeInput?.activityStart).length;
    widget.fromWidget({ type: 'response.cancel' });
    const after = gemini.json().filter((f) => f.realtimeInput?.activityStart).length;
    expect(after).toBeGreaterThan(before);
  });

  it('provider turnComplete → widget response.done', async () => {
    const { widget, gemini } = await connect();
    gemini.message({ serverContent: { turnComplete: true } });
    expect(widget.types()).toContain('response.done');
  });
});
