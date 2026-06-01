/**
 * OpenAI Realtime API adapter (WebSocket, server-side).
 *
 * Sibling to `google-live.ts`: a bidirectional WS transport for realtime voice
 * that runs server-side and holds the provider WS for the session lifetime. The
 * provider API key never leaves this process (ADR-v3-12).
 *
 * Auth difference from Gemini: OpenAI requires an `Authorization: Bearer`
 * HEADER. The global WHATWG `WebSocket` (undici) can't set request headers — so
 * Gemini uses a `?key=` query param over the global WS, while OpenAI uses the
 * `ws` library here, which can.
 *
 * Wire protocol reference: OpenAI Realtime API (server WS mode).
 *   - session.update (client → server): modalities, voice, instructions, manual VAD
 *   - input_audio_buffer.append (client → server, PCM16 24kHz base64)
 *   - input_audio_buffer.commit + response.create (end of utterance)
 *   - response.audio.delta (server → client, PCM16 24kHz base64)
 *   - response.done (server → client)
 *   - response.cancel + input_audio_buffer.clear (barge-in)
 */

import { WebSocket } from 'ws';

const DEFAULT_REALTIME_ENDPOINT = 'wss://api.openai.com/v1/realtime';

/** OpenAI Realtime input sample rate (mono PCM16) — differs from Gemini's 16kHz. */
export const OPENAI_REALTIME_INPUT_RATE = 24000;
/** OpenAI Realtime output sample rate (mono PCM16). */
export const OPENAI_REALTIME_OUTPUT_RATE = 24000;

/**
 * Normalized, CUMULATIVE per-session token usage emitted by a realtime
 * provider adapter (the basis for token-exact billing in the relay).
 *
 * Both adapters report CUMULATIVE session totals so the relay can settle on the
 * single LATEST snapshot — no per-turn delta bookkeeping:
 *   - OpenAI's `response.done.usage` is per-RESPONSE, so the adapter sums each
 *     response into a running total before emitting.
 *   - Gemini's `usageMetadata` is already cumulative per session, so the
 *     adapter emits the latest parsed snapshot as-is.
 *
 * `cached*` are a SUBSET of the matching input bucket (audio/text) that the
 * provider served from its context cache and bills at a discount; the relay
 * subtracts them from the full-rate input and bills them at the cached rate.
 */
export interface RealtimeUsage {
  /** Cumulative input audio tokens (mic → provider), incl. cached. */
  audioInputTokens: number;
  /** Cumulative output audio tokens (provider speech). */
  audioOutputTokens: number;
  /** Cumulative input text tokens (system prompt / transcripts), incl. cached. */
  textInputTokens: number;
  /** Cumulative output text tokens (model text). */
  textOutputTokens: number;
  /** Cumulative cached input audio tokens (subset of `audioInputTokens`). */
  cachedAudioInputTokens: number;
  /** Cumulative cached input text tokens (subset of `textInputTokens`). */
  cachedTextInputTokens: number;
}

/**
 * Provider-agnostic realtime voice session. Both `openGeminiLiveSession` and
 * `openOpenAIRealtimeSession` return this shape, so the transport skill bridges
 * either provider through one code path.
 */
export interface RealtimeUpstreamSession {
  /** Stable session id (provider-side handle is internal). */
  readonly sessionId: string;
  /** Push a chunk of mic audio (PCM16 mono, raw bytes, at the provider input rate). */
  sendAudio(pcm16: Uint8Array): void;
  /**
   * Send a finished user TEXT turn (text-in modality — the client did its own
   * STT) and request a response. No-op for providers that don't accept text.
   */
  sendText(text: string): void;
  /** Signal the start of a user utterance (push-to-talk press). */
  sendActivityStart(): void;
  /** Signal the end of a user utterance (push-to-talk release). */
  sendActivityEnd(): void;
  /** Barge-in: interrupt the model's current turn. */
  interrupt(): void;
  /** Close the provider session. */
  close(): void;
  /** Whether the provider WS is open and ready for audio. */
  isReady(): boolean;
}

export interface OpenAIRealtimeSessionOptions {
  apiKey: string;
  /** Provider model name, e.g. `gpt-realtime`. */
  model: string;
  /** Prebuilt voice name, e.g. `alloy`, `marin`, `cedar`. */
  voiceId: string;
  systemPrompt: string;
  /**
   * Inbound model audio: RAW PCM16 mono base64 at `sampleRate` (24kHz).
   * Consumers convert it to AudioBuffer samples synchronously and schedule
   * gaplessly (do NOT round-trip through `decodeAudioData` per chunk).
   */
  onAudioChunk: (pcmBase64: string, sampleRate: number) => void;
  onTurnComplete?: () => void;
  onInterrupted?: () => void;
  onText?: (text: string) => void;
  /**
   * Whether the provider should RESPOND with audio (default true). When false
   * the session is text-out only (`modalities:['text']`) — the client speaks
   * the streamed `onText` with its own TTS.
   */
  responseAudio?: boolean;
  onError?: (err: Error) => void;
  onReady?: () => void;
  /**
   * Cumulative session token usage, emitted whenever the provider reports it
   * (OpenAI: each `response.done`). Drives token-exact billing in the relay.
   */
  onUsage?: (usage: RealtimeUsage) => void;
  /** Override the WS endpoint (tests). */
  endpoint?: string;
  /** Inject a `ws`-compatible implementation (tests). Defaults to the `ws` lib. */
  webSocketImpl?: typeof WebSocket;
}

/** Coerce an unknown JSON field to a finite, non-negative token count. */
function tokenNum(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : 0;
}

function toBase64(bytes: Uint8Array): string {
  if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

/**
 * Open an OpenAI Realtime voice session. Returns immediately; the session is
 * usable once `onReady` fires (right after the initial `session.update`).
 */
export function openOpenAIRealtimeSession(
  opts: OpenAIRealtimeSessionOptions,
): RealtimeUpstreamSession {
  const sessionId =
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `oa_${Date.now()}_${Math.random().toString(36).slice(2)}`;

  const WS = opts.webSocketImpl ?? WebSocket;
  const endpoint = opts.endpoint ?? DEFAULT_REALTIME_ENDPOINT;
  const url = `${endpoint}?model=${encodeURIComponent(opts.model)}`;

  let ready = false;
  let closed = false;
  let audioChunkCount = 0;
  // Frames enqueued before the session is ready (e.g. eager audio).
  const preReadyQueue: string[] = [];

  // Running CUMULATIVE usage. OpenAI bills per RESPONSE (each `response.done`
  // carries that response's usage, and re-billing conversation context each
  // turn is intentional on their side), so we SUM each response into the total
  // and emit the cumulative snapshot the relay settles on.
  const usage: RealtimeUsage = {
    audioInputTokens: 0,
    audioOutputTokens: 0,
    textInputTokens: 0,
    textOutputTokens: 0,
    cachedAudioInputTokens: 0,
    cachedTextInputTokens: 0,
  };
  const accumulateUsage = (u: Record<string, unknown> | undefined) => {
    if (!u || !opts.onUsage) return;
    const itd = u.input_token_details as Record<string, unknown> | undefined;
    const otd = u.output_token_details as Record<string, unknown> | undefined;
    // `cached_tokens_details` (audio/text split of the cached input) may sit at
    // the usage top level or nested under input_token_details by version.
    const ctd = (u.cached_tokens_details ??
      itd?.cached_tokens_details) as Record<string, unknown> | undefined;
    if (itd) {
      usage.audioInputTokens += tokenNum(itd.audio_tokens);
      usage.textInputTokens += tokenNum(itd.text_tokens);
    } else {
      // No modality breakdown (some deployments omit it) — a realtime turn's
      // input is overwhelmingly audio, so attribute the lot to audio-in.
      usage.audioInputTokens += tokenNum(u.input_tokens);
    }
    if (otd) {
      usage.audioOutputTokens += tokenNum(otd.audio_tokens);
      usage.textOutputTokens += tokenNum(otd.text_tokens);
    } else {
      usage.audioOutputTokens += tokenNum(u.output_tokens);
    }
    if (ctd) {
      usage.cachedAudioInputTokens += tokenNum(ctd.audio_tokens);
      usage.cachedTextInputTokens += tokenNum(ctd.text_tokens);
    }
    opts.onUsage({ ...usage });
  };

  // GA Realtime API (`/v1/realtime`). The old `OpenAI-Beta: realtime=v1`
  // header opts into the RETIRED beta interface ("The Realtime Beta API is no
  // longer supported") — GA is the default, so we send no beta header.
  const ws = new WS(url, {
    headers: {
      Authorization: `Bearer ${opts.apiKey}`,
    },
  });

  const tag = `[openai-realtime ${sessionId.slice(0, 8)}]`;

  const sendRaw = (obj: unknown) => {
    const json = JSON.stringify(obj);
    if (!ready) {
      preReadyQueue.push(json);
      return;
    }
    try {
      ws.send(json);
    } catch (err) {
      opts.onError?.(err instanceof Error ? err : new Error(String(err)));
    }
  };

  console.log(`${tag} connecting model=${opts.model}`);

  ws.on('open', () => {
    console.log(`${tag} ws open → session.update`);
    try {
      // First frame: configure the session for manual push-to-talk, in the GA
      // shape — `type:'realtime'`, `output_modalities` (single modality), and
      // audio nested under `session.audio.{input,output}` with the format as an
      // OBJECT (`{type:'audio/pcm', rate}`), not the old `*_audio_format` string.
      ws.send(
        JSON.stringify({
          type: 'session.update',
          session: {
            type: 'realtime',
            model: opts.model,
            // Half-duplex: text-out only when the client speaks the reply itself
            // (responseAudio:false). With `['audio']` GA still streams a
            // transcript via response.output_audio_transcript.delta.
            output_modalities: opts.responseAudio === false ? ['text'] : ['audio'],
            instructions: opts.systemPrompt,
            audio: {
              // Audio in is always accepted (the user speaks). Server VAD off —
              // the widget drives turns (commit + response.create).
              input: {
                format: { type: 'audio/pcm', rate: OPENAI_REALTIME_OUTPUT_RATE },
                turn_detection: null,
              },
              // Output voice only matters for audio replies.
              ...(opts.responseAudio === false
                ? {}
                : {
                    output: {
                      format: { type: 'audio/pcm', rate: OPENAI_REALTIME_OUTPUT_RATE },
                      voice: opts.voiceId,
                    },
                  }),
            },
          },
        }),
      );
      // Unlike Gemini's setupComplete handshake, OpenAI accepts input right
      // after session.update — flush anything queued during connect.
      ready = true;
      for (const frame of preReadyQueue.splice(0)) {
        try {
          ws.send(frame);
        } catch (err) {
          opts.onError?.(err instanceof Error ? err : new Error(String(err)));
        }
      }
      opts.onReady?.();
    } catch (err) {
      opts.onError?.(err instanceof Error ? err : new Error(String(err)));
    }
  });

  ws.on('message', (data: unknown) => {
    let text: string;
    if (typeof data === 'string') text = data;
    else if (typeof Buffer !== 'undefined' && Buffer.isBuffer(data)) text = data.toString('utf8');
    else if (data instanceof ArrayBuffer) text = new TextDecoder().decode(data);
    else if (Array.isArray(data)) text = Buffer.concat(data as Buffer[]).toString('utf8');
    else return;

    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(text);
    } catch {
      return;
    }
    const type = msg.type as string | undefined;
    if (!type) return;

    switch (type) {
      case 'response.audio.delta':
      case 'response.output_audio.delta': {
        const delta = msg.delta as string | undefined;
        if (delta) {
          if (audioChunkCount === 0) console.log(`${tag} first audio chunk from model`);
          audioChunkCount++;
          // Pass the provider's RAW PCM16 (24kHz mono) base64 straight through
          // — same gapless-playback contract as Gemini Live.
          opts.onAudioChunk(delta, OPENAI_REALTIME_OUTPUT_RATE);
        }
        break;
      }
      // GA renamed the streaming text events `response.output_text.delta` +
      // `response.output_audio_transcript.delta`; keep the beta names for any
      // transition window.
      case 'response.output_text.delta':
      case 'response.output_audio_transcript.delta':
      case 'response.audio_transcript.delta':
      case 'response.text.delta': {
        const t = msg.delta as string | undefined;
        if (t) opts.onText?.(t);
        break;
      }
      case 'response.done':
      case 'response.completed': {
        console.log(`${tag} response.done (${audioChunkCount} audio chunk(s))`);
        audioChunkCount = 0;
        const response = msg.response as { usage?: Record<string, unknown> } | undefined;
        accumulateUsage(response?.usage);
        opts.onTurnComplete?.();
        break;
      }
      case 'response.cancelled': {
        opts.onInterrupted?.();
        break;
      }
      case 'error': {
        const e = msg.error as { message?: string } | undefined;
        console.warn(`${tag} error: ${e?.message ?? text.slice(0, 300)}`);
        opts.onError?.(new Error(e?.message ?? 'OpenAI Realtime error'));
        break;
      }
    }
  });

  ws.on('error', (err: Error) => {
    console.error(`${tag} ws error: ${err?.message ?? ''}`);
    opts.onError?.(err instanceof Error ? err : new Error(String(err)));
  });

  ws.on('close', (code: number, reason: Buffer) => {
    console.warn(
      `${tag} ws closed code=${code} reason=${reason?.toString() || '""'} ready=${ready}`,
    );
    closed = true;
    ready = false;
  });

  return {
    sessionId,
    sendAudio(pcm16: Uint8Array) {
      if (closed) return;
      sendRaw({ type: 'input_audio_buffer.append', audio: toBase64(pcm16) });
    },
    sendText(text: string) {
      if (closed || !text) return;
      // Text-in turn (client did its own STT): create a user message item,
      // then request a response.
      sendRaw({
        type: 'conversation.item.create',
        item: { type: 'message', role: 'user', content: [{ type: 'input_text', text }] },
      });
      sendRaw({ type: 'response.create' });
    },
    sendActivityStart() {
      if (closed) return;
      // Clear any stale input so each push-to-talk utterance starts clean.
      sendRaw({ type: 'input_audio_buffer.clear' });
    },
    sendActivityEnd() {
      if (closed) return;
      sendRaw({ type: 'input_audio_buffer.commit' });
      sendRaw({ type: 'response.create' });
    },
    interrupt() {
      if (closed) return;
      // Barge-in: cancel the in-flight response and drop buffered input.
      sendRaw({ type: 'response.cancel' });
      sendRaw({ type: 'input_audio_buffer.clear' });
    },
    close() {
      if (closed) return;
      closed = true;
      try {
        ws.close();
      } catch {
        /* already closing */
      }
    },
    isReady() {
      return ready && !closed;
    },
  };
}
