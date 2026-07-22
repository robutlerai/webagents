/**
 * Google Gemini Live (BidiGenerateContent) adapter.
 *
 * Distinct from the text adapter in `google.ts`: this is a bidirectional
 * WebSocket transport for realtime voice, not an SSE request/response. It
 * runs server-side (portal / agent runtime) and holds the provider WS for
 * the lifetime of a voice session. The provider API key never leaves this
 * process (see ADR-v3-12).
 *
 * Wire protocol reference: Gemini Live API `BidiGenerateContent`.
 *   - setup (client → server, once)
 *   - setupComplete (server → client)
 *   - realtimeInput.mediaChunks (client → server, PCM16 16kHz)
 *   - realtimeInput.activityStart / activityEnd (manual turn control)
 *   - serverContent.modelTurn.parts[].inlineData (server → client, PCM16 24kHz)
 *   - serverContent.turnComplete / serverContent.interrupted
 */

import type { RealtimeUsage, RealtimeHistoryItem } from './openai-realtime';
import { sanitizeSchemaForGemini } from './google';

const DEFAULT_LIVE_ENDPOINT =
  'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent';

/** Gemini Live input sample rate (mono PCM16). */
export const GEMINI_LIVE_INPUT_RATE = 16000;
/** Gemini Live output sample rate (mono PCM16). */
export const GEMINI_LIVE_OUTPUT_RATE = 24000;

export interface GeminiLiveSessionOptions {
  apiKey: string;
  /** Provider model name, e.g. `gemini-3.1-flash-live-preview`. `models/` prefix added if absent. */
  model: string;
  /** Prebuilt voice name, e.g. `Aoede`, `Puck`, `Charon`, `Kore`, `Fenrir`. */
  voiceId: string;
  systemPrompt: string;
  /**
   * Inbound model audio: RAW PCM16 mono base64 at `sampleRate` (24kHz).
   * Consumers convert it to AudioBuffer samples synchronously and schedule
   * gaplessly (do NOT round-trip through `decodeAudioData` per chunk).
   */
  onAudioChunk: (pcmBase64: string, sampleRate: number) => void;
  /** Fired when the model finishes a turn (serverContent.turnComplete). */
  onTurnComplete?: () => void;
  /** Fired when the model's current turn was interrupted by user barge-in. */
  onInterrupted?: () => void;
  /** Streamed transcript of the AGENT's reply (model text / audio transcript). */
  onText?: (text: string) => void;
  /**
   * Streamed transcript of the USER's speech — the provider's STT of the mic
   * audio (enabled via inputAudioTranscription). Mirrors `onText` for the user
   * side, so the widget can show the user's own words in native-audio mode.
   */
  onUserText?: (text: string) => void;
  /**
   * Prior conversation turns to seed the session BEFORE the first live turn, so
   * the model continues an existing chat. Emitted as a `clientContent` with
   * `turnComplete:false` (context only, no response) right after setupComplete.
   */
  initialHistory?: RealtimeHistoryItem[];
  /**
   * Whether the model should RESPOND with audio (default true). When false the
   * session is text-out only (`responseModalities:['TEXT']`) — the client
   * speaks the streamed `onText` with its own TTS.
   */
  responseAudio?: boolean;
  onError?: (err: Error) => void;
  /** Fired once the provider acknowledges setup. */
  onReady?: () => void;
  /**
   * Cumulative session token usage. Gemini's `usageMetadata` is already
   * cumulative per session, so the adapter emits the latest parsed snapshot.
   * Drives token-exact billing in the relay.
   */
  onUsage?: (usage: RealtimeUsage) => void;
  /**
   * Function tools the model may call (the agent's DECLARED tools — the model
   * can only call these). Each call surfaces via `onFunctionCall`; the caller
   * runs it and replies with `submitToolResult(id, output)`.
   */
  tools?: Array<{ name: string; description?: string; parameters?: unknown }>;
  /** Fired when the model requests a declared tool. `argsJson` is raw JSON. */
  onFunctionCall?: (callId: string, name: string, argsJson: string) => void;
  /** Override the WS endpoint (tests). */
  endpoint?: string;
  /**
   * Inject a WebSocket implementation (tests). Defaults to the global
   * `WebSocket` (Node 22+ / browser).
   */
  webSocketImpl?: typeof WebSocket;
}

export interface GeminiLiveSession {
  /** Stable session id (provider-side handle is internal). */
  readonly sessionId: string;
  /** Push a chunk of mic audio (PCM16 mono 16kHz, raw bytes). */
  sendAudio(pcm16: Uint8Array): void;
  /**
   * Send a finished user TEXT turn (text-in modality — the client did its own
   * STT) and request a response, via Gemini `clientContent`.
   */
  sendText(text: string): void;
  /**
   * Append SILENT context (session priming, ambient game/world state) the
   * model sees on its NEXT turn — `turnComplete: false`, never elicits a
   * response of its own.
   */
  sendContext(text: string): void;
  /** Return a declared tool's result to the model (after `onFunctionCall`). */
  submitToolResult(callId: string, output: string): void;
  /** Signal the start of a user utterance (push-to-talk press). */
  sendActivityStart(): void;
  /** Signal the end of a user utterance (push-to-talk release). */
  sendActivityEnd(): void;
  /** Barge-in: interrupt the model's current turn. */
  interrupt(): void;
  /** Close the provider session. */
  close(): void;
  /** Whether the provider WS is open and setup is complete. */
  isReady(): boolean;
}

function toBase64(bytes: Uint8Array): string {
  if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

/** Coerce an unknown JSON field to a finite, non-negative token count. */
function tokenNum(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : 0;
}

/** Sum the tokenCount of `[{ modality, tokenCount }]` entries for one modality. */
function sumModality(details: unknown, modality: 'AUDIO' | 'TEXT'): number {
  if (!Array.isArray(details)) return 0;
  let sum = 0;
  for (const d of details) {
    const m = String((d as { modality?: unknown })?.modality ?? '').toUpperCase();
    if (m === modality) sum += tokenNum((d as { tokenCount?: unknown }).tokenCount);
  }
  return sum;
}

/**
 * Parse a Gemini Live `usageMetadata` envelope into the normalized cumulative
 * `RealtimeUsage` shape. Gemini reports per-modality `promptTokensDetails` /
 * `responseTokensDetails`; when a modality breakdown is absent we attribute the
 * whole count to audio (a voice session's tokens are overwhelmingly audio).
 */
function parseGeminiUsage(um: Record<string, unknown>): RealtimeUsage {
  let audioIn = sumModality(um.promptTokensDetails, 'AUDIO');
  let textIn = sumModality(um.promptTokensDetails, 'TEXT');
  if (!audioIn && !textIn) audioIn = tokenNum(um.promptTokenCount);

  let audioOut = sumModality(um.responseTokensDetails, 'AUDIO');
  let textOut = sumModality(um.responseTokensDetails, 'TEXT');
  if (!audioOut && !textOut) audioOut = tokenNum(um.responseTokenCount);

  let cachedAudioIn = sumModality(um.cacheTokensDetails, 'AUDIO');
  let cachedTextIn = sumModality(um.cacheTokensDetails, 'TEXT');
  if (!cachedAudioIn && !cachedTextIn) {
    const cached = tokenNum(um.cachedContentTokenCount);
    if (cached) cachedAudioIn = cached; // unsplit cache → attribute to audio
  }

  return {
    audioInputTokens: audioIn,
    audioOutputTokens: audioOut,
    textInputTokens: textIn,
    textOutputTokens: textOut,
    cachedAudioInputTokens: cachedAudioIn,
    cachedTextInputTokens: cachedTextIn,
  };
}

/**
 * Open a Gemini Live voice session. Resolves once the provider WS is
 * connected (not yet setup-complete); use `onReady` to know when the
 * session can accept audio.
 */
export function openGeminiLiveSession(
  opts: GeminiLiveSessionOptions,
): GeminiLiveSession {
  const sessionId =
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `gl_${Date.now()}_${Math.random().toString(36).slice(2)}`;

  const WS = opts.webSocketImpl ?? (globalThis as { WebSocket?: typeof WebSocket }).WebSocket;
  if (!WS) {
    throw new Error('No WebSocket implementation available for Gemini Live');
  }

  const modelName = opts.model.startsWith('models/') ? opts.model : `models/${opts.model}`;
  const endpoint = opts.endpoint ?? DEFAULT_LIVE_ENDPOINT;
  const url = `${endpoint}?key=${encodeURIComponent(opts.apiKey)}`;

  let ready = false;
  let closed = false;
  let audioChunkCount = 0;
  // Queue outbound frames sent before setupComplete (e.g. eager audio).
  const preReadyQueue: string[] = [];

  const ws = new WS(url);
  ws.binaryType = 'arraybuffer';

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

  const tag = `[gemini-live ${sessionId.slice(0, 8)}]`;
  console.log(`${tag} connecting model=${modelName} endpoint=${endpoint}`);

  ws.addEventListener('open', () => {
    console.log(`${tag} ws open → sending setup`);
    // setup must be the very first message — bypass the queue gate.
    // Seed prior conversation as CONTEXT inside the system instruction. Gemini
    // Live rejects `clientContent` history-seeding in a realtime-audio session
    // (close 1007 "Request contains an invalid argument") even with clean
    // alternating user/model turns, so we fold the history into the system
    // prompt instead — reliable, and the model continues the conversation.
    let sysText = opts.systemPrompt;
    if (opts.initialHistory && opts.initialHistory.length) {
      const lines = opts.initialHistory
        .filter((h) => h && h.text)
        .map((h) => `${h.role === 'assistant' ? 'Assistant' : 'User'}: ${h.text}`);
      if (lines.length) {
        sysText +=
          `\n\n--- Recent conversation so far (most recent last). Continue it naturally; do not repeat greetings. ---\n` +
          lines.join('\n');
        console.log(`${tag} seeded ${lines.length} history line(s) into systemInstruction`);
      }
    }
    try {
      ws.send(
        JSON.stringify({
          setup: {
            model: modelName,
            generationConfig: {
              // Text-out only when the client speaks the reply itself.
              responseModalities: opts.responseAudio === false ? ['TEXT'] : ['AUDIO'],
              speechConfig: {
                voiceConfig: {
                  prebuiltVoiceConfig: { voiceName: opts.voiceId },
                },
              },
            },
            // Transcribe BOTH the user's mic audio and the model's spoken reply
            // so the widget shows text bubbles in native-audio mode (part.text
            // is empty when responseModalities is AUDIO). Surfaced via
            // serverContent.inputTranscription / .outputTranscription.
            inputAudioTranscription: {},
            outputAudioTranscription: {},
            systemInstruction: { parts: [{ text: sysText }] },
            // The agent's declared tools → Gemini `functionDeclarations`. The
            // model can only call these (no hallucinated calls).
            ...(opts.tools && opts.tools.length
              ? {
                  tools: [
                    {
                      functionDeclarations: opts.tools.map((t) => ({
                        name: t.name,
                        description: t.description,
                        // Gemini Live's function-declaration Schema is a strict
                        // OpenAPI subset and REJECTS keywords like
                        // `additionalProperties` (closing the WS with code 1007,
                        // killing the whole session). Sanitize each tool's
                        // params — same cleaner the Gemini TEXT adapter uses.
                        parameters: sanitizeSchemaForGemini(
                          t.parameters ?? { type: 'object', properties: {} },
                        ),
                      })),
                    },
                  ],
                }
              : {}),
            // Manual turn control: we drive activityStart/activityEnd from
            // the widget's push-to-talk so a half-duplex UX is deterministic.
            realtimeInputConfig: {
              automaticActivityDetection: { disabled: true },
            },
          },
        }),
      );
    } catch (err) {
      opts.onError?.(err instanceof Error ? err : new Error(String(err)));
    }
  });

  ws.addEventListener('message', (event: MessageEvent) => {
    let text: string;
    if (typeof event.data === 'string') {
      text = event.data;
    } else if (event.data instanceof ArrayBuffer) {
      text = new TextDecoder().decode(event.data);
    } else {
      return;
    }

    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(text);
    } catch {
      return;
    }

    if (msg.setupComplete) {
      console.log(`${tag} setupComplete — flushing ${preReadyQueue.length} queued frame(s)`);
      ready = true;
      for (const frame of preReadyQueue.splice(0)) {
        try {
          ws.send(frame);
        } catch (err) {
          opts.onError?.(err instanceof Error ? err : new Error(String(err)));
        }
      }
      // History is seeded via the systemInstruction at setup (clientContent
      // seeding is rejected by Gemini Live in a realtime-audio session) — see
      // the open handler above. Nothing to send here.
      opts.onReady?.();
      return;
    }

    // Cumulative token usage rides alongside serverContent (and sometimes in a
    // standalone trailing message). Gemini's usageMetadata is already a running
    // session total, so emit the latest snapshot for the relay to settle on.
    const usageMetadata = msg.usageMetadata as Record<string, unknown> | undefined;
    if (usageMetadata && opts.onUsage) {
      opts.onUsage(parseGeminiUsage(usageMetadata));
    }

    // Tool (function) calling: the model requests one or more DECLARED tools.
    const toolCall = msg.toolCall as { functionCalls?: Array<{ name?: string; args?: unknown; id?: string }> } | undefined;
    if (toolCall?.functionCalls?.length) {
      for (const fc of toolCall.functionCalls) {
        const callId = fc.id ?? fc.name ?? '';
        if (fc.name) {
          console.log(`${tag} function_call ${fc.name}`);
          opts.onFunctionCall?.(callId, fc.name, JSON.stringify(fc.args ?? {}));
        }
      }
      return;
    }

    // Anything that isn't setupComplete, serverContent, a usage envelope or a
    // toolCall is unexpected — most often a setup-rejection / goAway / error.
    // Log it (truncated) so a silent "no response" is diagnosable.
    if (!msg.serverContent && !usageMetadata && !toolCall) {
      console.warn(`${tag} non-content message: ${text.slice(0, 400)}`);
    }

    const serverContent = msg.serverContent as
      | {
          modelTurn?: { parts?: Array<Record<string, unknown>> };
          turnComplete?: boolean;
          interrupted?: boolean;
          inputTranscription?: { text?: string };
          outputTranscription?: { text?: string };
        }
      | undefined;

    if (serverContent) {
      // Native-audio transcripts (enabled via input/outputAudioTranscription).
      // In AUDIO mode part.text is empty, so these are the only text source;
      // in TEXT mode no audio is produced, so outputTranscription stays empty
      // and the model text arrives via part.text below (no double-forward).
      const inT = serverContent.inputTranscription?.text;
      if (inT) opts.onUserText?.(inT);
      const outT = serverContent.outputTranscription?.text;
      if (outT) opts.onText?.(outT);

      const parts = serverContent.modelTurn?.parts ?? [];
      for (const part of parts) {
        const inlineData = part.inlineData as
          | { mimeType?: string; data?: string }
          | undefined;
        if (inlineData?.data) {
          if (audioChunkCount === 0) console.log(`${tag} first audio chunk from model`);
          audioChunkCount++;
          // Pass the provider's RAW PCM16 (24kHz mono) base64 straight
          // through. The widget converts it to AudioBuffer samples
          // synchronously and schedules gaplessly — far smoother than
          // per-chunk WAV + async decodeAudioData, which reorders chunks
          // and resamples each one independently (audible clicks/gaps).
          opts.onAudioChunk(inlineData.data, GEMINI_LIVE_OUTPUT_RATE);
        }
        const partText = part.text as string | undefined;
        if (partText) opts.onText?.(partText);
      }
      if (serverContent.interrupted) opts.onInterrupted?.();
      if (serverContent.turnComplete) {
        console.log(`${tag} turnComplete (${audioChunkCount} audio chunk(s))`);
        audioChunkCount = 0;
        opts.onTurnComplete?.();
      }
    }
  });

  ws.addEventListener('error', (ev: Event) => {
    const m = (ev as unknown as { message?: string }).message;
    console.error(`${tag} ws error${m ? `: ${m}` : ''}`);
    opts.onError?.(new Error('Gemini Live WebSocket error'));
  });

  ws.addEventListener('close', (ev: CloseEvent) => {
    // `close()` sets `closed` BEFORE closing the socket, so `closed === false`
    // here means the SERVER hung up on us. Those closes carry the real reason
    // (e.g. 1007 "The requested combination of response modalities (TEXT) is
    // not supported by the model") and used to be swallowed — the caller then
    // sat until its own 60s timeout and reported a meaningless "timed out"
    // instead of the actual cause. Surface it immediately.
    const serverInitiated = !closed && ev?.code !== 1000;
    console.warn(`${tag} ws closed code=${ev?.code} reason=${ev?.reason || '""'} ready=${ready}`);
    closed = true;
    ready = false;
    if (serverInitiated) {
      const why = ev?.reason ? `: ${ev.reason}` : '';
      opts.onError?.(new Error(`Gemini Live closed (code ${ev?.code})${why}`));
    }
  });

  return {
    sessionId,
    sendAudio(pcm16: Uint8Array) {
      if (closed) return;
      // Gemini 3.1 Live deprecated `realtimeInput.mediaChunks` (array) in
      // favour of a single `realtimeInput.audio` Blob (close code 1007
      // otherwise: "realtime_input.media_chunks is deprecated").
      sendRaw({
        realtimeInput: {
          audio: {
            mimeType: `audio/pcm;rate=${GEMINI_LIVE_INPUT_RATE}`,
            data: toBase64(pcm16),
          },
        },
      });
    },
    sendText(text: string) {
      if (closed || !text) return;
      // Text-in turn (client did its own STT). A complete clientContent turn
      // triggers the model to respond.
      sendRaw({
        clientContent: {
          turns: [{ role: 'user', parts: [{ text }] }],
          turnComplete: true,
        },
      });
    },
    sendContext(text: string) {
      if (closed || !text) return;
      // SILENT context (session priming, ambient game/world state):
      // `turnComplete: false` appends to the conversation WITHOUT eliciting a
      // reply — the model sees it on its next turn. Gemini clientContent has
      // no system role, so the text itself should carry its framing.
      sendRaw({
        clientContent: {
          turns: [{ role: 'user', parts: [{ text }] }],
          turnComplete: false,
        },
      });
    },
    submitToolResult(callId: string, output: string) {
      if (closed) return;
      // Return the tool result; the model continues the turn with it. Gemini
      // wants the parsed value under `response.result`.
      let parsed: unknown = output;
      try { parsed = JSON.parse(output); } catch { /* keep as string */ }
      sendRaw({
        toolResponse: {
          functionResponses: [{ id: callId, response: { result: parsed } }],
        },
      });
    },
    sendActivityStart() {
      if (closed) return;
      sendRaw({ realtimeInput: { activityStart: {} } });
    },
    sendActivityEnd() {
      if (closed) return;
      sendRaw({ realtimeInput: { activityEnd: {} } });
    },
    interrupt() {
      if (closed) return;
      // Barge-in under manual activity control: opening a fresh activity
      // tells the model the user is speaking again, which cancels the
      // in-flight model turn. The widget also stops local playback.
      sendRaw({ realtimeInput: { activityStart: {} } });
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
