/**
 * Voice agent metadata types + validators (Plan v3-03).
 *
 * `agent_configs.metadata.voice.*` is the canonical schema for a
 * voice-capable agent. Two modes are supported:
 *
 *   - `on-device` (Mode 1) — widget downloads + runs models in-browser
 *     via WebGPU. The portal proxies NOTHING after page load. CRITICAL:
 *     Mode 1 metadata MUST NOT carry credentials — the widget runs in
 *     the untrusted `sandbox.<apex>` sandbox origin and anything
 *     leaked here is leaked to every cohabitating widget. Per ADR-v3-12
 *     we enforce this with both a write-time validator (rejecting any
 *     field key matching `/key|secret|token/i`) and a compile-time
 *     allowlist on the dispatcher output (the bootstrap object passed
 *     to the iframe is constructed by explicit destructure).
 *
 *   - `realtime-llm` (Mode 2) — portal authenticates with the realtime
 *     provider (OpenAI Realtime or Gemini Live) using a portal-held
 *     API key and proxies the audio stream to the widget. The
 *     `providerSessionId` is the ONLY provider-side handle propagated;
 *     ephemeral provider tokens never reach the widget.
 *
 * The zod schema lives in `lib/agents/voice-metadata-validator.ts`
 * (portal-only — webagents stays zod-free). This file ships only the
 * TypeScript types so both halves of the system can agree on shape.
 */

export type VoiceMode = 'on-device' | 'realtime-llm';

export type VoiceProvider = 'openai' | 'gemini';

export type VoiceTransport = 'portal-relay' | 'webrtc';

export type VoiceAcl =
  | 'public'
  | 'owner-only'
  | { whitelist: string[] };

/**
 * Canonical `metadata.voice.*` shape. Plan 2's migration 0028 owns the
 * `agent_configs.metadata` jsonb column. Plan 3 is the second writer
 * (`metadata.voice` namespace).
 *
 * IMPORTANT: For `mode === 'on-device'` this object is the *complete*
 * source of allowed iframe-bootstrap keys. The dispatcher constructs
 * the bootstrap object by explicit field destructure (see
 * `lib/widgets/voice-dispatch-handler.ts`) so any future addition to
 * this shape must be reviewed for credential-isolation impact.
 */
export interface VoiceAgentMetadata {
  /** Mode 1 vs Mode 2 — drives the dispatcher branch. */
  mode: VoiceMode;
  /** Provider for Mode 2. Unused for Mode 1. */
  provider?: VoiceProvider;
  /**
   * Model identifier. For Mode 1: an `onnx-community/*` Hugging Face
   * model id. For Mode 2: a provider-side model name (e.g.
   * `gpt-4o-realtime-preview`).
   */
  model: string;
  /** Voice identifier — provider-side voice name OR a Kokoro voice id. */
  voiceId: string;
  /** System prompt — the agent's persona / instructions. */
  systemPrompt: string;
  /** Transports the agent may serve. Mode 1 leaves this empty. */
  allowedTransports: VoiceTransport[];
  /** Caller ACL — checked BEFORE any credential fetch (V5 invariant). */
  acl: VoiceAcl;
}

/**
 * Subset of `VoiceAgentMetadata` that is safe to expose to a Mode 1
 * iframe bootstrap. The dispatcher MUST construct its `iframeBootstrap`
 * from exactly these keys — any new key added here is subject to a
 * security review. This is the compile-time half of the V4
 * credential-isolation invariant; the write-time validator in
 * `lib/agents/voice-metadata-validator.ts` is the runtime half.
 */
export type OnDeviceBootstrapAllowlist = Pick<
  VoiceAgentMetadata,
  'model' | 'voiceId' | 'systemPrompt'
>;

/**
 * Extracts the explicit allowlist into the bootstrap payload. By using
 * explicit destructure (rather than `Object.assign` or rest spread) we
 * guarantee that no extra metadata field — credential-shaped or
 * otherwise — can leak into the iframe even if the input object was
 * tampered with at runtime.
 */
export function pickOnDeviceBootstrap(
  meta: VoiceAgentMetadata,
): OnDeviceBootstrapAllowlist {
  return {
    model: meta.model,
    voiceId: meta.voiceId,
    systemPrompt: meta.systemPrompt,
  };
}

/**
 * Reject obvious credential-shaped keys. Centralised so both the
 * webagents-side skill author guidance and the portal-side write
 * validator share one regex.
 */
export const CREDENTIAL_FIELD_PATTERN = /key|secret|token|password|cred/i;

export function isCredentialShapedKey(key: string): boolean {
  return CREDENTIAL_FIELD_PATTERN.test(key);
}

/**
 * Hand-rolled (no zod) shape check used by webagents-side runtime
 * code that wants a defensive guard without pulling in zod. The
 * canonical validator is the zod one in
 * `lib/agents/voice-metadata-validator.ts` — that's the write-time
 * gate. This is a defence-in-depth check on the read path.
 */
export function assertVoiceAgentMetadata(
  input: unknown,
): asserts input is VoiceAgentMetadata {
  if (!input || typeof input !== 'object') {
    throw new Error('voice metadata: expected object');
  }
  const m = input as Record<string, unknown>;
  if (m.mode !== 'on-device' && m.mode !== 'realtime-llm') {
    throw new Error('voice metadata: mode must be "on-device" or "realtime-llm"');
  }
  if (typeof m.model !== 'string' || m.model.length === 0) {
    throw new Error('voice metadata: model is required');
  }
  if (typeof m.voiceId !== 'string' || m.voiceId.length === 0) {
    throw new Error('voice metadata: voiceId is required');
  }
  if (typeof m.systemPrompt !== 'string') {
    throw new Error('voice metadata: systemPrompt is required');
  }
  if (!Array.isArray(m.allowedTransports)) {
    throw new Error('voice metadata: allowedTransports must be an array');
  }
  // Mode-1 credential-isolation defence-in-depth.
  if (m.mode === 'on-device') {
    for (const key of Object.keys(m)) {
      if (isCredentialShapedKey(key)) {
        throw new Error(
          `voice metadata (on-device): field "${key}" looks credential-shaped and is forbidden`,
        );
      }
    }
  }
}
