/**
 * RealtimeTransportSkill re-export (Plan v3-03).
 *
 * The realtime transport skill already exists at
 * `webagents/typescript/src/skills/transport/realtime/skill.ts` —
 * Plan 3 reuses it for Mode 2 voice flows rather than reimplementing a
 * second copy under `skills/voice/`. This file documents that
 * relationship and provides a stable import path
 * (`skills/voice/realtime-transport-skill`) for Plan 3 consumers so
 * the voice barrel re-export keeps a flat, voice-shaped surface.
 *
 * If a future revision needs a voice-specific wrapping (e.g. provider
 * authentication + billing-token plumbing that doesn't belong in the
 * generic transport skill), wrap it here instead of forking
 * `transport/realtime/`.
 */

export {
  RealtimeTransportSkill,
  type RealtimeTransportConfig,
  type RealtimeProviderConfig,
} from '../transport/realtime/index';
