/**
 * Voice skill barrel (Plan v3-03).
 *
 * Two skills + the shared metadata types. The realtime transport is
 * NOT re-exported here — it lives at `skills/transport/realtime/` and
 * the top-level `skills/index.ts` exports it under the
 * `Transport Skills` group. Re-exporting it from this barrel would
 * collide with the transport barrel at the top-level `export *` site.
 * Consumers that want a voice-shaped import can pull it directly:
 *
 *   import { RealtimeTransportSkill } from 'webagents/skills/voice/realtime-transport-skill';
 *
 * The path-level re-export at `realtime-transport-skill.ts` is the
 * stable voice-shaped entry point.
 */

export {
  RealtimeLLMSkill,
  type RealtimeLLMConfig,
  type RealtimeUpstreamSpec,
} from './realtime-llm-skill';

export {
  type VoiceAgentMetadata,
  type VoiceMode,
  type VoiceProvider,
  type VoiceTransport,
  type VoiceAcl,
  type OnDeviceBootstrapAllowlist,
  pickOnDeviceBootstrap,
  isCredentialShapedKey,
  CREDENTIAL_FIELD_PATTERN,
  assertVoiceAgentMetadata,
} from './types';
