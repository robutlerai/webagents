/**
 * Transport Skills Module
 * 
 * Skills that handle different transport protocols.
 */

export { CompletionsTransportSkill } from './completions/index';
export type { CompletionsTransportConfig } from './completions/index';

export { PortalTransportSkill } from './portal/index';
export type { PortalTransportConfig } from './portal/index';

// PortalConnectSkill — the reverse WS bridge as a skill. This is what
// replaced the `connect(agent)` wrapper: attach it and serve the agent.
export { PortalConnectSkill } from './portal-connect/index';
export type { PortalConnectConfig } from './portal-connect/index';

export { UAMPTransportSkill } from './uamp/index';
export type { UAMPTransportConfig } from './uamp/index';

export { RealtimeTransportSkill } from './realtime/index';
export type { RealtimeTransportConfig, RealtimeUsage } from './realtime/index';

// A2A v1.0 (plan item 1.3, 2026-09-26): the transport, the card and its
// signature, the canonicaliser, the error table and the client
// (`./a2a/a2a-client`, also `A2ATransportSkill.callPeer`).
export {
  A2AClientError,
  a2aCallAgent,
  fetchAgentCard,
  pickInterface,
  A2ATransportSkill,
  resolveA2ASettings,
  peerTokenFor,
  AGENT_CARD_WELL_KNOWN_SUFFIX,
  A2A_RPC_SUBPATH,
  buildA2AAgentCard,
  cardSigningPayload,
  cardSigningBytes,
  signAgentCard,
  verifyAgentCard,
  identityCardSigner,
  hmacCardSigner,
  skillsFromTools,
  canonicalize,
  A2AError,
  A2A_ERRORS,
  A2A_VERSION,
  A2A_VERSION_HEADER,
  a2aReplyText,
} from './a2a/index';
export type {
  A2ACallOptions,
  A2ACallResult,
  A2ATransportConfig,
  A2ASettings,
  A2APeer,
  AgentCardV1,
  AgentInterface,
  AgentSkill,
  AgentProvider,
  AgentCardSignature,
  CardSigner,
  VerifyCardOptions,
  VerifyCardResult,
  Task,
  TaskState,
  TaskStatus,
  Message as A2AMessage,
  Part as A2APart,
  Artifact as A2AArtifact,
  StreamResponse as A2AStreamResponse,
} from './a2a/index';

// The Agent Client Protocol (plan item 1.6, 2026-09-26): the agent a code
// editor spawns over stdio, what `webagents acp` serves. The homegrown "Agent
// Commerce Protocol" stub that carried this name is retired (S-249).
export { ACPTransportSkill, AcpConnection, AcpSession, SessionStore, AcpError } from './acp/index';
export type { ACPTransportConfig, AcpSettings, AcpSessionRecord, SessionMessage } from './acp/index';
