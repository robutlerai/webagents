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

export { A2ATransportSkill } from './a2a/index';
export type { A2ATransportConfig, AgentCard, A2ATask, A2ATaskStatus } from './a2a/index';

export { ACPTransportSkill } from './acp/index';
export type { ACPTransportConfig, ACPService, ACPReceipt } from './acp/index';
