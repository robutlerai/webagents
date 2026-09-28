// The Agent Client Protocol transport (plan item 1.6, 2026-09-26): the agent
// a code editor spawns over stdio. The "Agent Commerce Protocol" stub that
// carried this name is retired (S-249).
export { ACPTransportSkill, AcpConnection, AcpSession, SessionStore } from './skill';
export type { ACPTransportConfig, AcpSettings, AcpSessionRecord, SessionMessage } from './skill';
export {
  AcpError,
  AGENT_CAPABILITIES as ACP_AGENT_CAPABILITIES,
  AUTH_METHODS as ACP_AUTH_METHODS,
  PROTOCOL_VERSION as ACP_PROTOCOL_VERSION,
  toolKind as acpToolKind,
  needsPermission as acpNeedsPermission,
} from './protocol';
export type { ToolKind as AcpToolKind, PlanEntry as AcpPlanEntry } from './protocol';
