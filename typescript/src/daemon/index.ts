/**
 * Daemon Module
 * 
 * WebAgents daemon for managing agents.
 */

export { AgentRegistry } from './registry';
export type { RegisteredAgent } from './registry';

export { AgentWatcher, agentDefinitionFrom, findAgentFiles } from './watcher';
export type { AgentDefinition } from './watcher';

// The `cron:` schedules of the served agents (plan item 1.7): the runner,
// where results go, and the words both SDKs use for them.
export { ScheduleRunner, HEARTBEAT_PROMPT, HEARTBEAT_SENTINEL, isQuietHeartbeat, nextCronRun, nextRunFor } from './schedule-runner';
export type { RunRecord, ScheduleEntry, ScheduleRunnerOptions } from './schedule-runner';
export { deliver, deliverChat, deliverFile, deliverWebhook, chatSessionId, chatTurn, webhookBody } from './deliver';
export type { ChatClient, DeliveryContext, Outcome, RunResult } from './deliver';

export { WebAgentsDaemon, buildDefinedAgent } from './server';
export type { DaemonConfig } from './server';

export { installService, uninstallService, generateLaunchdPlist, generateSystemdUnit } from './service';
export type { ServiceConfig } from './service';
