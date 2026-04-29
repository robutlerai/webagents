export interface ExtensionConfig {
  portalUrl: string;
  sessionToken: string | null;
  extensionToken: string | null;
  agentToken: string | null;
  username: string | null;
  agentName: string | null;
  agentId: string | null;
  llmMode: 'cloud' | 'local' | 'hybrid';
  cloudModel: string;
  localModel: string;
  hybridThreshold: number;
  maxToolCallsPerMinute: number;
  requireApproval: 'always' | 'sensitive' | 'never';
  trustedAgents: string[];
  acceptFrom: string[];
  blockedDomains: string[];
  allowJavascriptEvaluation: boolean;
}

export interface AgentStatus {
  connected: boolean;
  registered: boolean;
  agentName: string | null;
  uptime: number;
  taskCount: number;
  lastActivity: number | null;
  llmMode: 'cloud' | 'local' | 'hybrid';
  localModelLoaded: boolean;
}

export interface TaskRecord {
  id: string;
  source: string;
  instruction: string;
  status: 'running' | 'completed' | 'failed';
  startedAt: number;
  completedAt: number | null;
  result: string | null;
}

export type ContentScriptMessage =
  | { type: 'READ_PAGE'; tabId?: number }
  | { type: 'CLICK'; selector: string }
  | { type: 'FILL'; selector: string; value: string }
  | { type: 'PRESS_KEY'; key: string }
  | { type: 'SELECT_OPTION'; selector: string; value: string }
  | { type: 'SCREENSHOT'; tabId?: number; fullPage?: boolean }
  | { type: 'NAVIGATE'; url: string }
  | { type: 'GET_PAGE_INFO' }
  | { type: 'WAIT_FOR'; selector?: string; timeoutMs?: number }
  | { type: 'EXECUTE_SCRIPT'; code: string }
  | { type: 'READ_PAGE_RESULT'; text: string; html: string }
  | { type: 'CLICK_RESULT'; success: boolean; error?: string }
  | { type: 'FILL_RESULT'; success: boolean; error?: string }
  | { type: 'PRESS_KEY_RESULT'; success: boolean; error?: string }
  | { type: 'SELECT_OPTION_RESULT'; success: boolean; error?: string }
  | { type: 'SCREENSHOT_RESULT'; dataUrl: string }
  | { type: 'PAGE_INFO_RESULT'; url: string; title: string; meta: Record<string, string> }
  | { type: 'WAIT_FOR_RESULT'; success: boolean; error?: string }
  | { type: 'EXECUTE_SCRIPT_RESULT'; result: unknown; error?: string };

export type BackgroundMessage =
  | { type: 'GET_STATUS' }
  | { type: 'GET_CONFIG' }
  | { type: 'SET_CONFIG'; config: Partial<ExtensionConfig> }
  | { type: 'GET_TASKS' }
  | { type: 'LOGIN' }
  | { type: 'LOGOUT' }
  | { type: 'CONNECT' }
  | { type: 'DISCONNECT' }
  | { type: 'SEND_CHAT'; message: string }
  | { type: 'STATUS_RESPONSE'; status: AgentStatus }
  | { type: 'CONFIG_RESPONSE'; config: ExtensionConfig }
  | { type: 'TASKS_RESPONSE'; tasks: TaskRecord[] }
  | { type: 'CHAT_RESPONSE'; response: string }
  | { type: 'ERROR'; error: string };

declare const __ROBUTLER_EXTENSION_PORTAL_URL__: string | undefined;

function defaultPortalUrl(): string {
  if (typeof __ROBUTLER_EXTENSION_PORTAL_URL__ !== 'undefined' && __ROBUTLER_EXTENSION_PORTAL_URL__) {
    return __ROBUTLER_EXTENSION_PORTAL_URL__;
  }
  return 'https://robutler.ai';
}

export const DEFAULT_CONFIG: ExtensionConfig = {
  portalUrl: defaultPortalUrl(),
  sessionToken: null,
  extensionToken: null,
  agentToken: null,
  username: null,
  agentName: null,
  agentId: null,
  llmMode: 'cloud',
  cloudModel: 'gpt-4o',
  localModel: 'Llama-3.2-3B-Instruct-q4f16_1-MLC',
  hybridThreshold: 0.6,
  maxToolCallsPerMinute: 30,
  requireApproval: 'sensitive',
  trustedAgents: [],
  acceptFrom: [],
  blockedDomains: [],
  allowJavascriptEvaluation: false,
};
