export { ExtensionAgentRuntime } from './background/agent-runtime';
export {
  BROWSER_TOOL_DEFINITIONS,
  executeBrowserTool,
  listTabs,
  readPage,
  clickElement,
  fillInput,
  screenshot,
  navigate,
  getPageInfo,
  executeScript,
} from './background/browser-tools';
export type {
  BrowserToolName,
  BrowserToolResult,
} from './background/browser-tools';
export type {
  ExtensionConfig,
  AgentStatus,
  TaskRecord,
  ContentScriptMessage,
  BackgroundMessage,
} from './shared/types';
export { DEFAULT_CONFIG } from './shared/types';
export { loadConfig, saveConfig, clearConfig } from './shared/storage';
