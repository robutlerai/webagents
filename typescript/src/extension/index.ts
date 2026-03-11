export { ExtensionAgentRuntime } from './background/agent-runtime.js';
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
} from './background/browser-tools.js';
export type {
  BrowserToolName,
  BrowserToolResult,
} from './background/browser-tools.js';
export type {
  ExtensionConfig,
  AgentStatus,
  TaskRecord,
  ContentScriptMessage,
  BackgroundMessage,
} from './shared/types.js';
export { DEFAULT_CONFIG } from './shared/types.js';
export { loadConfig, saveConfig, clearConfig } from './shared/storage.js';
