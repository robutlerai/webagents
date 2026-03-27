/**
 * Browser Skills Module
 * 
 * Skills for browser-specific APIs.
 */

export { WakeLockSkill } from './wakelock';
export { NotificationsSkill as BrowserNotificationsSkill } from './notifications';
export { GeolocationSkill } from './geolocation';
export { StorageSkill } from './storage';
export { CameraSkill } from './camera';
export { MicrophoneSkill } from './microphone';

// Browser Automation (DOM control, screenshots, testing)
export { BrowserAutomationSkill } from './automation';
export type {
  ElementInfo,
  ScreenshotResult,
  NetworkEntry,
  AccessibilityInfo,
} from './automation';

// Web Search (DuckDuckGo, Google, Bing)
export { WebSearchSkill } from './search';
export type {
  SearchResult,
  SearchResponse,
  WebSearchConfig,
} from './search';
