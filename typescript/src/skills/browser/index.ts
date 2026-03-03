/**
 * Browser Skills Module
 * 
 * Skills for browser-specific APIs.
 */

export { WakeLockSkill } from './wakelock.js';
export { NotificationsSkill } from './notifications.js';
export { GeolocationSkill } from './geolocation.js';
export { StorageSkill } from './storage.js';
export { CameraSkill } from './camera.js';
export { MicrophoneSkill } from './microphone.js';

// Browser Automation (DOM control, screenshots, testing)
export { BrowserAutomationSkill } from './automation.js';
export type {
  ElementInfo,
  ScreenshotResult,
  NetworkEntry,
  AccessibilityInfo,
} from './automation.js';

// Web Search (DuckDuckGo, Google, Bing)
export { WebSearchSkill } from './search.js';
export type {
  SearchResult,
  SearchResponse,
  WebSearchConfig,
} from './search.js';
