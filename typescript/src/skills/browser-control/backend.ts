/**
 * BrowserBackend — adapter contract consumed by `BrowserControlSkill`.
 *
 * Plan v3-02 (Browser control): one unified skill, three backends. The
 * skill never speaks to a provider directly — it dispatches verb calls
 * (`navigate`, `click`, `type`, ...) to a backend that implements this
 * interface. Backends:
 *
 *   - `BrowserbaseBackend`   — backends/browserbase.ts
 *   - `BrowserUseBackend`    — backends/browser-use.ts
 *   - `ChromeBrowserBackend` — backends/chrome.ts (extension WS bridge)
 *
 * The legacy `BrowserControlSkill` at `../browser/skill.ts` stays for
 * backwards compatibility; this module is a SIBLING (not a replacement).
 * The legacy one drives tab-level operations on a connected Chrome
 * extension via the `BrowserControlAdapter` interface — different shape.
 *
 * ADR-v3-10 captures the rationale + verb set.
 */

/** Result returned by every backend verb call. */
export interface BackendActionResult {
  success: boolean;
  /** Provider-specific payload (e.g. screenshot bytes, extracted text). */
  data?: unknown;
  /** Set when `success === false`. */
  error?: string;
}

/** Returned from `backend.open()`. */
export interface SessionHandle {
  /** Provider-side session identifier (browserbase session id, browser-use job id, extension session uuid). */
  providerSessionId: string;
  /**
   * URL to the provider's polished live view, when the backend serves
   * one (browserbase / browser-use). For extension sessions this is
   * `undefined` — the viewer widget composes the WebRTC stream.
   */
  liveViewUrl?: string;
  /**
   * Provider's enforced session TTL in seconds. `null` for extension
   * sessions (user-controlled, no TTL); finite for cloud providers.
   */
  providerSessionTtlSeconds: number | null;
  /** Which Plan 1 live transport this session emits. */
  liveTransport: 'iframe-url' | 'webrtc';
}

/**
 * Backend contract — implemented by each of the three backends.
 *
 * Verb naming matches the skill's tool surface (without the `browser_`
 * prefix). Every verb returns a `BackendActionResult` — success/failure
 * is explicit so the skill can convert to `tool_progress` envelopes
 * without throwing in the agent loop.
 */
export interface BrowserBackend {
  /** Allocate a provider session. Called by `BrowserControlSkill.onActivate`. */
  open(args?: { initialUrl?: string }): Promise<SessionHandle>;

  /** Tear down the session. Called by `BrowserControlSkill.onDeactivate`. Idempotent. */
  close(): Promise<BackendActionResult>;

  navigate(url: string): Promise<BackendActionResult>;
  click(target: string | { x: number; y: number }): Promise<BackendActionResult>;
  type(text: string, selector?: string): Promise<BackendActionResult>;
  screenshot(): Promise<BackendActionResult>;
  scroll(direction: 'up' | 'down' | 'left' | 'right', amount?: number): Promise<BackendActionResult>;
  wait(args: { selector?: string; timeoutMs?: number }): Promise<BackendActionResult>;
  extract(selector: string, attribute?: string): Promise<BackendActionResult>;
  getUrl(): Promise<BackendActionResult>;
  back(): Promise<BackendActionResult>;
  forward(): Promise<BackendActionResult>;
}
