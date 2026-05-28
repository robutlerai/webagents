/**
 * BrowserbaseBackend — drives a Browserbase-hosted browser session.
 *
 * Plan v3-02. Cloud backend: Browserbase serves the polished live view
 * URL; portal exposes that URL via Plan 1's Case-A resolver
 * (`/api/agents/[id]/sessions/[contentId]/live?redirect=1`). The
 * caller never sees the Browserbase URL directly — LB-layer Location
 * redaction (Plan 1) hides the upstream origin from logs.
 *
 * Status: STRUCTURAL — `open()` / `close()` / verb calls are stubbed
 * with `// TODO Plan v3-02 (browserbase HTTP wire)` markers. The real
 * implementation requires `BROWSERBASE_API_KEY` + `BROWSERBASE_PROJECT_ID`
 * and a Playwright-over-WS CDP relay; we keep the adapter shape +
 * SessionHandle contract correct so the surrounding plumbing
 * (skill / route handler / seeds / metrics) compiles + works.
 */

import type { BackendActionResult, BrowserBackend, SessionHandle } from '../backend';

export interface BrowserbaseBackendConfig {
  apiKey?: string;
  projectId?: string;
  /** Default session TTL in seconds. Browserbase free tier caps at 900. */
  sessionTtlSeconds?: number;
  /** Override the API base for tests. */
  apiBase?: string;
  /** Optional injected `fetch` (tests use a mock). */
  fetchImpl?: typeof fetch;
}

const DEFAULT_BB_BASE = 'https://api.browserbase.com/v1';
const DEFAULT_BB_TTL = 900;

interface BrowserbaseSession {
  id: string;
  liveUrl: string;
  ttlSeconds: number;
}

export class BrowserbaseBackend implements BrowserBackend {
  private readonly apiKey: string;
  private readonly projectId: string;
  private readonly apiBase: string;
  private readonly sessionTtlSeconds: number;
  private readonly fetchImpl: typeof fetch;

  private session: BrowserbaseSession | null = null;

  constructor(config: BrowserbaseBackendConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.BROWSERBASE_API_KEY ?? '';
    this.projectId = config.projectId ?? process.env.BROWSERBASE_PROJECT_ID ?? '';
    this.apiBase = config.apiBase ?? DEFAULT_BB_BASE;
    this.sessionTtlSeconds = config.sessionTtlSeconds ?? DEFAULT_BB_TTL;
    this.fetchImpl = config.fetchImpl ?? (globalThis.fetch as typeof fetch);
  }

  async open(_args?: { initialUrl?: string }): Promise<SessionHandle> {
    if (!this.apiKey || !this.projectId) {
      // TODO Plan v3-02 (browserbase HTTP wire): require real creds in
      // production. For dev / structural compile we mint a synthetic
      // session so the skill lifecycle works end-to-end without a real
      // Browserbase account.
      const id = `bb-mock-${Math.random().toString(36).slice(2, 10)}`;
      const liveUrl = `https://www.browserbase.com/sessions/${id}/live?mock=1`;
      this.session = { id, liveUrl, ttlSeconds: this.sessionTtlSeconds };
      return {
        providerSessionId: id,
        liveViewUrl: liveUrl,
        providerSessionTtlSeconds: this.sessionTtlSeconds,
        liveTransport: 'iframe-url',
      };
    }

    // TODO Plan v3-02 (browserbase HTTP wire): replace mock with real
    // POST https://api.browserbase.com/v1/sessions
    //   Authorization: Bearer <apiKey>
    //   body: { projectId, keepAlive: false, region: 'us-west-2' }
    //   → 201 { id, connectUrl, debugUrl }
    // The debugUrl is the live view; connectUrl is the CDP relay.
    const resp = await this.fetchImpl(`${this.apiBase}/sessions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ projectId: this.projectId }),
    });
    if (!resp.ok) {
      throw new Error(`browserbase open failed: ${resp.status} ${resp.statusText}`);
    }
    const body = (await resp.json()) as { id: string; debugUrl?: string; liveUrl?: string };
    const liveUrl = body.liveUrl ?? body.debugUrl ?? '';
    this.session = { id: body.id, liveUrl, ttlSeconds: this.sessionTtlSeconds };
    return {
      providerSessionId: body.id,
      liveViewUrl: liveUrl,
      providerSessionTtlSeconds: this.sessionTtlSeconds,
      liveTransport: 'iframe-url',
    };
  }

  async close(): Promise<BackendActionResult> {
    const sess = this.session;
    this.session = null;
    if (!sess) return { success: true, data: { reason: 'no_session' } };

    if (!this.apiKey || !this.projectId) {
      return { success: true, data: { reason: 'mock_session_dropped', id: sess.id } };
    }

    // TODO Plan v3-02 (browserbase HTTP wire): DELETE /v1/sessions/:id
    try {
      const resp = await this.fetchImpl(`${this.apiBase}/sessions/${encodeURIComponent(sess.id)}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      if (!resp.ok && resp.status !== 404) {
        return { success: false, error: `browserbase close failed: ${resp.status}` };
      }
      return { success: true };
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  // ---- verb stubs: cloud sessions are driven via CDP over WS, not REST
  // (the user observes via the live view URL). The agent loop sees these
  // verbs succeed structurally; the actual page mutation is wired by the
  // CDP relay marked below.

  async navigate(url: string): Promise<BackendActionResult> {
    return this.cdpVerb('navigate', { url });
  }
  async click(target: string | { x: number; y: number }): Promise<BackendActionResult> {
    return this.cdpVerb('click', { target });
  }
  async type(text: string, selector?: string): Promise<BackendActionResult> {
    return this.cdpVerb('type', { text, selector });
  }
  async screenshot(): Promise<BackendActionResult> {
    return this.cdpVerb('screenshot', {});
  }
  async scroll(direction: 'up' | 'down' | 'left' | 'right', amount?: number): Promise<BackendActionResult> {
    return this.cdpVerb('scroll', { direction, amount });
  }
  async wait(args: { selector?: string; timeoutMs?: number }): Promise<BackendActionResult> {
    return this.cdpVerb('wait', args);
  }
  async extract(selector: string, attribute?: string): Promise<BackendActionResult> {
    return this.cdpVerb('extract', { selector, attribute });
  }
  async getUrl(): Promise<BackendActionResult> {
    return this.cdpVerb('getUrl', {});
  }
  async back(): Promise<BackendActionResult> {
    return this.cdpVerb('back', {});
  }
  async forward(): Promise<BackendActionResult> {
    return this.cdpVerb('forward', {});
  }

  /**
   * TODO Plan v3-02 (browserbase HTTP wire): replace this dispatcher
   * with a real CDP-over-WS client. For now we return a structural
   * "queued" envelope so the agent loop + tests can exercise the verb
   * surface without a live Browserbase session.
   */
  private async cdpVerb(verb: string, params: Record<string, unknown>): Promise<BackendActionResult> {
    if (!this.session) {
      return { success: false, error: 'no active browserbase session' };
    }
    return {
      success: true,
      data: {
        verb,
        params,
        sessionId: this.session.id,
        note: 'browserbase CDP wire pending — see TODO in backends/browserbase.ts',
      },
    };
  }
}
