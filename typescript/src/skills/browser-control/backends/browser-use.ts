/**
 * BrowserUseBackend — drives a browser-use.com hosted session.
 *
 * Plan v3-02. Mirrors the BrowserbaseBackend shape — only the API
 * base + auth header differ. browser-use serves its own polished
 * live view URL; portal exposes it via Plan 1's Case-A resolver.
 *
 * Status: STRUCTURAL — real HTTP calls marked with
 * `// TODO Plan v3-02 (browser-use HTTP wire)`. Mock mode kicks in
 * when BROWSERUSE_API_KEY / BROWSERUSE_PROJECT_ID are unset so the
 * skill compiles + tests in environments without provider creds.
 */

import type { BackendActionResult, BrowserBackend, SessionHandle } from '../backend';

export interface BrowserUseBackendConfig {
  apiKey?: string;
  projectId?: string;
  sessionTtlSeconds?: number;
  apiBase?: string;
  fetchImpl?: typeof fetch;
}

const DEFAULT_BU_BASE = 'https://api.browser-use.com/v1';
const DEFAULT_BU_TTL = 900;

interface BrowserUseSession {
  id: string;
  liveUrl: string;
  ttlSeconds: number;
}

export class BrowserUseBackend implements BrowserBackend {
  private readonly apiKey: string;
  private readonly projectId: string;
  private readonly apiBase: string;
  private readonly sessionTtlSeconds: number;
  private readonly fetchImpl: typeof fetch;

  private session: BrowserUseSession | null = null;

  constructor(config: BrowserUseBackendConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.BROWSERUSE_API_KEY ?? '';
    this.projectId = config.projectId ?? process.env.BROWSERUSE_PROJECT_ID ?? '';
    this.apiBase = config.apiBase ?? DEFAULT_BU_BASE;
    this.sessionTtlSeconds = config.sessionTtlSeconds ?? DEFAULT_BU_TTL;
    this.fetchImpl = config.fetchImpl ?? (globalThis.fetch as typeof fetch);
  }

  async open(_args?: { initialUrl?: string }): Promise<SessionHandle> {
    if (!this.apiKey || !this.projectId) {
      const id = `bu-mock-${Math.random().toString(36).slice(2, 10)}`;
      const liveUrl = `https://browser-use.com/sessions/${id}/live?mock=1`;
      this.session = { id, liveUrl, ttlSeconds: this.sessionTtlSeconds };
      return {
        providerSessionId: id,
        liveViewUrl: liveUrl,
        providerSessionTtlSeconds: this.sessionTtlSeconds,
        liveTransport: 'iframe-url',
      };
    }

    // TODO Plan v3-02 (browser-use HTTP wire): real
    // POST /v1/sessions { projectId } → { id, liveUrl, controlUrl }
    const resp = await this.fetchImpl(`${this.apiBase}/sessions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ projectId: this.projectId }),
    });
    if (!resp.ok) {
      throw new Error(`browser-use open failed: ${resp.status} ${resp.statusText}`);
    }
    const body = (await resp.json()) as { id: string; liveUrl?: string };
    const liveUrl = body.liveUrl ?? '';
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

    // TODO Plan v3-02 (browser-use HTTP wire): DELETE /v1/sessions/:id
    try {
      const resp = await this.fetchImpl(`${this.apiBase}/sessions/${encodeURIComponent(sess.id)}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      if (!resp.ok && resp.status !== 404) {
        return { success: false, error: `browser-use close failed: ${resp.status}` };
      }
      return { success: true };
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  async navigate(url: string): Promise<BackendActionResult> {
    return this.verb('navigate', { url });
  }
  async click(target: string | { x: number; y: number }): Promise<BackendActionResult> {
    return this.verb('click', { target });
  }
  async type(text: string, selector?: string): Promise<BackendActionResult> {
    return this.verb('type', { text, selector });
  }
  async screenshot(): Promise<BackendActionResult> {
    return this.verb('screenshot', {});
  }
  async scroll(direction: 'up' | 'down' | 'left' | 'right', amount?: number): Promise<BackendActionResult> {
    return this.verb('scroll', { direction, amount });
  }
  async wait(args: { selector?: string; timeoutMs?: number }): Promise<BackendActionResult> {
    return this.verb('wait', args);
  }
  async extract(selector: string, attribute?: string): Promise<BackendActionResult> {
    return this.verb('extract', { selector, attribute });
  }
  async getUrl(): Promise<BackendActionResult> {
    return this.verb('getUrl', {});
  }
  async back(): Promise<BackendActionResult> {
    return this.verb('back', {});
  }
  async forward(): Promise<BackendActionResult> {
    return this.verb('forward', {});
  }

  /**
   * TODO Plan v3-02 (browser-use HTTP wire): replace with real
   * action dispatch (POST /v1/sessions/:id/actions { type, params }).
   */
  private async verb(verb: string, params: Record<string, unknown>): Promise<BackendActionResult> {
    if (!this.session) {
      return { success: false, error: 'no active browser-use session' };
    }
    return {
      success: true,
      data: {
        verb,
        params,
        sessionId: this.session.id,
        note: 'browser-use action wire pending — see TODO in backends/browser-use.ts',
      },
    };
  }
}
