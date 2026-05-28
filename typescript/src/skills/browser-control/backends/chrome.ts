/**
 * ChromeBrowserBackend — drives a tab from the user's Chrome extension.
 *
 * Plan v3-02 (ADR-v3-11). The extension pivots from autonomous agent to
 * THIN TAB PROVIDER: it offers selected tabs into the host.live webrtc
 * resolution path, and accepts verb commands over the existing WS
 * bridge (`/api/extension/ws`).
 *
 * Status: STRUCTURAL — uses an injected `extensionTransport` (so tests
 * can mock the WS round-trip). The real transport sends commands as
 * `{type:'browser-control', verb, params}` frames and awaits matching
 * `{type:'browser-control-result', ...}` replies. The legacy adapter
 * at `webagents/typescript/src/extension/background/chrome-browser-adapter.ts`
 * already implements the in-extension half — Plan v3-02 wires the
 * portal half here. Full WebRTC tab-capture pipeline is marked TODO.
 */

import type { BackendActionResult, BrowserBackend, SessionHandle } from '../backend';

export interface ExtensionTransport {
  /** Send a verb command and await the result. Implementation: portal WS pod. */
  send(verb: string, params: Record<string, unknown>): Promise<BackendActionResult>;
  /** Allocate / acquire a tab from the extension popup. */
  acquireTab(args?: { initialUrl?: string }): Promise<{ tabId: number; title?: string }>;
  /** Release the tab + tear down WebRTC. */
  releaseTab(tabId: number): Promise<void>;
}

export interface ChromeBrowserBackendConfig {
  transport: ExtensionTransport;
}

export class ChromeBrowserBackend implements BrowserBackend {
  private readonly transport: ExtensionTransport;
  private currentTabId: number | null = null;

  constructor(config: ChromeBrowserBackendConfig) {
    this.transport = config.transport;
  }

  async open(args?: { initialUrl?: string }): Promise<SessionHandle> {
    const { tabId } = await this.transport.acquireTab(args);
    this.currentTabId = tabId;

    // TODO Plan v3-02: full WebRTC tab-capture pipeline.
    //   Extension side:
    //     chrome.tabCapture.getMediaStreamId({consumerTabId: ...})
    //       → streamId
    //     navigator.mediaDevices.getUserMedia({
    //       video: { mandatory: {
    //         chromeMediaSource: 'tab',
    //         chromeMediaSourceId: streamId,
    //       } } })
    //     RTCPeerConnection.addTrack(stream.getVideoTracks()[0])
    //     send SDP offer through portal WS → liveDispatcher
    //   Portal side (Plan 1 content:* handler) routes offer/answer
    //   between extension and the browser-stream-viewer widget.
    //   EXTENSION_WEBRTC_TURN_URL (Plan 1) supplies TURN config.
    return {
      providerSessionId: `ext-tab-${tabId}`,
      liveViewUrl: undefined, // widget composes its own viewer
      providerSessionTtlSeconds: null,
      liveTransport: 'webrtc',
    };
  }

  async close(): Promise<BackendActionResult> {
    const id = this.currentTabId;
    this.currentTabId = null;
    if (id == null) return { success: true, data: { reason: 'no_tab' } };
    try {
      await this.transport.releaseTab(id);
      return { success: true };
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  navigate(url: string): Promise<BackendActionResult> {
    return this.guarded('navigate', { url });
  }
  click(target: string | { x: number; y: number }): Promise<BackendActionResult> {
    return this.guarded('click', { target });
  }
  type(text: string, selector?: string): Promise<BackendActionResult> {
    return this.guarded('type', { text, selector });
  }
  screenshot(): Promise<BackendActionResult> {
    return this.guarded('screenshot', {});
  }
  scroll(direction: 'up' | 'down' | 'left' | 'right', amount?: number): Promise<BackendActionResult> {
    return this.guarded('scroll', { direction, amount });
  }
  wait(args: { selector?: string; timeoutMs?: number }): Promise<BackendActionResult> {
    return this.guarded('wait', args as Record<string, unknown>);
  }
  extract(selector: string, attribute?: string): Promise<BackendActionResult> {
    return this.guarded('extract', { selector, attribute });
  }
  getUrl(): Promise<BackendActionResult> {
    return this.guarded('getUrl', {});
  }
  back(): Promise<BackendActionResult> {
    return this.guarded('back', {});
  }
  forward(): Promise<BackendActionResult> {
    return this.guarded('forward', {});
  }

  private async guarded(verb: string, params: Record<string, unknown>): Promise<BackendActionResult> {
    if (this.currentTabId == null) {
      return { success: false, error: 'no active extension tab — call open() first' };
    }
    return this.transport.send(verb, { ...params, tabId: this.currentTabId });
  }
}
