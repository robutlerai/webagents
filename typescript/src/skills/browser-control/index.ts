/**
 * Browser control module — Plan v3-02 unified skill + 3 backends.
 *
 * One `BrowserControlSkill` exposes the canonical browser verb set as
 * agent tool functions; the skill is parameterised with a
 * `BrowserBackend` adapter at construction time. Three backends ship
 * in this module:
 *
 *   - `BrowserbaseBackend`   — backends/browserbase.ts
 *   - `BrowserUseBackend`    — backends/browser-use.ts
 *   - `ChromeBrowserBackend` — backends/chrome.ts (extension WS bridge)
 *
 * This is a SIBLING of the legacy `../browser/` skill — the legacy one
 * exposes a tab-multiplexed adapter (`BrowserControlAdapter`) and is
 * still consumed by the Chrome extension's in-extension agent for now.
 * Plan v3-02 (ADR-v3-11) pivots the extension to a tab-provider,
 * after which the new module is the canonical entry point.
 *
 * Wire (cloud session — browserbase / browser-use):
 *
 *   skill.initialize() →
 *     liveAttachRegistry.register({ scope:'agent', kind:'content', … }) →
 *     lockPaymentToken(...)                                              →
 *     backend.open() returns { liveViewUrl, providerSessionTtlSeconds } →
 *     publishLiveUrl('content:<contentId>', { transports:[iframe-url] })→
 *     skill emits HtmlContent.live envelope into the UAMP stream
 *
 *   skill.shutdown() (or DELETE /api/agents/[id]/sessions/[contentId]):
 *     backend.close() →
 *     unpublishLiveUrl('content:<contentId>') →
 *     liveAttachRegistry.release(...)         →
 *     settlePaymentToken(lockId, usage, agentId)
 */

import { Skill } from '../../core/skill';
import { tool, prompt } from '../../core/decorators';
import type { Context, SkillConfig } from '../../core/types';
import type {
  BrowserBackend,
  SessionHandle,
  BackendActionResult,
} from './backend';

export type {
  BrowserBackend,
  SessionHandle,
  BackendActionResult,
} from './backend';
export { BrowserbaseBackend } from './backends/browserbase';
export { BrowserUseBackend } from './backends/browser-use';
export { ChromeBrowserBackend } from './backends/chrome';
export { forwardChildLiveBlock } from './delegation-forwarding';

/**
 * Concurrent-cap reservation result returned from
 * `liveAttachRegistry.register`. Plan 1 owns the wire shape; we keep
 * a structural duplicate here so the skill module does not have to
 * depend on portal-internal imports at build time.
 */
export interface AgentSlotReservation {
  attachId: string;
  signalingChannelId?: string;
  expiresAt: string;
}

/**
 * Pluggable hook surface — the agent runtime injects bindings to the
 * portal's `liveAttachRegistry`, `lockPaymentToken`, `settlePaymentToken`,
 * `publishLiveUrl`, and `unpublishLiveUrl` when constructing the skill.
 *
 * Why injected (not direct imports)? The skill ships in `webagents/`
 * which is a standalone SDK — importing portal-internal modules
 * directly would create a circular dependency. The agent runtime
 * binds these at activation time.
 */
export interface BrowserControlPlatformHooks {
  reserveAgentSlot: (args: {
    userId: string;
    contentId: string;
    transport: 'iframe-url' | 'webrtc';
  }) => Promise<AgentSlotReservation | { ok: false; code: string; message: string }>;
  releaseAgentSlot: (args: { userId: string; attachId: string }) => Promise<{ ok: boolean }>;
  lockPayment: (args: {
    tokenOrJti: string;
    agentId: string;
    amount: bigint;
  }) => Promise<{ lockId: string; lockedAmount: bigint } | { error: string; available: bigint }>;
  settlePayment: (args: {
    lockId: string;
    amount: bigint;
    agentId: string;
    description?: string;
  }) => Promise<{ success: boolean; charged: bigint; remaining: bigint; error?: string }>;
  publishLiveUrl: (args: {
    contentId: string;
    liveViewUrl: string;
    expiresAt?: string;
    producerId: string;
    transport: 'iframe-url' | 'webrtc';
  }) => Promise<void>;
  unpublishLiveUrl: (args: { contentId: string }) => Promise<void>;
  emitToolProgress?: (kind: string, data: Record<string, unknown>) => void;
}

export interface BrowserControlSkillConfig extends SkillConfig {
  /** Required — the verb adapter. */
  backend: BrowserBackend;
  /**
   * Optional — agent-runtime bindings to portal-side concerns
   * (cap, payment, live-url store). When omitted the skill operates
   * in "standalone SDK" mode: open/close still work but no Redis
   * reservation, payment, or live-url publish happens (used by tests
   * + the legacy extension that hasn't pivoted to tab-provider yet).
   *
   * NOTE: named `platformHooks` (not `hooks`) to avoid shadowing the
   * base `Skill.hooks: Hook[]` field — that one is the skill-lifecycle
   * hook list, unrelated to platform integration bindings.
   */
  platformHooks?: BrowserControlPlatformHooks;
  /**
   * Identification used to build the live envelope + caps. Mostly
   * supplied by the agent runtime, not the skill author.
   */
  runtimeContext?: {
    userId: string;
    agentId: string;
    contentId?: string;
    paymentTokenOrJti?: string;
    /** Per-second cost in nano-dollars (1e9 = $1) — defaults to 0. */
    ratePerSecondNano?: bigint;
  };
}

const DEFAULT_RATE_NANO = BigInt(0);

interface ActiveSession {
  handle: SessionHandle;
  startedAt: number;
  contentId: string;
  attachId?: string;
  signalingChannelId?: string;
  paymentLockId?: string;
}

/** Cap on initial payment lock — single session × 1 hour at default rate. */
const DEFAULT_LOCK_AMOUNT_NANO = BigInt(1_000_000_000); // $1

export class BrowserControlSkill extends Skill {
  private readonly backend: BrowserBackend;
  private readonly platformHooks?: BrowserControlPlatformHooks;
  private readonly runtimeContext?: BrowserControlSkillConfig['runtimeContext'];

  private active: ActiveSession | null = null;

  constructor(config: BrowserControlSkillConfig) {
    super({ ...config, name: config.name || 'browser-control' });
    this.backend = config.backend;
    this.platformHooks = config.platformHooks;
    this.runtimeContext = config.runtimeContext;
  }

  @prompt({ priority: 50, name: 'browserControlGuide', scope: 'all' })
  browserControlGuide(_ctx: Context): string {
    return [
      '## Browser control skill (unified)',
      '',
      'You drive a remote browser session. Verbs: `browser_open`,',
      '`browser_close`, `browser_navigate`, `browser_click`, `browser_type`,',
      '`browser_screenshot`, `browser_scroll`, `browser_wait`,',
      '`browser_extract`, `browser_get_url`, `browser_back`, `browser_forward`.',
      '',
      'Open the session ONCE per task — calling `browser_open` mid-session',
      'discards any work in progress. The session emits a live block the',
      'user can watch from their chat / canvas; respect that the user is',
      'observing every action.',
    ].join('\n');
  }

  // ---- session lifecycle ------------------------------------------------

  private async openInternal(initialUrl?: string): Promise<BackendActionResult> {
    if (this.active) {
      return {
        success: false,
        error: 'session already open — call browser_close first or reuse the existing session',
      };
    }
    const contentId =
      this.runtimeContext?.contentId ??
      `bc-${Math.random().toString(36).slice(2, 10)}`;

    // 1) Reserve the concurrent-cap slot (Plan 1 unified registry).
    let reservation: AgentSlotReservation | undefined;
    if (this.platformHooks && this.runtimeContext?.userId) {
      const transportGuess: 'iframe-url' | 'webrtc' = 'iframe-url';
      const res = await this.platformHooks.reserveAgentSlot({
        userId: this.runtimeContext.userId,
        contentId,
        transport: transportGuess,
      });
      if ('ok' in res && res.ok === false) {
        return { success: false, error: `cap reservation failed: ${res.code} ${res.message}` };
      }
      reservation = res as AgentSlotReservation;
    }

    // 2) Lock the payment token (single lock per session — B6).
    let paymentLockId: string | undefined;
    if (
      this.platformHooks &&
      this.runtimeContext?.paymentTokenOrJti &&
      this.runtimeContext.agentId
    ) {
      const lock = await this.platformHooks.lockPayment({
        tokenOrJti: this.runtimeContext.paymentTokenOrJti,
        agentId: this.runtimeContext.agentId,
        amount: DEFAULT_LOCK_AMOUNT_NANO,
      });
      if ('error' in lock) {
        if (reservation && this.platformHooks && this.runtimeContext?.userId) {
          await this.platformHooks.releaseAgentSlot({
            userId: this.runtimeContext.userId,
            attachId: reservation.attachId,
          }).catch(() => undefined);
        }
        return { success: false, error: `payment lock failed: ${lock.error}` };
      }
      paymentLockId = lock.lockId;
    }

    // 3) Open the provider session.
    let handle: SessionHandle;
    try {
      handle = await this.backend.open(initialUrl ? { initialUrl } : undefined);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // Unwind on failure.
      if (reservation && this.platformHooks && this.runtimeContext?.userId) {
        await this.platformHooks.releaseAgentSlot({
          userId: this.runtimeContext.userId,
          attachId: reservation.attachId,
        }).catch(() => undefined);
      }
      return { success: false, error: `backend.open failed: ${msg}` };
    }

    // 4) Publish the live URL into the Plan 1 store (cloud only).
    if (handle.liveViewUrl && this.platformHooks && this.runtimeContext?.agentId) {
      const ttlSec = handle.providerSessionTtlSeconds ?? 900;
      const expiresAt = new Date(Date.now() + ttlSec * 1000).toISOString();
      try {
        await this.platformHooks.publishLiveUrl({
          contentId,
          liveViewUrl: handle.liveViewUrl,
          expiresAt,
          producerId: this.runtimeContext.agentId,
          transport: handle.liveTransport,
        });
      } catch (err) {
        // Best-effort — log and continue. Resolver will return 404 until republished.
        console.error('[browser-control] publishLiveUrl failed:', err);
      }
    }

    this.active = {
      handle,
      contentId,
      startedAt: Date.now(),
      attachId: reservation?.attachId,
      signalingChannelId: reservation?.signalingChannelId,
      paymentLockId,
    };

    this.platformHooks?.emitToolProgress?.('browser_action', {
      verb: 'browser_open',
      result: {
        contentId,
        liveTransport: handle.liveTransport,
        providerSessionTtlSeconds: handle.providerSessionTtlSeconds,
      },
    });

    return {
      success: true,
      data: {
        contentId,
        liveTransport: handle.liveTransport,
        providerSessionTtlSeconds: handle.providerSessionTtlSeconds,
      },
    };
  }

  private async closeInternal(reason: 'user' | 'agent' | 'error' | 'expired' = 'agent'): Promise<BackendActionResult> {
    if (!this.active) return { success: true, data: { reason: 'already_closed' } };
    const sess = this.active;
    this.active = null;

    let backendResult: BackendActionResult;
    try {
      backendResult = await this.backend.close();
    } catch (err) {
      backendResult = { success: false, error: err instanceof Error ? err.message : String(err) };
    }

    // Unpublish live url (best-effort).
    if (this.platformHooks && sess.handle.liveViewUrl) {
      await this.platformHooks.unpublishLiveUrl({ contentId: sess.contentId }).catch(() => undefined);
    }

    // Release the cap slot.
    if (this.platformHooks && sess.attachId && this.runtimeContext?.userId) {
      await this.platformHooks
        .releaseAgentSlot({ userId: this.runtimeContext.userId, attachId: sess.attachId })
        .catch(() => undefined);
    }

    // Settle the payment lock based on elapsed seconds * rate.
    if (this.platformHooks && sess.paymentLockId && this.runtimeContext?.agentId) {
      const elapsedSec = Math.max(1, Math.round((Date.now() - sess.startedAt) / 1000));
      const rate = this.runtimeContext.ratePerSecondNano ?? DEFAULT_RATE_NANO;
      const charge = BigInt(elapsedSec) * rate;
      if (charge > BigInt(0)) {
        await this.platformHooks
          .settlePayment({
            lockId: sess.paymentLockId,
            amount: charge,
            agentId: this.runtimeContext.agentId,
            description: `browser-control session ${sess.contentId} (${elapsedSec}s, reason=${reason})`,
          })
          .catch((err) => console.error('[browser-control] settle failed:', err));
      }
    }

    this.platformHooks?.emitToolProgress?.('browser_action', {
      verb: 'browser_close',
      result: { reason, elapsedMs: Date.now() - sess.startedAt },
    });

    return backendResult;
  }

  /** External callers (route handler / runtime shutdown) drive close through here. */
  async forceClose(reason: 'user' | 'agent' | 'error' | 'expired' = 'user'): Promise<BackendActionResult> {
    return this.closeInternal(reason);
  }

  /** Read-only view of the active session — used by route handlers / debug surfaces. */
  getActiveSession(): Readonly<ActiveSession> | null {
    return this.active;
  }

  // ---- tools ------------------------------------------------------------

  @tool({
    name: 'browser_open',
    description: 'Open a remote browser session. Returns the contentId of the live block.',
    provides: 'browser-control.session',
    parameters: {
      type: 'object',
      properties: {
        initial_url: { type: 'string', description: 'Optional URL to load on open.' },
      },
      required: [],
    },
  })
  async browserOpen(params: Record<string, unknown>, _ctx: Context): Promise<BackendActionResult> {
    return this.openInternal(typeof params.initial_url === 'string' ? params.initial_url : undefined);
  }

  @tool({
    name: 'browser_close',
    description: 'Close the active browser session.',
    provides: 'browser-control.session',
    parameters: { type: 'object', properties: {}, required: [] },
  })
  async browserClose(_params: Record<string, unknown>, _ctx: Context): Promise<BackendActionResult> {
    return this.closeInternal('agent');
  }

  private requireActive(): BackendActionResult | null {
    if (!this.active) {
      return { success: false, error: 'no active browser session — call browser_open first' };
    }
    return null;
  }

  @tool({
    name: 'browser_navigate',
    description: 'Navigate the active browser to a URL.',
    provides: 'browser-control.navigate',
    parameters: {
      type: 'object',
      properties: { url: { type: 'string' } },
      required: ['url'],
    },
  })
  async browserNavigate(params: Record<string, unknown>): Promise<BackendActionResult> {
    const guard = this.requireActive();
    if (guard) return guard;
    return this.backend.navigate(String(params.url ?? ''));
  }

  @tool({
    name: 'browser_click',
    description: 'Click an element by CSS selector or (x,y) coordinates.',
    provides: 'browser-control.action',
    parameters: {
      type: 'object',
      properties: {
        selector: { type: 'string' },
        x: { type: 'number' },
        y: { type: 'number' },
      },
      required: [],
    },
  })
  async browserClick(params: Record<string, unknown>): Promise<BackendActionResult> {
    const guard = this.requireActive();
    if (guard) return guard;
    if (typeof params.selector === 'string') {
      return this.backend.click(params.selector);
    }
    if (typeof params.x === 'number' && typeof params.y === 'number') {
      return this.backend.click({ x: params.x, y: params.y });
    }
    return { success: false, error: 'browser_click requires selector OR (x,y)' };
  }

  @tool({
    name: 'browser_type',
    description: 'Type text into a focused element or one matching a selector.',
    provides: 'browser-control.action',
    parameters: {
      type: 'object',
      properties: {
        text: { type: 'string' },
        selector: { type: 'string' },
      },
      required: ['text'],
    },
  })
  async browserType(params: Record<string, unknown>): Promise<BackendActionResult> {
    const guard = this.requireActive();
    if (guard) return guard;
    return this.backend.type(
      String(params.text ?? ''),
      typeof params.selector === 'string' ? params.selector : undefined,
    );
  }

  @tool({
    name: 'browser_screenshot',
    description: 'Capture a screenshot of the active page.',
    provides: 'browser-control.screenshot',
    parameters: { type: 'object', properties: {}, required: [] },
  })
  async browserScreenshot(): Promise<BackendActionResult> {
    const guard = this.requireActive();
    if (guard) return guard;
    return this.backend.screenshot();
  }

  @tool({
    name: 'browser_scroll',
    description: 'Scroll the active page.',
    provides: 'browser-control.action',
    parameters: {
      type: 'object',
      properties: {
        direction: { type: 'string', enum: ['up', 'down', 'left', 'right'] },
        amount: { type: 'number', description: 'Pixels (default 600).' },
      },
      required: ['direction'],
    },
  })
  async browserScroll(params: Record<string, unknown>): Promise<BackendActionResult> {
    const guard = this.requireActive();
    if (guard) return guard;
    const dir = String(params.direction ?? 'down') as 'up' | 'down' | 'left' | 'right';
    return this.backend.scroll(dir, typeof params.amount === 'number' ? params.amount : undefined);
  }

  @tool({
    name: 'browser_wait',
    description: 'Wait for a selector to appear, or for a timeout.',
    provides: 'browser-control.wait',
    parameters: {
      type: 'object',
      properties: {
        selector: { type: 'string' },
        timeout_ms: { type: 'number' },
      },
      required: [],
    },
  })
  async browserWait(params: Record<string, unknown>): Promise<BackendActionResult> {
    const guard = this.requireActive();
    if (guard) return guard;
    return this.backend.wait({
      selector: typeof params.selector === 'string' ? params.selector : undefined,
      timeoutMs: typeof params.timeout_ms === 'number' ? params.timeout_ms : undefined,
    });
  }

  @tool({
    name: 'browser_extract',
    description: 'Extract text or attribute from a selector match.',
    provides: 'browser-control.extract',
    parameters: {
      type: 'object',
      properties: {
        selector: { type: 'string' },
        attribute: { type: 'string', description: 'Omit to read textContent.' },
      },
      required: ['selector'],
    },
  })
  async browserExtract(params: Record<string, unknown>): Promise<BackendActionResult> {
    const guard = this.requireActive();
    if (guard) return guard;
    return this.backend.extract(
      String(params.selector ?? ''),
      typeof params.attribute === 'string' ? params.attribute : undefined,
    );
  }

  @tool({
    name: 'browser_get_url',
    description: 'Get the current URL of the active page.',
    provides: 'browser-control.read',
    parameters: { type: 'object', properties: {}, required: [] },
  })
  async browserGetUrl(): Promise<BackendActionResult> {
    const guard = this.requireActive();
    if (guard) return guard;
    return this.backend.getUrl();
  }

  @tool({
    name: 'browser_back',
    description: 'Navigate back in browser history.',
    provides: 'browser-control.navigate',
    parameters: { type: 'object', properties: {}, required: [] },
  })
  async browserBack(): Promise<BackendActionResult> {
    const guard = this.requireActive();
    if (guard) return guard;
    return this.backend.back();
  }

  @tool({
    name: 'browser_forward',
    description: 'Navigate forward in browser history.',
    provides: 'browser-control.navigate',
    parameters: { type: 'object', properties: {}, required: [] },
  })
  async browserForward(): Promise<BackendActionResult> {
    const guard = this.requireActive();
    if (guard) return guard;
    return this.backend.forward();
  }
}
