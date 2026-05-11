/**
 * NotificationSkill — pluggable approval + notification surface.
 *
 * Mirrors the {@link AuthSkill} pattern: a `Skill` subclass that
 * installs capabilities into `Context` during a `before_run` hook so
 * the agent loop reads through the standard `Context` interface
 * (exactly like it reads `context.auth`).
 *
 * Two abstract methods drive the contract:
 *   - `requestApproval(req)` — agent wants to call a tool decorated
 *     with `requiresConfirmation: true`. Return `'approved'` to let
 *     it run, `'rejected'` to short-circuit with a tool error.
 *   - `notify(msg)` — best-effort operator notification (toast, log,
 *     webhook, etc.). Failures are not propagated.
 *
 * Subclasses:
 *   - `LocalNotificationSkill` (this package, default in BaseAgent) —
 *     auto-approves with `console.warn`, logs notifications.
 *   - `PortalNotificationSkill` (portal repo) — bridges to the portal
 *     `tool_approval` notification + `CriticalToast` UI with a 30s
 *     chat-context timeout so the LLM stream doesn't outlast the
 *     provider deadline.
 */

import { Skill } from '../../core/skill';
import { hook } from '../../core/decorators';
import type {
  HookData,
  HookResult,
  Context,
  ApprovalRequest,
  ApprovalDecision,
  NotificationMessage,
} from '../../core/types';

export type {
  ApprovalRequest,
  ApprovalDecision,
  NotificationMessage,
} from '../../core/types';

export abstract class NotificationSkill extends Skill {
  /**
   * Decide whether the agent may execute a sensitive tool. Implementations
   * MUST resolve to `'approved'` or `'rejected'`; never throw — the agent
   * loop treats unhandled rejections as `'rejected'` for safety.
   */
  abstract requestApproval(req: ApprovalRequest): Promise<ApprovalDecision>;

  /**
   * Surface a structured notification. Best-effort; the agent loop ignores
   * the return value and swallows exceptions.
   */
  abstract notify(msg: NotificationMessage): Promise<void>;

  /**
   * Install the two `Context` hooks. Same lifecycle / priority as
   * AuthSkill (priority 5, after on_connection auth) so by the time the
   * agent loop reads `context.requestToolApproval` / `context.notify`
   * they are guaranteed present.
   *
   * Ordering: priority `5` runs early in `before_run` — earlier than
   * any messaging-skill hook that might surface a tool prompt — but
   * after on_connection auth has populated `context.auth`. The two
   * skills write disjoint context keys, so co-location is safe.
   */
  @hook({ lifecycle: 'before_run', priority: 5 })
  async installContextHooks(_data: HookData, context: Context): Promise<HookResult | void> {
    context.requestToolApproval = (req) => this.requestApproval(req);
    context.notify = (msg) => this.notify(msg).catch(() => undefined);
  }
}
