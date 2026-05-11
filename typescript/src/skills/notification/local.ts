/**
 * Default standalone-SDK implementation of {@link NotificationSkill}.
 *
 * - `requestApproval` auto-approves and emits a structured
 *   `console.warn` so headless / library consumers can audit
 *   sensitive tool calls without wiring a UI.
 * - `notify` writes to the matching `console` level.
 *
 * `BaseAgent` auto-injects this skill when the caller's skill list
 * contains no other `NotificationSkill` subclass. This matches the
 * `AuthSkill` default behaviour (auto-loaded when no auth skill is
 * present) and keeps the standalone SDK working without explicit
 * wiring.
 *
 * Library consumers wanting prompt-style approval can subclass
 * `NotificationSkill` directly (e.g. `CliPromptNotificationSkill`,
 * `WebhookApprovalNotificationSkill`) and pass it in `agent.skills`.
 */

import { NotificationSkill } from './skill';
import type {
  ApprovalRequest,
  ApprovalDecision,
  NotificationMessage,
} from '../../core/types';

export class LocalNotificationSkill extends NotificationSkill {
  override readonly name = 'LocalNotificationSkill';

  async requestApproval(req: ApprovalRequest): Promise<ApprovalDecision> {
    console.warn(
      `[notification:auto-approve] tool=${req.toolName} category=${req.category ?? 'other'}`,
      // Keep args summary terse — full args could be large or contain
      // sensitive material. Operators can swap in a custom skill if
      // they want richer auditing.
      typeof req.args === 'object' && req.args !== null
        ? Object.keys(req.args as Record<string, unknown>)
        : typeof req.args,
    );
    return 'approved';
  }

  async notify(msg: NotificationMessage): Promise<void> {
    const line = msg.body ? `${msg.title}: ${msg.body}` : msg.title;
    if (msg.level === 'error') {
      console.error(`[notification] ${line}`);
    } else if (msg.level === 'warn') {
      console.warn(`[notification] ${line}`);
    } else {
      console.log(`[notification] ${line}`);
    }
  }
}
