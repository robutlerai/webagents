/**
 * Webapp templates surfaced to agents through `get_webapp_template` /
 * `list_webapp_templates` and the `webappPatterns` @prompt.
 *
 * Each template is an end-to-end recipe: a runnable function source
 * plus the manifest permissions block the agent needs to merge in.
 *
 * Templates are NOT code that the platform ships and runs on the
 * agent's behalf — they're reference implementations the agent's
 * function-runtime is meant to copy / adapt. The `code` field is a
 * string (NOT a callable) so the LLM can return it verbatim through
 * tool output.
 */

import type { FunctionPermissions } from 'webagents/skills/functions';

/**
 * Subset of `FunctionPermissions` a template asks the agent to enable.
 * Agents merge this into `manifest.permissions` when adopting the
 * template — the `webapp-template-tool` returns it alongside `code`
 * so the LLM has all it needs in one tool round-trip.
 */
export type RequiredPermissions = Partial<
  Pick<FunctionPermissions, 'fetch' | 'kv' | 'secrets' | 'visitor_profile'>
>;

/** Public template shape consumed by the tool + tests. */
export interface WebappTemplate {
  /** Stable name; matches the `pattern` enum in `get_webapp_template`. */
  readonly name: string;
  /** One-liner shown by `list_webapp_templates`. */
  readonly description: string;
  /** When an agent should reach for this template. */
  readonly when_to_use: string;
  /**
   * Security caveats — surfaced verbatim to the LLM. Use plain prose
   * with bullet markers; keep it short but explicit about what the
   * caller must NOT do.
   */
  readonly security_notes: string;
  /**
   * Manifest permissions block the agent should merge into its
   * function's `manifest.permissions`. Empty object for templates
   * that don't need any (e.g. `minimal_html_page`).
   */
  readonly required_permissions: RequiredPermissions;
  /** The function source the agent should adapt. */
  readonly code: string;
}
