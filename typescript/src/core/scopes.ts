/**
 * Who may use a tool, prompt or handoff declared with a scope.
 *
 * ONE RULE, BOTH SDKs (ADR-0045 section 5, 2026-09-25). The tool gate used to
 * require every listed scope (`hasScopes`, AND) while Python accepted any one
 * (OR); prompts took `scopeHierarchy[s] || 1`, so an unknown prompt scope was
 * shown to everyone; and `hasScope('owner')` refused an admin whom the prompt
 * tier let through. One function now decides, and both SDKs run the same table
 * of cases (`python/tests/fixtures/scopes/scope_allows.json`):
 *
 *   - `all`, `undefined` or an empty list: anyone.
 *   - `user`: a verified caller (`user`, `owner` or `admin`).
 *   - `owner`: the owner, or an admin. `admin`: an admin only.
 *   - `group:<name>`: a member of that group, or the owner, or an admin.
 *   - anything else: only a caller holding exactly that scope. Unknown scopes
 *     fail closed.
 *   - a list: any one of its entries.
 *
 * A caller's scopes are `auth.scopes`, the tier in `auth.scope`, and `user`
 * when the context is authenticated (`callerScopes`). `scopes` is trusted as
 * the embedder or an auth skill set it, as `hasScope` always did; the access
 * check (ADR-0045) writes the `group:` entries there and removes any it did
 * not grant.
 */

import type { AuthInfo } from './types';

export const GROUP_PREFIX = 'group:';

/** A declared scope: none, one, or a list meaning any-of. */
export type RequiredScope = string | readonly string[] | null | undefined;

/** Every scope the caller an `AuthInfo` describes holds; empty for anonymous. */
export function callerScopes(auth: Partial<AuthInfo> | null | undefined): Set<string> {
  const held = new Set<string>();
  if (!auth) return held;
  for (const scope of auth.scopes ?? []) {
    if (typeof scope === 'string' && scope) held.add(scope);
  }
  if (typeof auth.scope === 'string' && auth.scope) held.add(auth.scope);
  if (auth.authenticated) held.add('user');
  return held;
}

function oneAllows(required: string, caller: ReadonlySet<string>): boolean {
  if (required === 'all') return true;
  if (required === 'admin') return caller.has('admin');
  if (required === 'owner') return caller.has('owner') || caller.has('admin');
  if (required === 'user') return caller.has('user') || caller.has('owner') || caller.has('admin');
  if (required.startsWith(GROUP_PREFIX)) {
    return caller.has(required) || caller.has('owner') || caller.has('admin');
  }
  return caller.has(required);
}

/** Whether a caller holding `caller` scopes may use something declared `required`. */
export function scopeAllows(required: RequiredScope, caller: Iterable<string> | null | undefined): boolean {
  const held = caller instanceof Set ? (caller as ReadonlySet<string>) : new Set(caller ?? []);
  if (required === null || required === undefined) return true;
  if (typeof required === 'string') return oneAllows(required, held);
  if (required.length === 0) return true;
  return required.some((entry) => typeof entry === 'string' && oneAllows(entry, held));
}
