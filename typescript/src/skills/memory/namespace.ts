/**
 * Whose memory a turn reads and writes (gap-closure plan item 2.1, 2026-09-26).
 *
 * THE NAMESPACE COMES FROM THE VERIFIED CALLER, never from a parameter. The
 * rule is the session skill's `conversationOwner` (`skills/session/skill.ts`),
 * so a caller's memory and a caller's conversations are keyed on the same
 * identity: the agent's owner gets `owner`; anyone else gets
 * `caller:<principal>` for the first identity something verified (`user:`,
 * `agent:`, `key:`, and `channel:` for a channel sender); a caller nothing
 * verified has no namespace and reads only `shared`. An admin is a caller like
 * any other: admin is a tier for tools, not a claim on the owner's notes.
 *
 * `shared` is written by the owner and read by everyone. A non-owner writes
 * only into its own namespace, whatever it asks for, so nothing a stranger
 * says can land in the owner's notes (plan principle 7).
 *
 * The local tier keeps each namespace in a folder: `owner`, `shared`, and
 * `callers/<sha256(principal)[:32]>` (the session skill's `callerKey`, pinned
 * by both fixtures). The Python twin is
 * `python/webagents/agents/skills/local/memory/memory_namespace.py`; both run
 * `python/tests/fixtures/memory_tool/definition.json`.
 */

import { createHash } from 'node:crypto';
import type { AuthInfo } from '../../core/types';
import { tierOf, userPrincipals } from '../access/skill';

export const OWNER_NAMESPACE = 'owner';
export const SHARED_NAMESPACE = 'shared';
export const CALLER_PREFIX = 'caller:';

/**
 * A namespace as a tool or the portal may name it. No whitespace and NO COMMA
 * in a caller principal (S-298, 2026-09-26): namespace lists used to travel to
 * the portal comma-joined, and a Web Bot Auth principal built from a
 * `jwks_uri` keeps a comma in its path, so a caller keyed at
 * `https://evil.example/x,owner` was read as two namespaces, the owner's
 * among them. Lists are repeated parameters now (`portal-store.ts`), and the
 * same grammar holds here, in the Python twin and on the portal
 * (`lib/storage/memory-scoped-service.ts`): a principal that is not a
 * namespace gets none (`namespaceOf`).
 */
export const NAMESPACE_RE = /^(owner|shared|caller:[^\s,]{1,300})$/;

/** A key as `memory_write` accepts it: one path segment, no leading dot. */
export const KEY_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

const VERIFIED_PRINCIPAL = /^(user|agent|key|channel):./;

/** The refusal for a key that is not a slug (the fixture's sentence). */
export function keyRefusal(key: unknown): string {
  return `memory: key must be a slug of letters, digits, dots, dashes and underscores (up to 128), not ${JSON.stringify(String(key))}.`;
}

export function isValidKey(key: unknown): key is string {
  return typeof key === 'string' && KEY_RE.test(key);
}

/**
 * The namespace of this turn's caller: `owner`, `caller:<principal>`, or null
 * for a caller nothing verified (file comment).
 */
export function namespaceOf(auth: Partial<AuthInfo> | undefined | null): string | null {
  if (!auth || auth.authenticated === false) return null;
  if (tierOf(auth) === 'owner') return OWNER_NAMESPACE;
  const listed = (auth as { principals?: unknown }).principals;
  const verified = Array.isArray(listed)
    ? listed.filter((p): p is string => typeof p === 'string' && VERIFIED_PRINCIPAL.test(p))
    : userPrincipals(auth);
  if (!verified[0]) return null;
  // A principal that is not a namespace (whitespace, a comma) has none: such
  // a caller reads `shared` alone, as a caller nothing verified does (S-298).
  const namespace = `${CALLER_PREFIX}${verified[0]}`;
  return NAMESPACE_RE.test(namespace) ? namespace : null;
}

export function isOwnerNamespace(namespace: string | null): boolean {
  return namespace === OWNER_NAMESPACE;
}

/**
 * The namespaces a caller may read: everything for the owner (`null` here
 * means "no filter"), its own plus `shared` for a verified caller, `shared`
 * alone for nobody.
 */
export function readableNamespaces(namespace: string | null): string[] | null {
  if (namespace === OWNER_NAMESPACE) return null;
  return namespace ? [namespace, SHARED_NAMESPACE] : [SHARED_NAMESPACE];
}

/** The namespaces a caller may write: `owner` and `shared` for the owner, its own for a caller, none for nobody. */
export function writableNamespaces(namespace: string | null): string[] {
  if (namespace === OWNER_NAMESPACE) return [OWNER_NAMESPACE, SHARED_NAMESPACE];
  return namespace ? [namespace] : [];
}

/**
 * The namespace a call acts on: the caller's own, or the one it names when it
 * may (the owner naming any namespace; `shared` for reads by anyone). Returns
 * null when the caller has none and named none it may use.
 */
export function targetNamespace(
  callerNamespace: string | null,
  requested: unknown,
  mode: 'read' | 'write',
): string | null {
  const named = typeof requested === 'string' && requested.trim() ? requested.trim() : undefined;
  if (!named) return callerNamespace;
  if (!NAMESPACE_RE.test(named)) return null;
  if (callerNamespace === OWNER_NAMESPACE) return named;
  const allowed = mode === 'write' ? writableNamespaces(callerNamespace) : (readableNamespaces(callerNamespace) ?? []);
  return allowed.includes(named) ? named : null;
}

/** The session skill's caller key: the first 32 hex characters of the principal's SHA-256. */
export function callerKey(principal: string): string {
  return createHash('sha256').update(principal, 'utf8').digest('hex').slice(0, 32);
}

/** Where the local tier keeps a namespace, relative to the memory root. */
export function localDirOf(namespace: string): string {
  if (namespace === OWNER_NAMESPACE || namespace === SHARED_NAMESPACE) return namespace;
  if (namespace.startsWith(CALLER_PREFIX)) return `callers/${callerKey(namespace.slice(CALLER_PREFIX.length))}`;
  throw new Error(`memory: not a namespace: ${JSON.stringify(namespace)}`);
}

/** RFC 4122 UUID v5 (SHA-1) in the URL namespace, as Python's `uuid.uuid5(NAMESPACE_URL, name)`. */
const URL_NAMESPACE = '6ba7b811-9dad-11d1-80b4-00c04fd430c8';

export function uuid5(name: string): string {
  const ns = Buffer.from(URL_NAMESPACE.replace(/-/g, ''), 'hex');
  const hash = createHash('sha1').update(Buffer.concat([ns, Buffer.from(name, 'utf8')])).digest();
  const b = Buffer.from(hash.subarray(0, 16));
  b[6] = (b[6] & 0x0f) | 0x50;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = b.toString('hex');
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

/**
 * An entry's id in every tier: uuid5 over the store (the agent's platform id,
 * or its name when it has none), the namespace and the key, joined by
 * newlines. The same key in the same namespace is the same entry everywhere,
 * so a sync merge by id is a merge by key.
 */
export function entryIdFor(store: string, namespace: string, key: string): string {
  return uuid5(`${store}\n${namespace}\n${key}`);
}
