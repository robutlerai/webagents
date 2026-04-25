/**
 * Optional cached resolver wrapping a TokenResolver with auto-refresh.
 *
 * Standalone hosts that don't have a portal-style refresh framework can
 * use this to add lazy refresh on top of an env- or file-backed resolver.
 * Pass a `refresh()` function that mints a new token and writes it back
 * via the wrapped resolver; this helper handles the cache + concurrent
 * refresh coordination (single-flight per provider/agent).
 */
import type { ResolvedToken, TokenResolver } from './options';

interface CacheEntry {
  resolved: ResolvedToken;
  cachedAt: number;
}

export interface RefreshableResolverOptions {
  /** Wrapped underlying resolver (env/file/db). */
  base: TokenResolver;
  /**
   * Refresh hook. Called when the cached token is within `skewMs` of
   * `expiresAt`. Returning the new ResolvedToken updates the cache.
   * Returning null causes the next read to fall through to `base`.
   */
  refresh: (current: ResolvedToken & { provider: string }) => Promise<ResolvedToken | null>;
  /** Skew window before expiry, default 60s. */
  skewMs?: number;
  /** Cache TTL when no expiresAt is known, default 60s. */
  defaultTtlMs?: number;
}

export function refreshableTokenResolver(opts: RefreshableResolverOptions): TokenResolver {
  const cache = new Map<string, CacheEntry>();
  const inflight = new Map<string, Promise<ResolvedToken | null>>();
  const skewMs = opts.skewMs ?? 60_000;
  const defaultTtlMs = opts.defaultTtlMs ?? 60_000;

  return {
    async getToken(input) {
      const key = `${input.provider}:${input.agentId ?? '*'}:${input.integrationId ?? '*'}`;
      const cached = cache.get(key);
      if (cached) {
        const expiresAt = cached.resolved.expiresAt?.getTime();
        if (expiresAt) {
          if (expiresAt - Date.now() > skewMs) return cached.resolved;
        } else if (Date.now() - cached.cachedAt < defaultTtlMs) {
          return cached.resolved;
        }
      }

      let promise = inflight.get(key);
      if (!promise) {
        promise = (async () => {
          const fresh = await opts.base.getToken(input);
          if (!fresh) return null;
          if (fresh.expiresAt && fresh.expiresAt.getTime() - Date.now() < skewMs) {
            const refreshed = await opts.refresh({ ...fresh, provider: input.provider });
            if (refreshed) {
              cache.set(key, { resolved: refreshed, cachedAt: Date.now() });
              return refreshed;
            }
          }
          cache.set(key, { resolved: fresh, cachedAt: Date.now() });
          return fresh;
        })().finally(() => inflight.delete(key));
        inflight.set(key, promise);
      }
      return promise;
    },
  };
}
