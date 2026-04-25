/**
 * Shared Meta Graph helpers used by WhatsApp / Messenger / Instagram skills.
 *
 * The Graph version is read from `META_GRAPH_API_VERSION` (default v25.0,
 * matching Meta's quarterly release cadence as of 2026-04). Tokens accept
 * an optional `appsecret_proof` per Meta's stricter security profile —
 * skills compute it when the host supplies the app secret via
 * `metadata.appSecret` or env `META_APP_SECRET`.
 */
import { createHmac } from 'node:crypto';

export const META_GRAPH_VERSION = (process.env.META_GRAPH_API_VERSION ?? 'v25.0').replace(
  /^v?/,
  'v',
);

export function metaGraphUrl(path: string): string {
  const cleaned = path.startsWith('/') ? path : `/${path}`;
  return `https://graph.facebook.com/${META_GRAPH_VERSION}${cleaned}`;
}

export function metaInstagramGraphUrl(path: string): string {
  const cleaned = path.startsWith('/') ? path : `/${path}`;
  return `https://graph.instagram.com${cleaned}`;
}

export function computeAppsecretProof(accessToken: string, appSecret: string): string {
  return createHmac('sha256', appSecret).update(accessToken).digest('hex');
}

export interface MetaGraphInit {
  accessToken: string;
  appsecretProof?: string;
  method?: string;
  body?: Record<string, unknown>;
  query?: Record<string, string | number | boolean | undefined>;
}

export async function metaGraph<T = unknown>(
  path: string,
  init: MetaGraphInit,
  baseUrl: (p: string) => string = metaGraphUrl,
): Promise<T> {
  const qs = new URLSearchParams({ access_token: init.accessToken });
  if (init.appsecretProof) qs.set('appsecret_proof', init.appsecretProof);
  if (init.query) {
    for (const [k, v] of Object.entries(init.query)) {
      if (v !== undefined) qs.set(k, String(v));
    }
  }
  const sep = path.includes('?') ? '&' : '?';
  const url = `${baseUrl(path)}${sep}${qs.toString()}`;
  const r = await fetch(url, {
    method: init.method ?? 'GET',
    headers: init.body ? { 'Content-Type': 'application/json' } : undefined,
    body: init.body ? JSON.stringify(init.body) : undefined,
  });
  const text = await r.text();
  if (!r.ok) {
    const err = new Error(text.slice(0, 300)) as Error & { status?: number };
    err.status = r.status;
    throw err;
  }
  return text ? (JSON.parse(text) as T) : ({} as T);
}
