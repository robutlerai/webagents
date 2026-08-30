/**
 * Platform registration surface for a served agent (TypeScript edition).
 *
 * Mirrors `python/webagents/server/core/registration.py`. These are not
 * conveniences: they are what the platform's registration path READS, and
 * they used to live in a `host()` wrapper — so the DOCUMENTED server
 * (`serve()`) produced an agent registration could never complete.
 *
 * The card itself is served by `createFetchHandler` (origin AND agent prefix,
 * with `metadata.publicKey` as an SPKI PEM). What lives here is the other
 * half: presence.
 */

export const HEARTBEAT_INTERVAL_MS = 60_000;

function envVar(name: string): string | undefined {
  return typeof process !== 'undefined' ? process.env?.[name] : undefined;
}

/** The platform's HTTP API base, for the heartbeat. */
export function resolvePortalApiUrl(configured?: string): string | undefined {
  return (
    configured ||
    envVar('ROBUTLER_INTERNAL_API_URL') ||
    envVar('ROBUTLER_API_URL') ||
    undefined
  );
}

/** The per-agent platform key. */
export function resolveAgentToken(configured?: string): string | undefined {
  return configured || envVar('WEBAGENTS_AGENT_TOKEN') || undefined;
}

export interface HeartbeatHandle {
  stop(): void;
}

/**
 * POST `{portalApiUrl}/api/agents/heartbeat` every 60s with the per-agent
 * token. A failed beat is a warning, never a crash: an agent that cannot
 * reach the platform must keep serving the callers that can reach IT. The
 * route takes no body and derives identity from the bearer.
 *
 * Returns `null` — with a reason logged — when there is nothing to beat with.
 * Silence here is the failure mode this guards against: an agent the platform
 * lists as unknown looks exactly like one that is merely idle.
 */
export function startHeartbeat(
  agentName: string,
  options: { portalApiUrl?: string; token?: string; intervalMs?: number } = {},
): HeartbeatHandle | null {
  const portalApiUrl = resolvePortalApiUrl(options.portalApiUrl);
  const token = resolveAgentToken(options.token);
  if (!portalApiUrl || !token) {
    console.info(
      `[webagents] no heartbeat for ${agentName}: needs WEBAGENTS_AGENT_TOKEN and ` +
        'ROBUTLER_API_URL. The platform will show this agent as unknown until it beats.',
    );
    return null;
  }

  const url = `${portalApiUrl.replace(/\/+$/, '')}/api/agents/heartbeat`;
  const beat = async () => {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        console.warn(`[webagents] heartbeat for ${agentName}: HTTP ${res.status} from ${url}`);
      }
    } catch (err) {
      console.warn(`[webagents] heartbeat for ${agentName} failed: ${(err as Error).message}`);
    }
  };

  void beat();
  const timer = setInterval(() => void beat(), options.intervalMs ?? HEARTBEAT_INTERVAL_MS);
  // Presence must never be the reason a process refuses to exit.
  (timer as unknown as { unref?: () => void }).unref?.();
  console.log(`[webagents] heartbeat started for ${agentName} -> ${url}`);
  return { stop: () => clearInterval(timer) };
}
