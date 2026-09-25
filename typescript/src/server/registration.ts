/**
 * Platform registration surface for a served agent (TypeScript edition).
 *
 * Mirrors `python/webagents/server/core/registration.py`. These are not
 * conveniences: they are what the platform's registration path READS, and
 * they used to live in a `host()` wrapper — so the DOCUMENTED server
 * (`serve()`) produced an agent registration could never complete.
 *
 * The card itself is served by `createFetchHandler` (under the agent URL,
 * self-naming: `client_id`, `url`, `jwks_uri`, see `card.ts`). What lives
 * here is the other half: presence, and the one signed call that registers.
 */

import {
  assertSignableAgentUrl,
  signedFetch,
  type SignatureAgentForm,
  type SigningIdentity,
} from '../crypto/http-signature';

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
/**
 * Heartbeats running in this process, by agent URL (or name when there is no
 * identity). `serve()` and `registerWithPlatform` both start one, and neither
 * may start a second for the same agent (2026-09-24).
 */
const runningHeartbeats = new Map<string, HeartbeatHandle>();

/** Whether a heartbeat is already running under this key. */
export function heartbeatRunning(key: string): boolean {
  return runningHeartbeats.has(key);
}

/** Stop the heartbeat running under this key, if any. */
export function stopHeartbeat(key: string): void {
  runningHeartbeats.get(key)?.stop();
}

export function startHeartbeat(
  agentName: string,
  options: { portalApiUrl?: string; token?: string; intervalMs?: number; key?: string } = {},
): HeartbeatHandle | null {
  const portalApiUrl = resolvePortalApiUrl(options.portalApiUrl);
  const token = resolveAgentToken(options.token);
  if (!portalApiUrl || !token) {
    // Name the half that is ACTUALLY missing. This used to print "needs
    // WEBAGENTS_AGENT_TOKEN and ROBUTLER_API_URL" whenever either was absent,
    // so an operator who had set ROBUTLER_API_URL read a line saying they had
    // not and went looking in the wrong place (2026-09-07 registration pass).
    // A diagnostic that names a variable the reader can see is set teaches
    // them to stop believing the diagnostics.
    const missing = [
      !token && 'WEBAGENTS_AGENT_TOKEN',
      !portalApiUrl && 'ROBUTLER_API_URL (or ROBUTLER_INTERNAL_API_URL)',
    ].filter(Boolean) as string[];
    console.info(
      `[webagents] no heartbeat for ${agentName}: ${missing.join(' and ')} ` +
        `${missing.length > 1 ? 'are' : 'is'} not set. ` +
        'The platform will show this agent as unknown until it beats.',
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
  const key = options.key ?? agentName;
  const handle: HeartbeatHandle = {
    stop: () => {
      clearInterval(timer);
      if (runningHeartbeats.get(key) === handle) runningHeartbeats.delete(key);
    },
  };
  runningHeartbeats.set(key, handle);
  return handle;
}

// ---------------------------------------------------------------------------
// Dynamic registration
//
// Serving the card is only half of it. The platform registers an agent on the
// FIRST REQUEST THAT VERIFIES (ADR 0038 step 5, 2026-09-17): it reads the
// `Signature-Agent` header of a signed request, fetches the key set it names
// (`{agentUrl}/.well-known/jwks.json`), selects the key by the signature's
// `keyid` (the RFC 7638 thumbprint), verifies the signature over the method,
// host, path, query and body digest, and only then reads the card at
// `{agentUrl}/.well-known/agent.json`, which must name itself. Until
// something presents such a request the agent has a card nobody has read and
// no row anywhere.
//
// Before that day the credential was a bearer JWT the identity minted, and
// the one fact this comment had to get right was its audience. There is no
// audience any more: the signature covers `@authority`, so a request signed
// for `robutler.ai` verifies at `robutler.ai` and nowhere else, and the
// platform URL is simply where the request is sent.
// ---------------------------------------------------------------------------

/** The platform base URL the signed registering request is sent to. */
export function resolvePlatformBaseUrl(configured?: string): string | undefined {
  return (
    configured ||
    envVar('ROBUTLER_API_URL') ||
    envVar('ROBUTLER_INTERNAL_API_URL') ||
    undefined
  );
}

/**
 * The three operations registration needs from a secret store.
 *
 * Declared structurally rather than imported from `skills/secrets`, on
 * purpose: nothing under `server/` imports from `skills/`, and registration
 * must not pull an optional native keystore into its module graph. Anything
 * of this shape will do, and `SecretsSkill`'s store satisfies it.
 */
export interface SecretStoreLike {
  get(name: string): Promise<string | null>;
  set(name: string, value: string): Promise<unknown>;
  delete(name: string): Promise<boolean>;
}

/** Default name the platform bearer is filed under. */
export const PLATFORM_TOKEN_SECRET = 'platform_token';

/**
 * The header that carries the OPERATOR's platform API key on the registering
 * request. Always among the signature's covered components when it is sent
 * (S-184, 2026-09-19).
 */
export const OWNER_KEY_HEADER = 'X-Robutler-Owner-Key';

export interface RegisterWithPlatformOptions {
  /** Platform base URL. Falls back to ROBUTLER_API_URL. */
  platformUrl?: string;
  /**
   * Start the presence heartbeat with the bearer registration returns, when
   * none is running for this agent (default true). It used to be the caller's
   * job to export that bearer as WEBAGENTS_AGENT_TOKEN and restart, and the
   * bridge then refused the same variable (it holds no `agent_id`).
   */
  heartbeat?: boolean;
  /**
   * The OPERATOR's own platform API key, so the agent is owned from birth and
   * needs no claim flow. Falls back to `ROBUTLER_API_KEY`.
   *
   * This is your key, not the agent's: it names the human who owns the agent.
   * Without it the agent registers OWNERLESS, which works but leaves it with
   * no payer — it cannot buy inference from the platform — and on a tunnel or
   * bare-IP host it cannot be claimed afterwards either, because a DNS proof
   * needs a domain you control.
   */
  ownerApiKey?: string;
  /**
   * The `Signature-Agent` form the registering request carries (W2 design
   * section 2.4). Default `dictionary-typed`, the P-00 dictionary; the other
   * two exist so the next draft revision is a default change, not a release.
   */
  signatureAgentForm?: SignatureAgentForm;
  /**
   * Sign for a plaintext (http) agent URL. The platform admits one only
   * where `ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1` (its local overlay), and this
   * defaults to that same variable in this process's environment. Loopback
   * is refused whatever this says.
   */
  allowHttp?: boolean;
  /**
   * Where to keep the platform bearer this returns, so the next start reads
   * it instead of registering again.
   *
   * OPTIONAL, and it must stay optional: an agent on a box with no keystore
   * and no writable home still has to register. Absent, this behaves exactly
   * as it did before the store existed.
   *
   * Worth keeping when you consider what is being stored. The bearer is good
   * for seven days, carries `agents:own`, and the platform records no `jti`
   * for it, so there is no revocation lever: removing a key from the agent's
   * published key set does not invalidate an already-minted one (portal
   * security log S-037, which amplifies S-034). Re-registering on every boot
   * mints another week-long unrevocable credential each time; reading one
   * back mints none.
   */
  secrets?: SecretStoreLike;
  /** Name to file the bearer under. Defaults to `platform_token`. */
  tokenName?: string;
  /**
   * Ignore a stored bearer and register again. The local check can only read
   * `exp`; it cannot know the platform suspended the principal, so this is
   * the lever for "the stored one stopped working".
   */
  refresh?: boolean;
  /**
   * Treat a stored bearer as spent this many seconds before it actually
   * expires. Defaults to 300, so a token about to lapse mid-run is replaced
   * rather than used once and then failing.
   */
  expirySkewSeconds?: number;
}

export interface PlatformRegistrationResult {
  ok: boolean;
  /** HTTP status of the signed call. */
  status: number;
  /** The platform username the agent was registered as (reversed domain). */
  username?: string;
  /** The platform user id backing that username. */
  userId?: string;
  /**
   * Whether the agent has an OWNER, i.e. an accountable human.
   *
   * `false` is not a failure and not a warning about the call — the agent is
   * registered and will serve. It is a statement about what the agent can do:
   * an ownerless agent has no payer, so the platform will not fund inference
   * for it. It can gain an owner later in two ways: a person redeems the claim
   * link (`claimUrl`, no DNS needed), or the agent signs any later request with
   * the operator's key in `X-Robutler-Owner-Key` covered by the signature, and
   * the platform adopts it for that operator. Pass `ownerApiKey` (or set
   * `ROBUTLER_API_KEY`) to be owned from birth.
   *
   * Undefined when the platform did not report it (an older deployment).
   */
  owned?: boolean;
  /**
   * A platform bearer for this identity, valid far longer than the signed
   * request's sixty-second window. This is what `WEBAGENTS_AGENT_TOKEN` wants.
   */
  accessToken?: string;
  /**
   * True when `accessToken` came from the secret store and no token was
   * minted. `status` is 0 in that case: nothing was called.
   */
  reused?: boolean;
  /**
   * Where a freshly minted bearer was persisted, or the reason it was not.
   * A store that refuses the write does NOT fail registration, because the
   * token in hand is still good; this is how the caller finds out.
   */
  stored?: 'saved' | 'not-requested' | string;
  /** Set when `ok` is false. */
  error?: string;
}

/**
 * Seconds until a JWT's `exp`, read WITHOUT verifying the signature.
 *
 * Unverified is correct here and only here: this is our own stored copy of a
 * token we are deciding whether to reuse, so the question is "has this
 * lapsed" rather than "is this genuine". The platform verifies it properly on
 * every call. Returns null when there is no readable `exp`, which is treated
 * as unusable rather than as eternal.
 */
function secondsUntilExpiry(token: string): number | null {
  const parts = token.split('.');
  if (parts.length !== 3) return null;
  try {
    const json = Buffer.from(parts[1].replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString(
      'utf-8',
    );
    const exp = (JSON.parse(json) as { exp?: unknown }).exp;
    if (typeof exp !== 'number' || !Number.isFinite(exp)) return null;
    return exp - Math.floor(Date.now() / 1000);
  } catch {
    return null;
  }
}

/**
 * Register this agent with the platform by making one authenticated call.
 *
 * There is no registration endpoint to post to. `POST /api/auth/agent/register`
 * exists and answers **410 by design** — it used to mint an agent from an
 * unauthenticated body with no proof the caller held the key it was storing.
 * Registration is now implicit in verification, so the way to register is to
 * call an authenticated route and let the verifier do it.
 *
 * `POST /api/auth/cli/token` is the call this uses, because it is the one that
 * answers with the identity that was just minted (`user_id`, `username`) plus
 * a platform bearer the agent can keep. Any signature-accepting route
 * registers the agent equally well; this one lets the caller SEE that it
 * happened.
 *
 * Requirements the caller has to meet, all of them about reachability rather
 * than about crypto:
 *
 *  * `identity.issuer`, the agent URL (`publicUrl + basePath` as `serve()`
 *    composes it), must be an https address the PLATFORM can fetch. It
 *    resolves the key set and the card over the public internet through an
 *    SSRF guard that refuses loopback, RFC 1918, link-local and
 *    100.64.0.0/10, which includes Tailscale addresses, so a funnel host that
 *    serves the platform itself is still not a place an agent's key set can
 *    live. Plain http is admitted only by the platform's local overlay.
 *  * the key set at `{agentUrl}/.well-known/jwks.json` must list the key that
 *    signed the request, and it must survive restarts: the platform stores
 *    the thumbprints it read at registration and verifies every later
 *    request against that stored set. A new key is admitted from the
 *    published set; once the platform enforces continuity, only when a
 *    stored key co-signs, which `AgentIdentityConfig.previousKeys` does.
 */
/**
 * A URL a human can open to take ownership of this agent.
 *
 * THE PROOF IS THE KEY, so this works where DNS cannot. The agent mints a
 * short-lived, single-use claim token (the ONE JWT left in this SDK, W2
 * design section 7.2: it is not a request from the agent, so it cannot be a
 * signed request) with a key the platform pinned at registration; the person
 * opens the link while signed in; the platform selects the key by the
 * token's `kid` from the registration's key set, verifies it and records
 * them as the owner. The token proves the agent, the session proves the
 * human, and redemption binds them: neither needs to know the other in
 * advance.
 *
 * That matters most for the setup every developer has on day one: an agent
 * behind a tunnel or on a bare IP cannot be claimed by DNS at all, because you
 * cannot put a TXT record on a hostname you do not own.
 *
 * PUT IT IN THE FRAGMENT, NOT THE QUERY. A claim token is a bearer until it is
 * spent. In `?token=` it reaches the platform's access logs and any `Referer`
 * a redirect leaks; after `#` the browser keeps it client-side. Ten minutes,
 * one use, and never in a log is the whole security story.
 *
 * Prefer being owned from birth: pass `ownerApiKey` (or set
 * `ROBUTLER_API_KEY`) to `registerWithPlatform` and no claim is needed.
 */
export async function claimUrl(
  identity: { mintClaimToken(platformUrl: string, ttlSeconds?: number): Promise<string> },
  agentUserId: string,
  options: { platformUrl?: string; ttlSeconds?: number } = {},
): Promise<string | null> {
  const platformUrl = resolvePlatformBaseUrl(options.platformUrl);
  if (!platformUrl) return null;
  const base = platformUrl.replace(/\/+$/, '');
  const token = await identity.mintClaimToken(base, options.ttlSeconds ?? 600);
  return `${base}/claim/${agentUserId}#${token}`;
}

export async function registerWithPlatform(
  identity: SigningIdentity,
  options: RegisterWithPlatformOptions = {},
): Promise<PlatformRegistrationResult> {
  const result = await registerWithPlatformOnce(identity, options);
  // THE BEARER GOES TO THE HEARTBEAT HERE (2026-09-24), not through the
  // environment: registration is the one place that holds it, and exporting
  // it as WEBAGENTS_AGENT_TOKEN also handed it to the bridge, which refuses it.
  if (options.heartbeat !== false && result.ok && result.accessToken && !heartbeatRunning(identity.issuer)) {
    startHeartbeat(identity.issuer, {
      token: result.accessToken,
      portalApiUrl: resolvePortalApiUrl() ?? resolvePlatformBaseUrl(options.platformUrl),
      key: identity.issuer,
    });
  }
  return result;
}

async function registerWithPlatformOnce(
  identity: SigningIdentity,
  options: RegisterWithPlatformOptions,
): Promise<PlatformRegistrationResult> {
  const tokenName = options.tokenName ?? PLATFORM_TOKEN_SECRET;

  // Read back before signing. A stored bearer that has not lapsed is the same
  // credential a fresh registration would hand back, so calling again buys
  // nothing and costs another seven-day unrevocable token.
  if (options.secrets && !options.refresh) {
    try {
      const stored = await options.secrets.get(tokenName);
      if (stored) {
        const remaining = secondsUntilExpiry(stored);
        if (remaining !== null && remaining > (options.expirySkewSeconds ?? 300)) {
          return { ok: true, status: 0, accessToken: stored, reused: true, stored: 'saved' };
        }
        // Lapsed or unreadable. Drop it rather than leave a dead credential
        // sitting in the keystore looking live.
        await options.secrets.delete(tokenName).catch(() => false);
      }
    } catch (err) {
      // A store that cannot be read is a reason to register normally, never a
      // reason to fail. Say so, because silence here looks like a cache miss.
      console.warn(
        `[webagents] could not read "${tokenName}" from the secret store: ` +
          `${(err as Error).message}. Registering instead.`,
      );
    }
  }

  const platformUrl = resolvePlatformBaseUrl(options.platformUrl);
  if (!platformUrl) {
    return {
      ok: false,
      status: 0,
      error:
        'no platform URL: pass platformUrl or set ROBUTLER_API_URL to the ' +
        "platform's base URL",
    };
  }

  // Catch the commonest misconfigurations HERE rather than as a bare 401 from
  // the platform. `serve()` falls back to `http://localhost:<port>` when
  // WEBAGENTS_PUBLIC_URL is unset, and `localhost` is refused BY NAME on the
  // platform's side, before any address resolution; a plaintext agent URL is
  // refused by the platform outside its local overlay (operator decision 2).
  // Every other unreachable address (RFC 1918, an overlay's 100.64/10) can
  // only be judged where the fetch happens, so this checks what is decidable
  // locally and says what it is instead of guessing at the rest. The signer
  // applies the same rule and throws; catching it here turns the throw into
  // a result the caller can print.
  try {
    assertSignableAgentUrl(identity.issuer, { allowHttp: options.allowHttp });
  } catch (err) {
    return { ok: false, status: 0, error: (err as Error).message };
  }

  const platform = platformUrl.replace(/\/+$/, '');
  const tokenUrl = `${platform}/api/auth/cli/token`;

  let res: Response;
  try {
    // TWO CREDENTIALS, TWO PRINCIPALS. The signature (`Signature-Agent`,
    // `Signature-Input`, `Signature` and `Content-Digest`, added by
    // `signedFetch`) proves the agent is itself: it names the agent's key
    // set and covers the method, the platform's host, the path, the query
    // and the digest of the `{}` body. `X-Robutler-Owner-Key` carries the
    // OPERATOR's platform API key and says who owns it.
    //
    // Sending the second is what makes claiming unnecessary. Without it the
    // platform registers an ownerless agent: it works, but it has no payer, so
    // it cannot buy inference, and on a tunnel or IP host it can never be
    // claimed afterwards either. With it the agent is owned from birth.
    //
    // Optional by design, and a key the platform will not accept is ignored
    // rather than fatal: the registration still succeeds, ownerless, and the
    // result says so.
    //
    // THE OWNER KEY IS INSIDE THE SIGNATURE (S-184, fixed 2026-09-19). It
    // decides who OWNS the agent, and until that day it rode outside the
    // covered components: whoever could rewrite headers between this process
    // and the platform's TLS edge (an egress gateway, a sidecar) could swap
    // in a platform key of their own and register the agent under their
    // account, with the agent's signature still verifying. Covered, a swapped
    // or added header no longer matches the signature base, and the platform
    // honours the header on a signed request only when it is covered.
    //
    // NEVER FOLLOW A REDIRECT WITH IT (S-190, fixed 2026-09-19). Coverage
    // stops tampering, not disclosure: fetch strips only `Authorization` and
    // `Cookie` when a redirect crosses origins, so under the default
    // `redirect: 'follow'` the operator's platform key travelled to whatever
    // the Location named. A redirect is now a failed registration with a
    // message that says so, as in the Python SDK, which never followed here.
    const ownerKey = (options.ownerApiKey ?? envVar('ROBUTLER_API_KEY') ?? '').trim();
    res = await signedFetch(
      identity,
      tokenUrl,
      {
        method: 'POST',
        redirect: 'error',
        headers: {
          'Content-Type': 'application/json',
          ...(ownerKey ? { [OWNER_KEY_HEADER]: ownerKey } : {}),
        },
        body: '{}',
      },
      {
        form: options.signatureAgentForm,
        allowHttp: options.allowHttp,
        ...(ownerKey ? { coveredHeaders: [OWNER_KEY_HEADER] } : {}),
      },
    );
  } catch (err) {
    return { ok: false, status: 0, error: (err as Error).message };
  }

  if (!res.ok) {
    // 401 here is almost never a bad signature. In order of how often it is
    // actually the cause: the platform could not FETCH the key set (a
    // private, plaintext or unroutable `publicUrl`), the key set does not
    // carry the signing key's thumbprint (a key regenerated since
    // registration, `WEBAGENTS_KEYS_DIR` not persisted), or the card does not
    // name itself (`client_id`, `url` and `jwks_uri` must equal the URLs it
    // is served at). The body's `error_code` says which.
    const body = await res.text().catch(() => '');
    return {
      ok: false,
      status: res.status,
      error: `${res.status} from ${tokenUrl}: ${body.slice(0, 300)}`,
    };
  }

  const payload = (await res.json().catch(() => ({}))) as {
    access_token?: string;
    user_id?: string;
    username?: string;
    owned?: boolean;
  };
  let stored: PlatformRegistrationResult['stored'] = 'not-requested';
  if (options.secrets && payload.access_token) {
    try {
      await options.secrets.set(tokenName, payload.access_token);
      stored = 'saved';
    } catch (err) {
      // Registration SUCCEEDED. The caller holds a working bearer; only the
      // persistence failed, so report it and hand the token over rather than
      // throwing away a live credential over a storage problem.
      stored = `not stored: ${(err as Error).message}`;
      console.warn(
        `[webagents] registered, but could not persist "${tokenName}": ${(err as Error).message}`,
      );
    }
  }

  // Say it out loud. An ownerless agent is the state a developer is most
  // likely to be in without knowing, and the one that quietly costs them the
  // most: no payer, and no way to claim it later from a tunnel or IP host. The
  // SDK already warns in this voice about a missing AuthSkill and a missing
  // heartbeat; this is the same kind of half-state and deserves the same
  // treatment.
  if (payload.owned === false) {
    console.warn(
      `[webagents] registered as ${payload.username ?? 'this agent'} but it is UNCLAIMED: `
      + 'it has no owner, so the platform will not fund inference for it. Set ROBUTLER_API_KEY '
      + '(your own platform key) and restart to be owned from birth.',
    );
  }

  return {
    ok: true,
    status: res.status,
    username: payload.username,
    userId: payload.user_id,
    accessToken: payload.access_token,
    owned: payload.owned,
    reused: false,
    stored,
  };
}
