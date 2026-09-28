/**
 * Where a scheduled run's reply goes (plan item 1.7, 2026-09-26): the
 * `deliver` target of a `cron:` schedule (`agents/schedules.ts`), one
 * deliverer per kind, the same words and bytes as
 * `python/webagents/cli/daemon/deliver.py`, held by the shared fixture
 * (`python/tests/fixtures/daemon/cron.json`: `details`, `webhook`, `chat`).
 *
 * `file` appends an entry to a path inside the agent's folder. The loader
 * already refused a path that names its way out (`..`, an absolute path);
 * this checks the REAL path too, because a link inside the folder can point
 * outside it, and appending through one would write where the file never
 * said.
 *
 * `webhook` POSTs the run as JSON, signed the way the REST tool signs an
 * outbound request (Web Bot Auth, `crypto/http-signature.ts`) when the agent
 * holds a signing identity (`agent.identity`, what `serve()` attaches); an
 * agent without one posts unsigned and the run's record says so, as the REST
 * tool reports its own unsigned requests. A hop that cannot be reached, or
 * that answers 408, 429 or 5xx, is tried again up to `retries` times with
 * doubling backoff (1, 2, 4 ... capped at 30 s); any other 4xx is final,
 * since asking again does not change the answer. Every try is signed afresh:
 * a signature carries a nonce and a 60 s window, and a replayed one is
 * refused by any verifier that keeps nonces.
 *
 * `chat` records the run into the owner's chat with the agent on Robutler,
 * through the route the chat's `robutler` session backend uses
 * (`cli/robutler-sessions.ts`, `POST /api/agents/{id}/conversations`): the
 * schedule's prompt as the owner's words, the reply as the agent's, a
 * heartbeat's report alone. The session id is uuid5 of the agent and
 * schedule names in the URL namespace (`chatSessionId`), so every run of one
 * schedule lands in the same chat and nothing has to be stored for it. It
 * needs the person signed in and the agent published, as the chat does;
 * without either the run fails and its record says which.
 *
 * `channel` is the channel relay's, refused by the loader until it exists;
 * a kind with no deliverer is reported as such in the run's record rather
 * than silently dropped, so a schedule never looks delivered when nothing
 * happened.
 *
 * The seams on `DeliveryContext` (`fetch`, `sleep`, `chat`) exist for the
 * tests, which deliver to a local server and a mocked platform client.
 */

import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';

import type { DeliverTarget } from '../agents/schedules';
import type { ConversationsTarget, Unavailable } from '../cli/robutler-sessions';
import { signMessage, type SigningIdentity } from '../crypto/http-signature';

/** `delivered`, `nothing` (a heartbeat with nothing to report) or `failed`, with a detail. */
export type Outcome = readonly [outcome: 'delivered' | 'nothing' | 'failed', detail: string];

/** One appended entry (the fixture's `file_entry`). */
export const FILE_ENTRY = '## {schedule}, {ran_at}\n\n{content}\n\n';
/** Seconds waited before each retry of a webhook (the fixture's `webhook.backoff_seconds`). */
export const WEBHOOK_BACKOFF_SECONDS: readonly number[] = [1, 2, 4, 8, 16, 30, 30, 30, 30, 30];
/** Answers worth a retry beside 5xx (the fixture's `webhook.retry_statuses`). */
export const WEBHOOK_RETRY_STATUSES: ReadonlySet<number> = new Set([408, 429]);
/** The uuid5 namespace a chat session id is derived in (RFC 4122's URL namespace, the fixture's `chat.namespace`). */
export const CHAT_SESSION_NAMESPACE = '6ba7b811-9dad-11d1-80b4-00c04fd430c8';
/** Characters one recorded message keeps (`cli/robutler-sessions.ts` RECORD_CHARS, the platform's limit). */
const CHAT_RECORD_CHARS = 100_000;

/** What one scheduled turn produced. */
export interface RunResult {
  agent: string;
  schedule: string;
  /** `cron` or `every`. */
  kind: 'cron' | 'every';
  /** The message the turn was given; absent for a heartbeat. */
  prompt?: string;
  content: string;
  /** When the slot fired, ISO 8601 UTC to the second. */
  ranAt: string;
}

/** One recorded turn, as the platform takes it (`recordPlatformTurn`). */
export interface ChatTurn {
  sessionId: string;
  messages: { role: 'user' | 'assistant'; content: string }[];
}

/** The platform client the chat deliverer posts through; the daemon's is `platformChatClient`. */
export interface ChatClient {
  /** Where and as whom, for this agent, or why not. */
  target(agentDir: string, agentName: string): Promise<ConversationsTarget | Unavailable>;
  /** Record the turn; answers the chat's id. */
  record(target: ConversationsTarget, turn: ChatTurn, command: (rest: string) => string): Promise<string>;
}

/** Where the agent lives and, when a deliverer signs or posts as it, the agent itself. */
export interface DeliveryContext {
  agentDir: string;
  agent?: unknown;
  /** The webhook deliverer's fetch; global `fetch` by default. */
  fetch?: typeof fetch;
  /** The webhook deliverer's wait between tries; a real sleep by default. */
  sleep?: (seconds: number) => Promise<void>;
  /** The chat deliverer's platform client; the person's sign-in and the folder's link by default. */
  chat?: ChatClient;
}

export type Deliverer = (target: DeliverTarget, result: RunResult, ctx: DeliveryContext) => Promise<Outcome>;

function describeError(error: unknown): string {
  const message = (error as { message?: unknown } | undefined)?.message;
  return typeof message === 'string' && message ? message : String(error);
}

/** `1 try` or `N tries` (the fixture's `{tries}`). */
export function tries(n: number): string {
  return n === 1 ? '1 try' : `${n} tries`;
}

// -- file -----------------------------------------------------------------------------------

/** Append the entry to `target.path` under the agent's folder, never outside it. */
export async function deliverFile(target: DeliverTarget, result: RunResult, ctx: DeliveryContext): Promise<Outcome> {
  if (target.kind !== 'file') return ['failed', 'not a file target'];
  const root = fs.realpathSync(path.resolve(ctx.agentDir));
  const wanted = path.resolve(root, target.path);
  fs.mkdirSync(path.dirname(wanted), { recursive: true });
  let dest = path.join(fs.realpathSync(path.dirname(wanted)), path.basename(wanted));
  // The file itself may be a link, and a DANGLING one is invisible to
  // `existsSync`: appending through it would create the target outside.
  try {
    if (fs.lstatSync(dest).isSymbolicLink()) {
      const linked = path.resolve(path.dirname(dest), fs.readlinkSync(dest));
      dest = fs.existsSync(linked) ? fs.realpathSync(linked) : linked;
    }
  } catch {
    // Not there yet: it will be created inside.
  }
  if (dest !== root && !dest.startsWith(root + path.sep)) {
    return ['failed', `file ${target.path}: outside the agent's folder`];
  }
  const entry = FILE_ENTRY.replace('{schedule}', result.schedule).replace('{ran_at}', result.ranAt).replace('{content}', result.content);
  fs.appendFileSync(dest, entry, 'utf-8');
  return ['delivered', `file ${target.path}`];
}

// -- webhook --------------------------------------------------------------------------------

/** The JSON a webhook receives (the fixture's `webhook.body`): these keys, this order. */
export function webhookBody(result: RunResult): string {
  return JSON.stringify({
    agent: result.agent,
    schedule: result.schedule,
    kind: result.kind,
    prompt: result.prompt ?? null,
    content: result.content,
    ran_at: result.ranAt,
  });
}

/** The agent's signing identity, where `serve()` leaves it (the REST tool reads the same). */
function signingIdentityOf(agent: unknown): SigningIdentity | undefined {
  const candidate = (agent as { identity?: unknown } | undefined)?.identity as { issuer?: unknown; getHeldKeys?: unknown } | undefined;
  if (candidate && typeof candidate.issuer === 'string' && typeof candidate.getHeldKeys === 'function') {
    return candidate as SigningIdentity;
  }
  return undefined;
}

/** The signature headers for one try, and the words for the record: `signed as ...` or `unsigned: ...`. */
async function signedHeaders(agent: unknown, url: string, body: Uint8Array): Promise<{ headers: Record<string, string>; signing: string }> {
  const identity = signingIdentityOf(agent);
  if (!identity) return { headers: {}, signing: 'unsigned: this agent holds no signing key' };
  try {
    const signed = await signMessage(identity, { method: 'POST', url, body });
    const headers: Record<string, string> = {
      'Signature-Agent': signed.headers['signature-agent'],
      'Signature-Input': signed.headers['signature-input'],
      Signature: signed.headers.signature,
    };
    if (signed.headers['content-digest']) headers['Content-Digest'] = signed.headers['content-digest'];
    return { headers, signing: `signed as ${identity.issuer}` };
  } catch (err) {
    return { headers: {}, signing: `unsigned: this agent's key could not sign: ${describeError(err)}` };
  }
}

const realSleep = (seconds: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, seconds * 1000));

/** POST the run to `target.url` (file comment, `webhook`). */
export async function deliverWebhook(target: DeliverTarget, result: RunResult, ctx: DeliveryContext): Promise<Outcome> {
  if (target.kind !== 'webhook') return ['failed', 'not a webhook target'];
  const fetchFn = ctx.fetch ?? fetch;
  const sleep = ctx.sleep ?? realSleep;
  const text = webhookBody(result);
  const body = new TextEncoder().encode(text);
  const attempts = target.retries + 1;
  let last: { status?: number } = {};
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const { headers, signing } = await signedHeaders(ctx.agent, target.url, body);
    try {
      const response = await fetchFn(target.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...headers },
        body,
        signal: AbortSignal.timeout(target.timeout * 1000),
      });
      try {
        await response.body?.cancel();
      } catch {
        // The answer's bytes are not the point.
      }
      if (response.ok) return ['delivered', `webhook ${target.url} (${signing})`];
      last = { status: response.status };
      if (!(WEBHOOK_RETRY_STATUSES.has(response.status) || response.status >= 500)) {
        return ['failed', `webhook ${target.url}: answered ${response.status} after ${tries(attempt)}`];
      }
    } catch {
      last = {};
    }
    if (attempt < attempts) await sleep(WEBHOOK_BACKOFF_SECONDS[Math.min(attempt - 1, WEBHOOK_BACKOFF_SECONDS.length - 1)]);
  }
  const reason = last.status === undefined ? 'could not be reached' : `answered ${last.status}`;
  return ['failed', `webhook ${target.url}: ${reason} after ${tries(attempts)}`];
}

// -- chat -----------------------------------------------------------------------------------

/** uuid5 of `webagents:cron:<agent>/<schedule>` in the URL namespace (the fixture's `chat.sessions`). */
export function chatSessionId(agent: string, schedule: string): string {
  const namespace = Buffer.from(CHAT_SESSION_NAMESPACE.replace(/-/g, ''), 'hex');
  const digest = createHash('sha1').update(namespace).update(Buffer.from(`webagents:cron:${agent}/${schedule}`, 'utf8')).digest();
  const bytes = Buffer.from(digest.subarray(0, 16));
  bytes[6] = (bytes[6] & 0x0f) | 0x50;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = bytes.toString('hex');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/** The turn the run records (the fixture's `chat.turn` and `chat.heartbeat_turn`). */
export function chatTurn(result: RunResult): ChatTurn {
  const messages: ChatTurn['messages'] = [];
  if (result.prompt !== undefined) messages.push({ role: 'user', content: result.prompt.slice(0, CHAT_RECORD_CHARS) });
  messages.push({ role: 'assistant', content: result.content.slice(0, CHAT_RECORD_CHARS) });
  return { sessionId: chatSessionId(result.agent, result.schedule), messages };
}

/** The daemon's platform client: the person's sign-in, the configured platform, the folder's link. */
async function platformChatClient(): Promise<ChatClient> {
  const { getToken } = await import('../cli/credentials.js');
  const { resolvePlatformUrl } = await import('../cli/config-store.js');
  const { linkedPlatformAgent, recordPlatformTurn } = await import('../cli/robutler-sessions.js');
  return {
    async target(agentDir, agentName) {
      const token = await getToken();
      if (!token) return 'signed_out';
      const agentId = linkedPlatformAgent(agentDir, agentName);
      if (!agentId) return 'not_published';
      return { base: resolvePlatformUrl()[0], token, agentId };
    },
    record: recordPlatformTurn,
  };
}

/** Record the run into the owner's chat with the agent (file comment, `chat`). */
export async function deliverChat(target: DeliverTarget, result: RunResult, ctx: DeliveryContext): Promise<Outcome> {
  if (target.kind !== 'chat') return ['failed', 'not a chat target'];
  const { cliCommand } = await import('../cli/config-store.js');
  const client = ctx.chat ?? (await platformChatClient());
  const where = await client.target(ctx.agentDir, result.agent);
  if (where === 'signed_out') return ['failed', `chat owner: not signed in: run \`${cliCommand('login')}\``];
  if (where === 'not_published') return ['failed', `chat owner: this agent is not on Robutler: run \`${cliCommand('publish')}\``];
  try {
    const chatId = await client.record(where, chatTurn(result), cliCommand);
    return ['delivered', `chat owner (${chatId})`];
  } catch (err) {
    return ['failed', `chat owner: ${describeError(err)}`];
  }
}

// -- dispatch -------------------------------------------------------------------------------

/** The deliverer for each `deliver` kind. */
export const DELIVERERS: Map<string, Deliverer> = new Map([
  ['file', deliverFile],
  ['webhook', deliverWebhook],
  ['chat', deliverChat],
]);

/** Deliver `result` where `target` says; a kind nothing serves is a failed run, said. */
export async function deliver(target: DeliverTarget, result: RunResult, ctx: DeliveryContext): Promise<Outcome> {
  const deliverer = DELIVERERS.get(target.kind);
  if (!deliverer) return ['failed', `${target.kind} delivery is not available in this build`];
  return deliverer(target, result, ctx);
}
