/**
 * The A2A v1.0 client: how an agent built with this SDK calls a peer that
 * speaks A2A. The Python twin is `a2a/a2a_client.py`; the shared fixture
 * `python/tests/fixtures/a2a/vectors.json` (`client`) pins the interface
 * choice and the result shapes both read. Plan item 1.3, 2026-09-26.
 *
 * DISCOVERY IS HERMES' WALK (spec pack 3.7): `GET {base}/.well-known/agent-card.json`,
 * then `/.well-known/agent.json` on a 404, then the first `supportedInterfaces`
 * entry whose binding is `JSONRPC` and whose version is 1.0. Hermes takes the
 * first JSONRPC entry without checking its version; the version is checked
 * here because a v0.3 card (`url` plus `preferredTransport`, no
 * `supportedInterfaces`) names no interface a v1.0 `SendMessage` could reach,
 * and refusing it at the card is clearer than a -32601 or a parse error one
 * hop later.
 *
 * THE CALL is one JSON-RPC `SendMessage` with `A2A-Version: 1.0`, the text as
 * a `ROLE_USER` part with a `mediaType` (what Hermes sends), a bearer
 * configured out of band (the agent file's `peers`, `peerTokenFor`; never
 * discovered), and ONE retry as `message/send` with the same body on -32601,
 * the way OpenClaw retries. Both result shapes are read, `{task}` and
 * `{message}`, plus a bare legacy task or message, and a task that is not yet
 * settled is polled with `GetTask` until it is or the deadline passes. A 3xx
 * is refused: an A2A endpoint is called where it is, never followed.
 *
 * A card's signature is checked when asked (`verifyCard`), with the key
 * fetched from the signature's `jku`, which `card.ts` allows only at the
 * card's own origin.
 */

import {
  AGENT_CARD_WELL_KNOWN_SUFFIX,
  LEGACY_CARD_WELL_KNOWN_SUFFIX,
  verifyAgentCard,
  type VerifyCardOptions,
  type VerifyCardResult,
} from './card';
import { replyText } from './protocol';
import { A2A_VERSION, A2A_VERSION_HEADER, A2A_VERSIONS_ACCEPTED, isSettled, type Message, type Task } from './types';

/** What OpenClaw and Hermes wait for a blocking send. */
export const A2A_CLIENT_TIMEOUT_MS = 120_000;
export const A2A_POLL_INTERVAL_MS = 500;
export const METHOD_NOT_FOUND = -32601;

/** The v0.3 spelling a v1.0 method is retried under, once, on -32601. */
const DOTTED_ALIASES: Record<string, string> = {
  SendMessage: 'message/send',
  GetTask: 'tasks/get',
  CancelTask: 'tasks/cancel',
};

export interface A2AClientOptions {
  /** The bearer the peer gets, configured out of band. */
  token?: string;
  /** Extra headers on every request. */
  headers?: Record<string, string>;
  /** Per-request timeout, and the deadline for polling a task. Default 120 s. */
  timeoutMs?: number;
  fetch?: typeof fetch;
}

export class A2AClientError extends Error {
  /** The JSON-RPC error code, when the peer answered one. */
  readonly code?: number;
  /** The HTTP status, when that is what went wrong. */
  readonly status?: number;
  readonly data?: unknown;

  constructor(message: string, options: { code?: number; status?: number; data?: unknown } = {}) {
    super(message);
    this.name = 'A2AClientError';
    this.code = options.code;
    this.status = options.status;
    this.data = options.data;
  }
}

export interface FetchedCard {
  card: Record<string, unknown>;
  /** Where the card was found: the v1.0 path or the legacy one. */
  cardUrl: string;
}

export interface PickedInterface {
  url: string;
  tenant?: string;
}

export type SendResult = { task: Task } | { message: Message };

export interface CallOptions extends A2AClientOptions {
  /** The conversation on the peer's side. */
  contextId?: string;
  /** Ask for the task at once and poll, rather than a blocking send. */
  returnImmediately?: boolean;
  pollIntervalMs?: number;
  /**
   * Refuse a card whose signature does not verify (`card.ts`). `true`
   * verifies every card, so an unsigned one is refused; `'jku'` verifies a
   * card whose signature names a `jku` and lets an unsigned card through as
   * unverified (`delegate` asks for this: OpenClaw's card is unsigned). Off
   * by default.
   */
  verifyCard?: boolean | 'jku';
  /** Options for that verification: held keys, `allowHttp` for a loopback peer. */
  verify?: VerifyCardOptions;
  /** A card already fetched from `baseUrl` (a probe), so discovery is not repeated. */
  card?: FetchedCard;
}

export interface CallResult extends FetchedCard {
  rpcUrl: string;
  verified?: VerifyCardResult;
  task?: Task;
  message?: Message;
  /** The peer's text: artifacts first, then the status message (what Hermes reads). */
  reply: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** One request with the client's headers, bearer and timeout; a redirect is refused. */
async function request(
  url: string,
  init: { method: string; headers: Record<string, string>; body?: string },
  options: A2AClientOptions,
): Promise<Response> {
  const doFetch = options.fetch ?? fetch;
  const controller = new AbortController();
  const timeoutMs = options.timeoutMs ?? A2A_CLIENT_TIMEOUT_MS;
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const headers: Record<string, string> = {
      ...init.headers,
      ...(options.headers ?? {}),
      ...(options.token ? { authorization: `Bearer ${options.token}` } : {}),
    };
    const response = await doFetch(url, {
      method: init.method,
      headers,
      ...(init.body !== undefined ? { body: init.body } : {}),
      redirect: 'manual',
      signal: controller.signal,
    });
    if (response.status >= 300 && response.status < 400) {
      throw new A2AClientError(`${url} answered a redirect (${response.status}); an A2A endpoint is called where it is`, { status: response.status });
    }
    return response;
  } catch (error) {
    if (controller.signal.aborted) throw new A2AClientError(`${url} did not answer within ${timeoutMs} ms`);
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

/** The peer's card: the v1.0 path, then the legacy path on a 404. */
export async function fetchAgentCard(baseUrl: string, options: A2AClientOptions = {}): Promise<FetchedCard> {
  const base = baseUrl.replace(/\/+$/, '');
  for (const suffix of [AGENT_CARD_WELL_KNOWN_SUFFIX, LEGACY_CARD_WELL_KNOWN_SUFFIX]) {
    const url = `${base}${suffix}`;
    const response = await request(url, { method: 'GET', headers: { accept: 'application/json' } }, options);
    if (response.status === 404) continue;
    if (!response.ok) throw new A2AClientError(`${url} answered ${response.status}`, { status: response.status });
    let card: unknown;
    try {
      card = await response.json();
    } catch {
      throw new A2AClientError(`${url} is not JSON`);
    }
    if (!isRecord(card)) throw new A2AClientError(`${url} is not a JSON object`);
    return { card, cardUrl: url };
  }
  throw new A2AClientError(`no agent card at ${base}: ${AGENT_CARD_WELL_KNOWN_SUFFIX} and ${LEGACY_CARD_WELL_KNOWN_SUFFIX} both answered 404`, { status: 404 });
}

function isVersion10(value: unknown): boolean {
  return typeof value === 'string' && A2A_VERSIONS_ACCEPTED.includes(value.trim());
}

/** Whether any of the card's signatures names a `jku` in its protected header: what `verifyCard: 'jku'` verifies against. */
export function cardNamesJku(card: Record<string, unknown>): boolean {
  const signatures = Array.isArray(card.signatures) ? card.signatures : [];
  for (const entry of signatures) {
    if (!isRecord(entry) || typeof entry.protected !== 'string') continue;
    try {
      const padded = entry.protected.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (entry.protected.length % 4)) % 4);
      const header = JSON.parse(atob(padded)) as { jku?: unknown };
      if (typeof header.jku === 'string' && header.jku) return true;
    } catch {
      // not a readable header: nothing named
    }
  }
  return false;
}

/** The first JSON-RPC interface at protocol version 1.0, with its tenant; null when the card names none. */
export function pickInterface(card: Record<string, unknown>): PickedInterface | null {
  const interfaces = Array.isArray(card.supportedInterfaces) ? card.supportedInterfaces : [];
  for (const entry of interfaces) {
    if (!isRecord(entry) || entry.protocolBinding !== 'JSONRPC' || !isVersion10(entry.protocolVersion)) continue;
    if (typeof entry.url !== 'string' || !entry.url) continue;
    return { url: entry.url, ...(typeof entry.tenant === 'string' && entry.tenant ? { tenant: entry.tenant } : {}) };
  }
  return null;
}

/** A `ROLE_USER` text message, the part with a `mediaType` as Hermes sends it. */
export function textMessage(text: string, options: { contextId?: string; messageId?: string } = {}): Message {
  return {
    messageId: options.messageId ?? crypto.randomUUID(),
    role: 'ROLE_USER',
    parts: [{ text, mediaType: 'text/plain' }],
    ...(options.contextId ? { contextId: options.contextId } : {}),
  };
}

/** The text of a message result. */
export function messageText(message: Message): string {
  return (message.parts ?? []).map((p) => p.text).filter((t): t is string => typeof t === 'string').join('');
}

/**
 * A send's result as either SDK answers it (`{task}` or `{message}`), or as a
 * legacy server does (the bare task or message).
 */
export function readSendResult(result: unknown): SendResult {
  if (!isRecord(result)) throw new A2AClientError('the peer answered no result object');
  if (isRecord(result.task)) return { task: result.task as unknown as Task };
  if (isRecord(result.message)) return { message: result.message as unknown as Message };
  if (typeof result.id === 'string' && isRecord(result.status)) {
    const task = result as unknown as Task;
    return { task: { ...task, artifacts: task.artifacts ?? [], history: task.history ?? [] } };
  }
  if (Array.isArray(result.parts)) return { message: result as unknown as Message };
  throw new A2AClientError('the peer answered neither a task nor a message');
}

/** The reply text of a send result. */
export function replyOfResult(result: SendResult): string {
  return 'task' in result ? replyText(result.task) : messageText(result.message);
}

/**
 * One JSON-RPC call with `A2A-Version: 1.0`; on -32601 the v0.3 alias is
 * tried once with the same params. A JSON-RPC error is thrown with its code.
 */
export async function rpc(rpcUrl: string, method: string, params: Record<string, unknown>, options: A2AClientOptions = {}): Promise<unknown> {
  const call = async (name: string): Promise<unknown> => {
    const response = await request(
      rpcUrl,
      {
        method: 'POST',
        headers: { 'content-type': 'application/json', accept: 'application/json', [A2A_VERSION_HEADER]: A2A_VERSION },
        body: JSON.stringify({ jsonrpc: '2.0', id: crypto.randomUUID(), method: name, params }),
      },
      options,
    );
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new A2AClientError(`${rpcUrl} answered ${name} with ${response.status} and no JSON`, { status: response.status });
    }
    if (!isRecord(body)) throw new A2AClientError(`${rpcUrl} answered ${name} with something other than a JSON-RPC response`, { status: response.status });
    if (isRecord(body.error)) {
      const code = typeof body.error.code === 'number' ? body.error.code : undefined;
      throw new A2AClientError(typeof body.error.message === 'string' ? body.error.message : `${name} failed`, { code, status: response.status, data: body.error.data });
    }
    if (!response.ok) throw new A2AClientError(`${rpcUrl} answered ${response.status} to ${name}`, { status: response.status, data: body });
    return body.result;
  };
  try {
    return await call(method);
  } catch (error) {
    const alias = DOTTED_ALIASES[method];
    if (alias && error instanceof A2AClientError && error.code === METHOD_NOT_FOUND) return call(alias);
    throw error;
  }
}

/** `SendMessage` to a JSON-RPC interface. */
export async function sendMessage(
  rpcUrl: string,
  message: Message,
  options: A2AClientOptions & { configuration?: Record<string, unknown>; tenant?: string } = {},
): Promise<SendResult> {
  const { configuration, tenant, ...rest } = options;
  const params: Record<string, unknown> = { message, ...(configuration ? { configuration } : {}), ...(tenant ? { tenant } : {}) };
  return readSendResult(await rpc(rpcUrl, 'SendMessage', params, rest));
}

/** `GetTask` on a JSON-RPC interface. */
export async function getTask(rpcUrl: string, id: string, options: A2AClientOptions & { tenant?: string } = {}): Promise<Task> {
  const { tenant, ...rest } = options;
  const result = readSendResult(await rpc(rpcUrl, 'GetTask', { id, ...(tenant ? { tenant } : {}) }, rest));
  if (!('task' in result)) throw new A2AClientError(`GetTask ${id} answered a message, not a task`);
  return result.task;
}

/**
 * Call a peer: its card, the interface, one send, and the task polled to a
 * settled state. `input` is a text or a whole message.
 */
export async function callAgent(baseUrl: string, input: string | Message, options: CallOptions = {}): Promise<CallResult> {
  const fetched = options.card ?? (await fetchAgentCard(baseUrl, options));
  let verified: VerifyCardResult | undefined;
  const wantsVerification = options.verifyCard === true || (options.verifyCard === 'jku' && cardNamesJku(fetched.card));
  if (wantsVerification) {
    verified = await verifyAgentCard(fetched.card, { cardUrl: fetched.cardUrl, ...(options.fetch ? { fetch: options.fetch } : {}), ...(options.verify ?? {}) });
    if (!verified.ok) throw new A2AClientError(`the card at ${fetched.cardUrl} does not verify: ${verified.reason ?? 'no signature verified'}`);
  }
  const iface = pickInterface(fetched.card);
  if (!iface) throw new A2AClientError(`${fetched.cardUrl} names no A2A 1.0 JSON-RPC interface`);
  const message = typeof input === 'string' ? textMessage(input, { contextId: options.contextId }) : input;
  const deadline = Date.now() + (options.timeoutMs ?? A2A_CLIENT_TIMEOUT_MS);
  const result = await sendMessage(iface.url, message, {
    ...options,
    tenant: iface.tenant,
    ...(options.returnImmediately ? { configuration: { returnImmediately: true } } : {}),
  });
  if ('message' in result) {
    return { ...fetched, rpcUrl: iface.url, verified, message: result.message, reply: messageText(result.message) };
  }
  let task = result.task;
  while (!isSettled(task.status.state) && Date.now() < deadline) {
    await sleep(options.pollIntervalMs ?? A2A_POLL_INTERVAL_MS);
    task = await getTask(iface.url, task.id, { ...options, tenant: iface.tenant });
  }
  return { ...fetched, rpcUrl: iface.url, verified, task, reply: replyText(task) };
}
