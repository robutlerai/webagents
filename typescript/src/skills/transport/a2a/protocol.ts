/**
 * A2A v1.0 protocol rules shared by the JSON-RPC and HTTP+JSON bindings of
 * `A2ATransportSkill`: which methods exist and what they alias, how the
 * version is decided, how a wire message becomes the agent's run and how the
 * run's output becomes parts. The Python twin is `a2a/protocol.py`; the
 * shared fixture `python/tests/fixtures/a2a/vectors.json` pins both.
 *
 * VERSION RULES, AND WHY THEY DEPART FROM THE SPEC (plan item 1.3,
 * 2026-09-26). Section 3.6 says a missing `A2A-Version` header MUST be read
 * as 0.3. OpenClaw's A2A channel sends no header at all and speaks v1.0
 * (`SendMessage`, `ROLE_USER`, `returnImmediately`); a2a-python reads that as
 * 0.3 and refuses it with -32009, which is exactly the interop the plan wants.
 * So the version comes from the method name when the header is missing: a
 * PascalCase method is 1.0, a dotted v0.3 alias gets 0.3-tolerant parsing
 * (`kind` on parts, bare roles, `mimeType`, `file.uri`). The tolerant parse
 * is applied to every request, since a v1 body never carries those spellings
 * and reading them costs nothing. An explicit version other than 1.0 is
 * VersionNotSupported, `0.3` included: no v0.3 client ever sent the header.
 *
 * ANSWERS ARE ALWAYS v1.0-SHAPED (`{"task": ...}`), for the dotted aliases
 * too: OpenClaw retries `message/send` with its v1 body on -32601 and reads
 * `result.task.id`, and Hermes reads `result.task` or `result.message`.
 */

import type { Message as RunMessage, ContentItem, AudioFormat } from '../../../uamp/types';
import { CREDENTIAL_HEADERS } from '../../../server/credential-floor';
import type { InboundRequestShape } from '../../../server/endpoint-gate';
import { A2AError, A2A_VERSIONS_ACCEPTED, A2A_VERSION_HEADER, type Message, type Part, type Role, type Task } from './types';

// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

export type Operation =
  | 'send'
  | 'stream'
  | 'get'
  | 'list'
  | 'cancel'
  | 'subscribe'
  | 'push_config'
  | 'extended_card';

/** v1.0 PascalCase names and the v0.3 dotted aliases, to one operation each. */
const METHODS: Record<string, { op: Operation; dotted: boolean }> = {
  SendMessage: { op: 'send', dotted: false },
  'message/send': { op: 'send', dotted: true },
  SendStreamingMessage: { op: 'stream', dotted: false },
  'message/stream': { op: 'stream', dotted: true },
  GetTask: { op: 'get', dotted: false },
  'tasks/get': { op: 'get', dotted: true },
  ListTasks: { op: 'list', dotted: false },
  CancelTask: { op: 'cancel', dotted: false },
  'tasks/cancel': { op: 'cancel', dotted: true },
  SubscribeToTask: { op: 'subscribe', dotted: false },
  'tasks/resubscribe': { op: 'subscribe', dotted: true },
  CreateTaskPushNotificationConfig: { op: 'push_config', dotted: false },
  GetTaskPushNotificationConfig: { op: 'push_config', dotted: false },
  ListTaskPushNotificationConfigs: { op: 'push_config', dotted: false },
  DeleteTaskPushNotificationConfig: { op: 'push_config', dotted: false },
  'tasks/pushNotificationConfig/set': { op: 'push_config', dotted: true },
  'tasks/pushNotificationConfig/get': { op: 'push_config', dotted: true },
  'tasks/pushNotificationConfig/list': { op: 'push_config', dotted: true },
  'tasks/pushNotificationConfig/delete': { op: 'push_config', dotted: true },
  GetExtendedAgentCard: { op: 'extended_card', dotted: false },
  'agent/getAuthenticatedExtendedCard': { op: 'extended_card', dotted: true },
};

/** The operation a JSON-RPC method names, or undefined for none. */
export function resolveMethod(method: unknown): { op: Operation; dotted: boolean } | undefined {
  return typeof method === 'string' ? METHODS[method] : undefined;
}

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

/** The `A2A-Version` a request carries: the header, else the `?A2A-Version=` query. */
export function requestedVersion(headers: { get(name: string): string | null }, url?: URL): string | null {
  const header = headers.get(A2A_VERSION_HEADER);
  if (header !== null && header.trim() !== '') return header.trim();
  const query = url?.searchParams.get(A2A_VERSION_HEADER) ?? url?.searchParams.get(A2A_VERSION_HEADER.toLowerCase());
  return query && query.trim() ? query.trim() : null;
}

/**
 * Refuse a version we do not serve. A missing version is accepted whatever
 * the method (see the file comment); the method decides the parse.
 */
export function checkVersion(version: string | null): void {
  if (version === null) return;
  if (!A2A_VERSIONS_ACCEPTED.includes(version)) {
    throw new A2AError('VERSION_NOT_SUPPORTED', `Protocol version ${version} is not supported; this agent serves A2A 1.0`, {
      requested: version,
      supported: '1.0',
    });
  }
}

// ---------------------------------------------------------------------------
// Inbound message
// ---------------------------------------------------------------------------

const CONTEXT_ID_MAX = 256;

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function readRole(value: unknown): Role {
  if (value === undefined || value === null || value === '' || value === 'ROLE_USER' || value === 'user') return 'ROLE_USER';
  if (value === 'ROLE_AGENT' || value === 'agent' || value === 'assistant') return 'ROLE_AGENT';
  throw new A2AError('INVALID_PARAMS', `message.role must be ROLE_USER or ROLE_AGENT, not ${JSON.stringify(value)}`);
}

/** One part, v1.0 or the v0.3 spelling, as a v1.0 part. */
export function readPart(raw: unknown, index: number): Part {
  if (!isRecord(raw)) throw new A2AError('INVALID_PARAMS', `message.parts[${index}] must be an object`);
  const mediaType = typeof raw.mediaType === 'string' ? raw.mediaType : typeof raw.mimeType === 'string' ? raw.mimeType : undefined;
  const filename = typeof raw.filename === 'string' ? raw.filename : typeof raw.name === 'string' ? raw.name : undefined;
  const metadata = isRecord(raw.metadata) ? raw.metadata : undefined;
  const extras = { ...(mediaType ? { mediaType } : {}), ...(filename ? { filename } : {}), ...(metadata ? { metadata } : {}) };
  if (typeof raw.text === 'string') return { text: raw.text, ...extras };
  if (typeof raw.url === 'string') return { url: raw.url, ...extras };
  if (typeof raw.raw === 'string') return { raw: raw.raw, ...extras };
  // v0.3: `{"kind": "file", "file": {"uri" | "bytes", "mimeType", "name"}}`
  if (isRecord(raw.file)) {
    const file = raw.file;
    const fileType = typeof file.mimeType === 'string' ? file.mimeType : typeof file.mediaType === 'string' ? file.mediaType : mediaType;
    const fileName = typeof file.name === 'string' ? file.name : filename;
    const fileExtras = { ...(fileType ? { mediaType: fileType } : {}), ...(fileName ? { filename: fileName } : {}), ...(metadata ? { metadata } : {}) };
    if (typeof file.uri === 'string') return { url: file.uri, ...fileExtras };
    if (typeof file.url === 'string') return { url: file.url, ...fileExtras };
    if (typeof file.bytes === 'string') return { raw: file.bytes, ...fileExtras };
    if (typeof file.data === 'string') return { raw: file.data, ...fileExtras };
  }
  if ('data' in raw && raw.data !== undefined) return { data: raw.data, ...extras };
  throw new A2AError('INVALID_PARAMS', `message.parts[${index}] carries none of text, raw, url or data`);
}

/** The `message` of a send, validated and normalised to v1.0. Ids are minted when missing. */
export function readMessage(raw: unknown): Message {
  if (!isRecord(raw)) throw new A2AError('INVALID_PARAMS', 'params.message is required');
  if (!Array.isArray(raw.parts) || raw.parts.length === 0) throw new A2AError('INVALID_PARAMS', 'message.parts must be a non-empty array');
  const contextId = raw.contextId;
  if (contextId !== undefined && contextId !== null && (typeof contextId !== 'string' || contextId.length === 0 || contextId.length > CONTEXT_ID_MAX)) {
    throw new A2AError('INVALID_PARAMS', `message.contextId must be a string of at most ${CONTEXT_ID_MAX} characters`);
  }
  const taskId = raw.taskId;
  if (taskId !== undefined && taskId !== null && (typeof taskId !== 'string' || !taskId)) {
    throw new A2AError('INVALID_PARAMS', 'message.taskId must be a non-empty string');
  }
  const message: Message = {
    messageId: typeof raw.messageId === 'string' && raw.messageId ? raw.messageId : crypto.randomUUID(),
    role: readRole(raw.role),
    parts: raw.parts.map((part, index) => readPart(part, index)),
  };
  if (typeof contextId === 'string') message.contextId = contextId;
  if (typeof taskId === 'string') message.taskId = taskId;
  if (isRecord(raw.metadata)) message.metadata = raw.metadata;
  if (Array.isArray(raw.extensions)) message.extensions = raw.extensions.filter((e): e is string => typeof e === 'string');
  if (Array.isArray(raw.referenceTaskIds)) message.referenceTaskIds = raw.referenceTaskIds.filter((e): e is string => typeof e === 'string');
  return message;
}

export interface SendConfiguration {
  returnImmediately: boolean;
  historyLength?: number;
}

/** `params.configuration`: push delivery is refused here, before any task exists. */
export function readConfiguration(raw: unknown): SendConfiguration {
  if (raw === undefined || raw === null) return { returnImmediately: false };
  if (!isRecord(raw)) throw new A2AError('INVALID_PARAMS', 'params.configuration must be an object');
  if (raw.taskPushNotificationConfig !== undefined && raw.taskPushNotificationConfig !== null) {
    throw new A2AError('PUSH_NOTIFICATION_NOT_SUPPORTED');
  }
  const historyLength = typeof raw.historyLength === 'number' && raw.historyLength >= 0 ? Math.floor(raw.historyLength) : undefined;
  return { returnImmediately: raw.returnImmediately === true, ...(historyLength !== undefined ? { historyLength } : {}) };
}

// ---------------------------------------------------------------------------
// Parts to and from the agent's run
// ---------------------------------------------------------------------------

function mediaKind(mediaType: string | undefined): 'image' | 'audio' | 'video' | 'file' {
  const type = (mediaType ?? '').toLowerCase();
  if (type.startsWith('image/')) return 'image';
  if (type.startsWith('audio/')) return 'audio';
  if (type.startsWith('video/')) return 'video';
  return 'file';
}

/** A v1.0 part as one of the run's content items. */
export function partToContentItem(part: Part): ContentItem {
  if (part.text !== undefined) return { type: 'text', text: part.text };
  if (part.data !== undefined) {
    const label = part.mediaType ? `[data ${part.mediaType}] ` : '[data] ';
    return { type: 'text', text: label + JSON.stringify(part.data) };
  }
  const kind = mediaKind(part.mediaType);
  const mediaType = part.mediaType ?? 'application/octet-stream';
  if (part.url !== undefined) {
    if (kind === 'image') return { type: 'image', image: { url: part.url } };
    if (kind === 'audio') return { type: 'audio', audio: { url: part.url } };
    if (kind === 'video') return { type: 'video', video: { url: part.url } };
    return { type: 'file', file: { url: part.url }, filename: part.filename ?? 'attachment', mime_type: mediaType };
  }
  const raw = part.raw ?? '';
  const dataUrl = `data:${mediaType};base64,${raw}`;
  if (kind === 'image') return { type: 'image', image: dataUrl };
  if (kind === 'audio') return { type: 'audio', audio: raw, format: (mediaType.split('/')[1] ?? 'wav') as AudioFormat };
  if (kind === 'video') return { type: 'video', video: dataUrl };
  return { type: 'file', file: dataUrl, filename: part.filename ?? 'attachment', mime_type: mediaType };
}

/** A v1.0 message as one of the run's messages: text joined, media as content items. */
export function messageToRunMessage(message: Message): RunMessage {
  const items = message.parts.map(partToContentItem);
  const texts = items.filter((i): i is Extract<ContentItem, { type: 'text' }> => i.type === 'text').map((i) => i.text);
  const media = items.filter((i) => i.type !== 'text');
  return {
    role: message.role === 'ROLE_AGENT' ? 'assistant' : 'user',
    content: texts.join('\n'),
    ...(media.length ? { content_items: media } : {}),
  };
}

/** The run's output as v1.0 parts: the text first, then any media it produced. */
export function outputToParts(content: string, items: ContentItem[] | undefined): Part[] {
  const parts: Part[] = [{ text: content ?? '' }];
  for (const item of items ?? []) {
    switch (item.type) {
      case 'image':
        parts.push(mediaPart(item.image, 'image/png'));
        break;
      case 'audio':
        parts.push(mediaPart(item.audio, `audio/${item.format ?? 'wav'}`));
        break;
      case 'video':
        parts.push(mediaPart(item.video, 'video/mp4'));
        break;
      case 'file':
        parts.push({ ...mediaPart(item.file, item.mime_type), filename: item.filename });
        break;
      default:
        break;
    }
  }
  return parts;
}

function mediaPart(source: string | { url: string }, mediaType: string): Part {
  if (typeof source !== 'string') return { url: source.url, mediaType };
  const dataUrl = /^data:([^;,]+)(?:;base64)?,(.*)$/s.exec(source);
  if (dataUrl) return { raw: dataUrl[2], mediaType: dataUrl[1] || mediaType };
  if (/^https?:\/\//.test(source)) return { url: source, mediaType };
  return { raw: source, mediaType };
}

/** The text a task's output carries, artifacts first, then the status message (what Hermes reads). */
export function replyText(task: Task): string {
  const fromArtifacts = task.artifacts.flatMap((a) => a.parts).map((p) => p.text).filter((t): t is string => typeof t === 'string');
  if (fromArtifacts.length) return fromArtifacts.join('');
  return (task.status.message?.parts ?? []).map((p) => p.text).filter((t): t is string => typeof t === 'string').join('');
}

// ---------------------------------------------------------------------------
// The caller
// ---------------------------------------------------------------------------

/**
 * Request metadata for the run, the shape the fetch handler gives
 * `/chat/completions`: the credential headers `AuthSkill` reads, the method
 * and user agent, and the payment token when one came.
 */
export function runMetadata(inbound: InboundRequestShape): Record<string, unknown> {
  const metadata: Record<string, unknown> = { method: inbound.method, userAgent: inbound.headers['user-agent'] ?? null, transport: 'a2a' };
  for (const name of CREDENTIAL_HEADERS) {
    const value = inbound.headers[name];
    if (value) metadata[name] = value;
  }
  const paymentToken = inbound.headers['x-payment-token'];
  if (paymentToken) metadata['x-payment-token'] = paymentToken;
  return metadata;
}

/**
 * The key a task is owned under. A caller the agent verified is its
 * principal (the access block's, else the auth skill's user or agent id);
 * one it could not verify is the credential it presented, hashed, so the
 * same bearer reads back its own tasks and nobody else's. No credential at
 * all is the anonymous key, which the floor keeps off every task route.
 */
export async function callerKey(auth: unknown, inbound: InboundRequestShape): Promise<string> {
  const info = (auth ?? {}) as { authenticated?: boolean; principals?: unknown; user_id?: unknown; agent_id?: unknown; agentId?: unknown; claims?: { sub?: unknown } };
  if (info.authenticated) {
    const principal = Array.isArray(info.principals) ? info.principals.find((p): p is string => typeof p === 'string') : undefined;
    const id = principal ?? info.user_id ?? info.agent_id ?? info.agentId ?? info.claims?.sub;
    if (typeof id === 'string' && id) return `id:${id}`;
  }
  const credential = CREDENTIAL_HEADERS.map((name) => inbound.headers[name]).find((v) => !!v) ?? inbound.headers['signature'];
  if (!credential) return 'anonymous';
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(credential));
  return `cred:${Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')}`;
}

/** ISO 8601 UTC with `Z`, as the spec's timestamps are written. */
export function nowIso(): string {
  return new Date().toISOString();
}
