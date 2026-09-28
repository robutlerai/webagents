/**
 * The `access:` block of an agent file, and the decision it makes (ADR-0045).
 *
 *     access:
 *       deny:    [agent:https://spam.example/**]
 *       groups:
 *         admins:  [user:@alice]
 *         friends: [agent:https://*.acme.com/**, key:<thumbprint>, domain:partner.example]
 *         billing:                 # a trust-gated group (plan item 2.5): any verified agent
 *           trust: {min: 0.6, topic: billing}   # whose TrustFlow on the topic is at least 0.6
 *         partners:                # members AND trust, both required
 *           members: [domain:partner.example]
 *           trust: {min: 0.4}
 *       default: public           # the group of a verified caller in no group; `none` refuses them
 *       instructions:
 *         friends: FRIENDS.md     # added to the instructions for callers in `friends`
 *       tools:
 *         friends: [rest]         # skills or tools only these groups (and the owner) may use
 *
 * THE DECISION, in this order, over principals that came from verified
 * credentials only (the access skill collects them; nothing here reads a
 * request):
 *   1. a principal matching `deny` refuses;
 *   2. the owner, or an admin, is let in and placed in no group (they pass
 *      every group scope anyway);
 *   3. the caller joins EVERY group one of its principals matches, and, for a
 *      group with a `trust` requirement, only when the VERIFIED calling agent
 *      (its `agent:` principal, from a Web Bot Auth signature) has the
 *      TrustFlow the group asks for;
 *   4. none matched: the `default` group, or refused when `default: none`.
 *
 * TRUST IS EVIDENCE, NOT A PROMISE. TrustFlow is a platform service: the
 * access skill asks the platform for the calling agent's score (`trustflow/
 * trust-lookup.ts`, authenticated as THIS agent) and hands what it learned to
 * `decide` as `TrustEvidence`: the agent it is about and a score per topic
 * key (`''` for the overall score). Nothing here dials anything, so the table
 * both SDKs run can stub the platform. FAIL CLOSED: no evidence (the platform
 * unreachable, no credential, no agent principal), evidence about another
 * agent, or a topic the platform could not score all mean the trust-gated
 * group is NOT joined; the caller still gets whatever else it matched, or the
 * default.
 *
 * PATTERNS. `user:<id>` and `user:@<handle>` (handles compared without case),
 * `key:<RFC 7638 thumbprint>`, `domain:<host>` (that host or any subdomain,
 * never a suffix), `agent:<URL>` where `*` is one host label or one path
 * segment and `**` any number of path segments, and `channel:<type>:<sender id>`
 * for a sender the platform relayed from a connected channel (plan item 2.2:
 * `channel:telegram:8842`, `channel:email:alice@example.com`), where the
 * sender id is compared without case and `*` is every sender of that type
 * (`channel:slack:*`). A channel principal comes only from the platform's
 * caller assertion on a relayed turn (`portal/connect.ts`), never from a
 * request. The Python twin is `python/webagents/access/policy.py`; both run
 * `python/tests/fixtures/access/policy.json` and refuse a malformed block with
 * the same sentence.
 */

export const KNOWN_KEYS = ['deny', 'groups', 'default', 'instructions', 'tools'] as const;
export const RESERVED_GROUPS: ReadonlySet<string> = new Set(['owner', 'admin', 'user', 'all', 'none']);
export const DEFAULT_GROUP = 'everyone';
const GROUP_NAME = /^[a-z][a-z0-9_-]{0,31}$/;
const THUMBPRINT = /^[A-Za-z0-9_-]{43}$/;
const HOST = /^(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)*$/;
const AGENT = /^(https?):\/\/([^/?#\s]+)(\/[^?#\s]*)?$/i;
const HOST_LABEL = /^(\*|[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)$/;
const NOT_AN_IDENTITY =
  'is not an identity. Use user:<id>, user:@<handle>, agent:<https URL>, key:<thumbprint>, domain:<host> or channel:<type>:<sender id>.';
/** A channel type is a slug, the same rule as a group name; the sender id has no whitespace. */
const CHANNEL_TYPE = /^[a-z][a-z0-9-]{0,31}$/;

/** A malformed `access:` block; the message names where and what to write. */
export class AccessConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AccessConfigError';
  }
}

export interface Pattern {
  kind: 'user' | 'key' | 'domain' | 'agent' | 'channel';
  /** For `channel`: `<type>:<sender id>` with the type lower-cased, the sender id as written or `*`. */
  value: string;
  regex?: RegExp;
}

function escapeRe(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\/-]/g, '\\$&');
}

function principalHost(url: string): string | null {
  const m = AGENT.exec(url);
  if (!m || m[2].startsWith('[')) return null;
  return m[2].toLowerCase().split(':')[0];
}

function lowerOrigin(url: string): string {
  const m = AGENT.exec(url);
  if (!m) return url;
  return `${m[1].toLowerCase()}://${m[2].toLowerCase()}${(m[3] ?? '').replace(/\/+$/, '')}`;
}

function agentRegex(url: string): RegExp | null {
  const m = AGENT.exec(url);
  if (!m) return null;
  const scheme = m[1].toLowerCase();
  const hostport = m[2].toLowerCase();
  const path = (m[3] ?? '').replace(/\/+$/, '');
  const colon = hostport.indexOf(':');
  const host = colon >= 0 ? hostport.slice(0, colon) : hostport;
  const port = colon >= 0 ? hostport.slice(colon + 1) : '';
  const labels = host.split('.');
  if (labels.length === 0 || labels.some((label) => !HOST_LABEL.test(label))) return null;
  if (port && !/^\d+$/.test(port)) return null;
  const hostRe = labels.map((label) => (label === '*' ? '[a-z0-9-]+' : escapeRe(label))).join('\\.');
  const portRe = port ? `:${port}` : '';
  let pathRe = '';
  for (const segment of path.split('/').slice(1)) {
    if (segment === '**') pathRe += '(?:/[^/]+)*';
    else if (segment === '*') pathRe += '/[^/]+';
    else if (!segment || segment.includes('*')) return null;
    else pathRe += `/${escapeRe(segment)}`;
  }
  return new RegExp(`^${scheme}://${hostRe}${portRe}${pathRe}$`);
}

export function patternMatches(pattern: Pattern, principal: string): boolean {
  const colon = principal.indexOf(':');
  const kind = colon >= 0 ? principal.slice(0, colon) : principal;
  const value = colon >= 0 ? principal.slice(colon + 1) : '';
  if (pattern.kind === 'user') {
    if (kind !== 'user') return false;
    if (pattern.value.startsWith('@')) return value.startsWith('@') && value.toLowerCase() === pattern.value.toLowerCase();
    return value === pattern.value;
  }
  if (pattern.kind === 'key') return kind === 'key' && value === pattern.value;
  if (pattern.kind === 'channel') {
    if (kind !== 'channel') return false;
    const sep = value.indexOf(':');
    if (sep < 0) return false;
    const patternSep = pattern.value.indexOf(':');
    const wantType = pattern.value.slice(0, patternSep);
    const wantSender = pattern.value.slice(patternSep + 1);
    if (value.slice(0, sep).toLowerCase() !== wantType) return false;
    const sender = value.slice(sep + 1);
    return sender.length > 0 && (wantSender === '*' || sender.toLowerCase() === wantSender.toLowerCase());
  }
  if (kind !== 'agent') return false;
  if (pattern.kind === 'domain') {
    const host = principalHost(value);
    return host !== null && (host === pattern.value || host.endsWith(`.${pattern.value}`));
  }
  return pattern.regex !== undefined && pattern.regex.test(lowerOrigin(value));
}

export function parsePattern(entry: unknown, where: string): Pattern {
  const bad = new AccessConfigError(`${where}: "${String(entry)}" ${NOT_AN_IDENTITY}`);
  if (typeof entry !== 'string') throw bad;
  const colon = entry.indexOf(':');
  if (colon < 0) throw bad;
  const kind = entry.slice(0, colon);
  const value = entry.slice(colon + 1);
  if (!value) throw bad;
  if (kind === 'user') {
    const name = value.startsWith('@') ? value.slice(1) : value;
    if (name && !/\s/.test(name)) return { kind: 'user', value };
    throw bad;
  }
  if (kind === 'key') {
    if (THUMBPRINT.test(value)) return { kind: 'key', value };
    throw bad;
  }
  if (kind === 'domain') {
    const lowered = value.toLowerCase();
    if (HOST.test(lowered)) return { kind: 'domain', value: lowered };
    throw bad;
  }
  if (kind === 'agent') {
    const regex = agentRegex(value);
    if (regex) return { kind: 'agent', value, regex };
  }
  if (kind === 'channel') {
    // `<type>:<sender id>`: the type named (never `*`), the sender id one
    // token without whitespace, or `*` for every sender of that type.
    const sep = value.indexOf(':');
    const type = sep >= 0 ? value.slice(0, sep).toLowerCase() : '';
    const sender = sep >= 0 ? value.slice(sep + 1) : '';
    if (CHANNEL_TYPE.test(type) && sender && sender.length <= 200 && !/\s/.test(sender)) {
      return { kind: 'channel', value: `${type}:${sender}` };
    }
  }
  throw bad;
}

function patterns(value: unknown, where: string, listMessage: string): Pattern[] {
  if (!Array.isArray(value)) throw new AccessConfigError(listMessage);
  return value.map((entry) => parsePattern(entry, where));
}

/** What a trust-gated group asks of the calling agent's TrustFlow. */
export interface TrustRequirement {
  /** The least TrustFlow score, 0 to 1. */
  min: number;
  /** The topic the score is on; null is the overall score. */
  topic: string | null;
}

/** One group: who may join by identity, and what trust they need. Either may be absent, never both. */
export interface GroupRule {
  /** Identity patterns; null when the group is open to any verified agent with the trust. */
  members: Pattern[] | null;
  trust: TrustRequirement | null;
}

/** What the platform said about the calling agent (file comment, "TRUST IS EVIDENCE"). */
export interface TrustEvidence {
  /** The agent URL the scores are about: must be the caller's `agent:` principal. */
  agent: string;
  /** Score per topic key (`trustKey`); a topic missing here was not scored. */
  scores: Record<string, number>;
}

/** The key a requirement's score is filed under in `TrustEvidence.scores`. */
export function trustKey(topic: string | null | undefined): string {
  return topic ?? '';
}

const GROUP_RULE_KEYS = ['members', 'trust'] as const;

function parseTrust(raw: unknown, where: string): TrustRequirement {
  const message = `${where}.trust must be a mapping of min (a number from 0 to 1) and an optional topic.`;
  if (!isMapping(raw)) throw new AccessConfigError(message);
  for (const key of Object.keys(raw)) if (key !== 'min' && key !== 'topic') throw new AccessConfigError(message);
  const min = raw.min;
  if (typeof min !== 'number' || !Number.isFinite(min) || min < 0 || min > 1) throw new AccessConfigError(message);
  let topic: string | null = null;
  if ('topic' in raw) {
    if (typeof raw.topic !== 'string' || !raw.topic.trim()) throw new AccessConfigError(message);
    topic = raw.topic.trim();
  }
  return { min, topic };
}

/** A group as written: a list of identities, or `{members, trust}`. */
function parseGroup(raw: unknown, where: string): GroupRule {
  if (Array.isArray(raw)) return { members: patterns(raw, where, `${where} must be a list of identities.`), trust: null };
  if (!isMapping(raw)) throw new AccessConfigError(`${where} must be a list of identities, or a mapping of members and trust.`);
  for (const key of Object.keys(raw)) {
    if (!(GROUP_RULE_KEYS as readonly string[]).includes(key)) {
      throw new AccessConfigError(`${where}: unknown key "${key}". It takes members and trust.`);
    }
  }
  const members = 'members' in raw ? patterns(raw.members, where, `${where}.members must be a list of identities.`) : null;
  const trust = 'trust' in raw ? parseTrust(raw.trust, where) : null;
  if (members === null && trust === null) throw new AccessConfigError(`${where} must name members or trust.`);
  return { members, trust };
}

export interface AccessPolicy {
  deny: Pattern[];
  groups: Map<string, GroupRule>;
  /** The group of a caller in no group; null is `default: none`, refuse them. */
  default: string | null;
  /** Group name to the Markdown file named for it, relative to the agent file. */
  instructions: Map<string, string>;
  /** Group name to the skill or tool names only its members (and the owner) may use. */
  tools: Map<string, string[]>;
}

export function knownGroups(policy: AccessPolicy): string[] {
  const names = [...policy.groups.keys()];
  if (policy.default && !names.includes(policy.default)) names.push(policy.default);
  return names;
}

function isMapping(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

/** The block as written in the agent file, or an `AccessConfigError`. */
export function parseAccess(raw: unknown): AccessPolicy {
  if (!isMapping(raw)) {
    throw new AccessConfigError('access must be a mapping of deny, groups, default, instructions and tools.');
  }
  for (const key of Object.keys(raw)) {
    if (!(KNOWN_KEYS as readonly string[]).includes(key)) {
      throw new AccessConfigError(`access: unknown key "${key}". It takes deny, groups, default, instructions and tools.`);
    }
  }
  const policy: AccessPolicy = { deny: [], groups: new Map(), default: DEFAULT_GROUP, instructions: new Map(), tools: new Map() };
  if ('deny' in raw) {
    policy.deny = patterns(raw.deny, 'access.deny', 'access.deny must be a list of identities, for example agent:https://spam.example/**.');
  }
  if ('groups' in raw) {
    if (!isMapping(raw.groups)) throw new AccessConfigError('access.groups must map a group name to a list of identities.');
    for (const [name, members] of Object.entries(raw.groups)) {
      if (!GROUP_NAME.test(name)) {
        throw new AccessConfigError(
          `access.groups: "${name}" is not a group name (lower-case letters, digits, - and _, starting with a letter).`,
        );
      }
      if (RESERVED_GROUPS.has(name)) throw new AccessConfigError(`access.groups: "${name}" is reserved.`);
      policy.groups.set(name, parseGroup(members, `access.groups.${name}`));
    }
  }
  if ('default' in raw) {
    const value = raw.default;
    if (value === 'none') policy.default = null;
    else if (typeof value === 'string' && GROUP_NAME.test(value) && !RESERVED_GROUPS.has(value)) policy.default = value;
    else throw new AccessConfigError('access.default must be a group name or none.');
  }
  const known = new Set(knownGroups(policy));
  if ('instructions' in raw) {
    const message = 'access.instructions must map a group name to a Markdown file next to the agent file.';
    if (!isMapping(raw.instructions)) throw new AccessConfigError(message);
    for (const [name, file] of Object.entries(raw.instructions)) {
      if (!known.has(name)) throw new AccessConfigError(`access.instructions: "${name}" is not a group this block defines.`);
      if (typeof file !== 'string' || !file.trim()) throw new AccessConfigError(message);
      policy.instructions.set(name, file.trim());
    }
  }
  if ('tools' in raw) {
    const message = 'access.tools must map a group name to a list of skill or tool names.';
    if (!isMapping(raw.tools)) throw new AccessConfigError(message);
    for (const [name, names] of Object.entries(raw.tools)) {
      if (!known.has(name)) throw new AccessConfigError(`access.tools: "${name}" is not a group this block defines.`);
      if (!Array.isArray(names) || !names.every((n) => typeof n === 'string' && n)) throw new AccessConfigError(message);
      policy.tools.set(name, names as string[]);
    }
  }
  return policy;
}

export interface Decision {
  allow: boolean;
  groups: string[];
}

/** Every distinct trust requirement the block makes, one per topic key: what the access skill must look up. */
export function trustRequirements(policy: AccessPolicy): TrustRequirement[] {
  const byKey = new Map<string, TrustRequirement>();
  for (const rule of policy.groups.values()) {
    if (rule.trust && !byKey.has(trustKey(rule.trust.topic))) byKey.set(trustKey(rule.trust.topic), rule.trust);
  }
  return [...byKey.values()];
}

/** The verified calling agent: the URL of the first `agent:` principal, or null. */
export function verifiedAgentOf(principals: readonly string[]): string | null {
  const principal = principals.find((p) => p.startsWith('agent:'));
  return principal ? principal.slice('agent:'.length) : null;
}

/** Whether `evidence` shows the verified calling agent meets `requirement` (file comment: fail closed). */
export function meetsTrust(requirement: TrustRequirement, principals: readonly string[], evidence: TrustEvidence | null | undefined): boolean {
  const agent = verifiedAgentOf(principals);
  if (!agent || !evidence || evidence.agent !== agent) return false;
  const score = evidence.scores[trustKey(requirement.topic)];
  return typeof score === 'number' && Number.isFinite(score) && score >= requirement.min;
}

/**
 * Deny, then the owner, then every matching group, then the default. A
 * group with `members` needs one principal to match; a group with `trust`
 * needs `evidence` to show it (both, when it has both).
 */
export function decide(
  policy: AccessPolicy,
  principals: readonly string[],
  tier?: string | null,
  trust?: TrustEvidence | null,
): Decision {
  if (policy.deny.some((pattern) => principals.some((p) => patternMatches(pattern, p)))) return { allow: false, groups: [] };
  if (tier === 'owner' || tier === 'admin') return { allow: true, groups: [] };
  const groups = [...policy.groups.entries()]
    .filter(([, rule]) => {
      const byIdentity = rule.members === null || rule.members.some((pattern) => principals.some((p) => patternMatches(pattern, p)));
      const byTrust = rule.trust === null || meetsTrust(rule.trust, principals, trust);
      return byIdentity && byTrust;
    })
    .map(([name]) => name);
  if (groups.length) return { allow: true, groups };
  if (policy.default === null) return { allow: false, groups: [] };
  return { allow: true, groups: [policy.default] };
}
