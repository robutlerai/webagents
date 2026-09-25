/**
 * Which network addresses an agent may connect to on a caller's or a model's
 * behalf.
 *
 * WHY THIS EXISTS (ADR-0045, 2026-09-25). The REST tool calls URLs a model
 * chose, and the Web Bot Auth verifier fetches key directories a caller named.
 * Either one, left alone, makes the agent's host fetch its own loopback
 * services, the private network it sits in, or the cloud metadata endpoint that
 * hands out the machine's credentials. So both ask this module first, about
 * every address a name resolves to, and connect only to an address it passed
 * (`guarded-request.ts` pins the connection to it).
 *
 * THE SAME TABLE AS PYTHON (`python/webagents/net/addresses.py`), and both SDKs
 * run the same cases (`python/tests/fixtures/net/addresses.json`): the IANA
 * special-purpose registries, plus the embedded-IPv4 forms judged by the
 * address they carry (`::ffff:a.b.c.d` and 6to4 `2002::/16`).
 *
 * ALLOW LISTS. An agent file may name private addresses the REST tool may call
 * anyway (`allow_private`: IPs or CIDRs, each optionally with a port).
 * Link-local and cloud metadata addresses stay refused whatever the list says.
 */

import { isIPv4, isIPv6 } from 'node:net';

/** An IP address as bytes: 4 for IPv4, 16 for IPv6. */
export interface IpAddress {
  readonly version: 4 | 6;
  readonly bytes: Uint8Array;
}

interface Network {
  readonly version: 4 | 6;
  readonly bytes: Uint8Array;
  readonly prefix: number;
}

function v4(text: string): Uint8Array | null {
  if (!isIPv4(text)) return null;
  return Uint8Array.from(text.split('.').map(Number));
}

function v6(raw: string): Uint8Array | null {
  let text = raw.startsWith('[') && raw.endsWith(']') ? raw.slice(1, -1) : raw;
  const zone = text.indexOf('%');
  if (zone >= 0) text = text.slice(0, zone);
  if (!isIPv6(text)) return null;
  const groups = (part: string): number[] | null => {
    if (part === '') return [];
    const out: number[] = [];
    const pieces = part.split(':');
    for (let i = 0; i < pieces.length; i += 1) {
      const piece = pieces[i];
      if (piece.includes('.')) {
        if (i !== pieces.length - 1) return null;
        const quad = v4(piece);
        if (!quad) return null;
        out.push((quad[0] << 8) | quad[1], (quad[2] << 8) | quad[3]);
      } else {
        out.push(parseInt(piece, 16));
      }
    }
    return out;
  };
  const halves = text.split('::');
  const head = groups(halves[0]);
  const tail = halves.length > 1 ? groups(halves[1]) : [];
  if (!head || !tail) return null;
  const fill = 8 - head.length - tail.length;
  const all = halves.length > 1 ? [...head, ...new Array<number>(fill).fill(0), ...tail] : head;
  if (all.length !== 8) return null;
  const bytes = new Uint8Array(16);
  all.forEach((group, i) => {
    bytes[i * 2] = group >> 8;
    bytes[i * 2 + 1] = group & 0xff;
  });
  return bytes;
}

/** An IP literal (IPv6 with or without brackets), or null for anything else. */
export function parseIp(text: string): IpAddress | null {
  const four = v4(text);
  if (four) return { version: 4, bytes: four };
  const six = v6(text);
  if (six) return { version: 6, bytes: six };
  return null;
}

/**
 * The address as text: dotted IPv4, and IPv6 in hex groups with the first
 * longest run of two or more zero groups written `::`. That is the WHATWG URL
 * serialisation (`new URL('http://[::ffff:127.0.0.1]/').hostname` is
 * `[::ffff:7f00:1]`), and Python builds the same string from the same bytes.
 */
export function ipText(address: IpAddress): string {
  const b = address.bytes;
  if (address.version === 4) return Array.from(b).join('.');
  const groups: number[] = [];
  for (let i = 0; i < 16; i += 2) groups.push((b[i] << 8) | b[i + 1]);
  let bestStart = -1;
  let bestLength = 1;
  for (let i = 0; i < 8; ) {
    if (groups[i] !== 0) {
      i += 1;
      continue;
    }
    let j = i;
    while (j < 8 && groups[j] === 0) j += 1;
    if (j - i > bestLength) {
      bestStart = i;
      bestLength = j - i;
    }
    i = j;
  }
  const hex = groups.map((g) => g.toString(16));
  if (bestStart < 0) return hex.join(':');
  const left = hex.slice(0, bestStart).join(':');
  const right = hex.slice(bestStart + bestLength).join(':');
  return `${left}::${right}`;
}

function network(cidr: string): Network {
  const [base, bits] = cidr.split('/');
  const parsed = parseIp(base);
  if (!parsed) throw new Error(`bad network ${cidr}`);
  return { version: parsed.version, bytes: parsed.bytes, prefix: Number(bits) };
}

function inNetwork(address: IpAddress, net: Network): boolean {
  if (address.version !== net.version) return false;
  const whole = Math.floor(net.prefix / 8);
  for (let i = 0; i < whole; i += 1) if (address.bytes[i] !== net.bytes[i]) return false;
  const rest = net.prefix % 8;
  if (rest === 0) return true;
  const mask = (0xff << (8 - rest)) & 0xff;
  return (address.bytes[whole] & mask) === (net.bytes[whole] & mask);
}

const NOT_PUBLIC_V4 = [
  '0.0.0.0/8',
  '10.0.0.0/8',
  '100.64.0.0/10',
  '127.0.0.0/8',
  '169.254.0.0/16',
  '172.16.0.0/12',
  '192.0.0.0/24',
  '192.0.2.0/24',
  '192.88.99.0/24',
  '192.168.0.0/16',
  '198.18.0.0/15',
  '198.51.100.0/24',
  '203.0.113.0/24',
  '224.0.0.0/4',
  '240.0.0.0/4',
].map(network);

const NOT_PUBLIC_V6 = [
  '::/96',
  '64:ff9b::/32',
  '100::/64',
  '2001::/23',
  '2001:db8::/32',
  '3fff::/20',
  '5f00::/16',
  'fc00::/7',
  'fe80::/10',
  'fec0::/10',
  'ff00::/8',
].map(network);

const ALWAYS_BLOCKED = ['169.254.0.0/16', '100.100.100.200/32', 'fe80::/10', 'fd00:ec2::254/128'].map(network);

const MAPPED_V4 = network('::ffff:0:0/96');
const SIX_TO_FOUR = network('2002::/16');

/** The address a packet to `address` is routed as: the IPv4 inside a v4-mapped or 6to4 address. */
function effective(address: IpAddress): IpAddress {
  if (address.version === 6) {
    if (inNetwork(address, MAPPED_V4)) return { version: 4, bytes: address.bytes.slice(12, 16) };
    if (inNetwork(address, SIX_TO_FOUR)) return { version: 4, bytes: address.bytes.slice(2, 6) };
  }
  return address;
}

/** Whether `address` is an ordinary public internet address. */
export function isPublicAddress(address: IpAddress): boolean {
  const routed = effective(address);
  const table = routed.version === 4 ? NOT_PUBLIC_V4 : NOT_PUBLIC_V6;
  return !table.some((net) => inNetwork(routed, net));
}

/** Link-local and cloud metadata: refused even when an allow list covers them. */
export function isAlwaysBlocked(address: IpAddress): boolean {
  const routed = effective(address);
  return ALWAYS_BLOCKED.some((net) => inNetwork(routed, net));
}

/** An `allow_private` entry that is not an IP, a CIDR, or either with a port. */
export class AllowListError extends Error {}

export interface AllowEntry {
  readonly network: Network;
  readonly port?: number;
}

const BRACKETED = /^\[([^\]]+)\]:(\d+)$/;
const V4_PORT = /^([0-9.]+):(\d+)$/;

/** `10.0.0.0/8`, `127.0.0.1`, `127.0.0.1:8080`, `::1` or `[::1]:8080`. */
export function parseAllowEntry(entry: unknown): AllowEntry {
  if (typeof entry !== 'string' || !entry.trim()) {
    throw new AllowListError(
      'allow_private entries are IP addresses or CIDR ranges, for example 127.0.0.1:8080 or 10.0.0.0/8',
    );
  }
  let text = entry.trim();
  let port: number | undefined;
  const match = BRACKETED.exec(text) ?? V4_PORT.exec(text);
  if (match) {
    text = match[1];
    port = Number(match[2]);
    if (!(port >= 1 && port <= 65535)) {
      throw new AllowListError(`allow_private entry ${JSON.stringify(entry)} has a port outside 1-65535`);
    }
  }
  const bad = new AllowListError(
    `allow_private entry ${JSON.stringify(entry)} is not an IP address or CIDR range ` +
      '(for example 127.0.0.1:8080, 10.0.0.0/8 or [::1]:8080)',
  );
  const slash = text.indexOf('/');
  const base = slash >= 0 ? text.slice(0, slash) : text;
  const bitsText = slash >= 0 ? text.slice(slash + 1) : undefined;
  const parsed = base.startsWith('[') ? null : parseIp(base);
  if (!parsed) throw bad;
  const max = parsed.version === 4 ? 32 : 128;
  const prefix = bitsText === undefined ? max : /^\d+$/.test(bitsText) ? Number(bitsText) : NaN;
  if (!(prefix >= 0 && prefix <= max)) throw bad;
  const net: Network = { version: parsed.version, bytes: parsed.bytes, prefix };
  // A CIDR with host bits set (`10.0.0.1/8`) is refused, as Python's strict parse does.
  const hostBits = parsed.bytes.some((byte, i) => {
    const keep = Math.max(0, Math.min(8, prefix - i * 8));
    const mask = keep === 0 ? 0 : (0xff << (8 - keep)) & 0xff;
    return (byte & ~mask & 0xff) !== 0;
  });
  if (hostBits) throw bad;
  return port === undefined ? { network: net } : { network: net, port };
}

export function parseAllowList(entries: unknown): AllowEntry[] {
  if (entries === undefined || entries === null) return [];
  if (!Array.isArray(entries)) throw new AllowListError('allow_private is a list of IP addresses or CIDR ranges');
  return entries.map(parseAllowEntry);
}

/** Whether a connection to `address:port` may be made. */
export function addressAllowed(address: IpAddress, port: number, allow: readonly AllowEntry[] = []): boolean {
  if (isAlwaysBlocked(address)) return false;
  if (isPublicAddress(address)) return true;
  const routed = effective(address);
  return allow.some((entry) => inNetwork(routed, entry.network) && (entry.port === undefined || entry.port === port));
}

/**
 * IPv4 first, then by value, without duplicates: both SDKs check and pin the
 * same address for a name, whatever order the resolver answered in.
 */
export function sortAddresses(addresses: readonly IpAddress[]): IpAddress[] {
  const key = (a: IpAddress): string => `${a.version}:${Array.from(a.bytes, (b) => b.toString(16).padStart(2, '0')).join('')}`;
  const unique = new Map<string, IpAddress>();
  for (const a of addresses) unique.set(key(a), a);
  return [...unique.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([, a]) => a);
}
