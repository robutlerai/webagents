/**
 * One HTTP exchange that connects only to an address `addresses.ts` allowed.
 *
 * CHECK, THEN CONNECT TO WHAT WAS CHECKED (ADR-0045, 2026-09-25). Resolving a
 * name, checking the answer and handing the NAME to `fetch` is not a guard:
 * fetch resolves again, and a name can answer differently the second time (DNS
 * rebinding). Node's built-in fetch offers no DNS hook, so this uses
 * `http(s).request` with a `lookup` that answers with the one address checked
 * here, after every address the name resolves to passed (one private answer
 * refuses the request). TLS still verifies the certificate against the
 * HOSTNAME: `host` and `servername` stay the name; only the socket's address is
 * pinned.
 *
 * No redirects followed behind the caller's back (each hop must be checked and,
 * for the REST tool, signed again), no proxy from the environment, no cookie
 * jar, and no content decoding (the caller asks for `Accept-Encoding: identity`).
 *
 * The Python twin is `python/webagents/net/guarded_http.py`; the two answer with
 * the same codes and messages.
 */

import * as http from 'node:http';
import * as https from 'node:https';
import { lookup as dnsLookup } from 'node:dns/promises';
import type { LookupFunction } from 'node:net';

import { addressAllowed, ipText, parseIp, sortAddresses, type AllowEntry, type IpAddress } from './addresses';

/** A refusal or failure with a stable `code` and a sentence for a person. */
export class GuardError extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = 'GuardError';
  }
}

export function refusedAddressMessage(host: string, address: IpAddress, literal: boolean): string {
  if (literal) return `${ipText(address)} is not a public address, so it is not called.`;
  return `${host} resolves to ${ipText(address)}, which is not a public address, so it is not called.`;
}

/** The URL's hostname without the brackets an IPv6 literal carries in a URL. */
export function bareHost(hostname: string): string {
  return hostname.startsWith('[') && hostname.endsWith(']') ? hostname.slice(1, -1) : hostname;
}

/**
 * The address to connect to for `host:port`, or a `GuardError`. Every address
 * the name resolves to must be allowed; the first in `sortAddresses` order is
 * the one to connect to.
 */
export async function resolveAllowed(host: string, port: number, allow: readonly AllowEntry[] = []): Promise<IpAddress> {
  const literal = parseIp(host);
  if (literal) {
    if (!addressAllowed(literal, port, allow)) {
      throw new GuardError('blocked_address', refusedAddressMessage(host, literal, true));
    }
    return literal;
  }
  let answers: Array<{ address: string }>;
  try {
    answers = await dnsLookup(host, { all: true, verbatim: true });
  } catch {
    throw new GuardError('network_error', `${host} did not resolve.`);
  }
  const addresses = sortAddresses(
    answers.map((a) => parseIp(a.address.split('%')[0])).filter((a): a is IpAddress => a !== null),
  );
  if (addresses.length === 0) throw new GuardError('network_error', `${host} did not resolve.`);
  for (const address of addresses) {
    if (!addressAllowed(address, port, allow)) {
      throw new GuardError('blocked_address', refusedAddressMessage(host, address, false));
    }
  }
  return addresses[0];
}

/** What came back. `body` holds at most `maxBytes`; `truncated` says more existed. */
export interface Exchange {
  status: number;
  /** Lower-cased names, in the order received. */
  headers: Array<[string, string]>;
  body: Buffer;
  truncated: boolean;
}

export function headerOf(exchange: Exchange, name: string): string | undefined {
  const lowered = name.toLowerCase();
  return exchange.headers.find(([key]) => key === lowered)?.[1];
}

function networkMessage(host: string, error: NodeJS.ErrnoException): string {
  const code = String(error.code ?? '');
  const text = `${code} ${error.message ?? ''}`.toLowerCase();
  if (code.startsWith('ERR_TLS') || code.includes('CERT') || text.includes('certificate') || text.includes('ssl')) {
    return `Could not connect to ${host}: its TLS certificate was not accepted.`;
  }
  if (code === 'ECONNREFUSED' || text.includes('refused')) return `Could not connect to ${host}: the connection was refused.`;
  if (code === 'ECONNRESET' || text.includes('reset')) return `Could not connect to ${host}: the connection was reset.`;
  return `Could not connect to ${host}: the connection failed.`;
}

export interface ExchangeOptions {
  method: string;
  scheme: 'http' | 'https';
  /** The hostname, IPv6 without brackets. */
  host: string;
  port: number;
  /** The request line's path and query, sent exactly as given. */
  target: string;
  /** Sent in this order, `Host` included. */
  headers: Array<[string, string]>;
  body: Buffer;
  /** Already checked by `resolveAllowed`. */
  address: IpAddress;
  /** `Date.now()` value by which the whole exchange must be done. */
  deadline: number;
  maxBytes: number;
}

/**
 * Send one request to `address` and read at most `maxBytes` of the answer
 * before `deadline`.
 */
export function exchange(options: ExchangeOptions): Promise<Exchange> {
  const { method, scheme, host, port, target, headers, body, address, deadline, maxBytes } = options;
  const pinned = ipText(address);
  const family = address.version;
  const literal = parseIp(host) !== null;
  const lookup: LookupFunction = (_hostname, lookupOptions, callback) => {
    if ((lookupOptions as { all?: boolean }).all) {
      (callback as unknown as (err: null, list: Array<{ address: string; family: number }>) => void)(null, [
        { address: pinned, family },
      ]);
    } else {
      (callback as unknown as (err: null, address: string, family: number) => void)(null, pinned, family);
    }
  };
  const raw: string[] = [];
  for (const [name, value] of headers) raw.push(name, value);

  return new Promise<Exchange>((resolve, reject) => {
    const left = deadline - Date.now();
    if (left <= 0) {
      reject(new GuardError('timeout', ''));
      return;
    }
    let settled = false;
    const finish = (fn: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn();
    };
    const requestOptions: https.RequestOptions = {
      method,
      host: literal ? pinned : host,
      port,
      path: target,
      headers: raw as unknown as http.OutgoingHttpHeaders,
      setHost: false,
      agent: false,
      lookup,
      ...(scheme === 'https' && !literal ? { servername: host } : {}),
    };
    const request = (scheme === 'https' ? https : http).request(requestOptions, (response) => {
      const chunks: Buffer[] = [];
      let size = 0;
      let truncated = false;
      const done = (): void =>
        finish(() => {
          const pairs: Array<[string, string]> = [];
          for (let i = 0; i + 1 < response.rawHeaders.length; i += 2) {
            pairs.push([response.rawHeaders[i].toLowerCase(), response.rawHeaders[i + 1]]);
          }
          resolve({ status: response.statusCode ?? 0, headers: pairs, body: Buffer.concat(chunks), truncated });
        });
      response.on('data', (chunk: Buffer) => {
        if (settled || truncated) return;
        const room = maxBytes - size;
        if (chunk.length > room) {
          chunks.push(chunk.subarray(0, room));
          size = maxBytes;
          truncated = true;
          done();
          response.destroy();
          return;
        }
        chunks.push(chunk);
        size += chunk.length;
      });
      response.on('end', done);
      response.on('error', (error: NodeJS.ErrnoException) =>
        finish(() => reject(new GuardError('network_error', networkMessage(host, error)))),
      );
    });
    const timer = setTimeout(() => {
      finish(() => reject(new GuardError('timeout', '')));
      request.destroy();
    }, left);
    request.on('error', (error: NodeJS.ErrnoException) =>
      finish(() => reject(new GuardError('network_error', networkMessage(host, error)))),
    );
    if (body.length > 0) request.write(body);
    request.end();
  });
}
