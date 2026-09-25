/**
 * Which addresses the REST tool and the key-directory fetch may connect to
 * (ADR-0045), from the table both SDKs run
 * (`python/tests/fixtures/net/addresses.json`; Python tests/net/test_addresses.py).
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  AllowListError,
  addressAllowed,
  ipText,
  isAlwaysBlocked,
  isPublicAddress,
  parseAllowList,
  parseIp,
  sortAddresses,
} from '../../../src/net/addresses';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const TABLE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/net/addresses.json'), 'utf8'),
) as {
  addresses: Array<{ address: string; public: boolean; always_blocked: boolean }>;
  allow_lists: Array<{ entries: string[]; address: string; port: number; allowed: boolean }>;
  bad_allow_entries: string[];
};

describe('classification', () => {
  for (const c of TABLE.addresses) {
    it(c.address, () => {
      const address = parseIp(c.address)!;
      expect(address).not.toBeNull();
      expect(isPublicAddress(address)).toBe(c.public);
      expect(isAlwaysBlocked(address)).toBe(c.always_blocked);
    });
  }
});

describe('allow lists', () => {
  for (const c of TABLE.allow_lists) {
    it(`${JSON.stringify(c.entries)} -> ${c.address}:${c.port}`, () => {
      expect(addressAllowed(parseIp(c.address)!, c.port, parseAllowList(c.entries))).toBe(c.allowed);
    });
  }
  for (const entry of TABLE.bad_allow_entries) {
    it(`refuses ${JSON.stringify(entry)}`, () => {
      expect(() => parseAllowList([entry])).toThrow(AllowListError);
    });
  }
});

describe('text and order', () => {
  it('writes an address the way the URL parser does', () => {
    expect(ipText(parseIp('::ffff:127.0.0.1')!)).toBe('::ffff:7f00:1');
    expect(ipText(parseIp('2001:db8:0:1:1:1:1:1')!)).toBe('2001:db8:0:1:1:1:1:1');
    expect(ipText(parseIp('fe80::1')!)).toBe('fe80::1');
    expect(ipText(parseIp('::')!)).toBe('::');
    expect(new URL('http://[::ffff:127.0.0.1]/').hostname).toBe('[::ffff:7f00:1]');
  });

  it('IPv4 first, then by value', () => {
    const ordered = sortAddresses(['::1', '127.0.0.2', '127.0.0.1', '::1'].map((a) => parseIp(a)!));
    expect(ordered.map(ipText)).toEqual(['127.0.0.1', '127.0.0.2', '::1']);
  });
});
