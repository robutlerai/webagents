/**
 * RFC 9651 Structured Field Values, the PARSING subset the Web Bot Auth
 * verifier reads (ADR-0045, 2026-09-25): `Signature-Input`, `Signature`,
 * `Signature-Agent` and `Content-Digest`, into the value types the signer
 * already serialises (`SfItem`, `SfInnerList`, `SfToken` in
 * `http-signature.ts`), so a parsed `Signature-Input` member re-serialises
 * with the SAME code that wrote it.
 *
 * WHY HAND-WRITTEN HERE when the portal takes `structured-headers`: the SDK
 * carries no parser, a new dependency is a lockfile change the local cluster's
 * install policy has crashlooped on before, and the Python SDK needs the same
 * rules in a second language anyway. Both parsers are held to the same cases
 * (`python/tests/fixtures/web_bot_auth/structured-fields.json`).
 *
 * STRICTER THAN THE RFC WHERE LAXITY COULD ONLY HELP A FORGER, and every such
 * point fails closed:
 *   - a field longer than 8 KiB is refused unparsed, a Dictionary with more
 *     than 16 members after;
 *   - a repeated Dictionary key or Parameter key is refused (the RFC keeps the
 *     last, which lets two readers see two different fields);
 *   - Decimals, Dates and Display Strings are refused: nothing in the profile
 *     uses them, and a base must re-serialise exactly what was signed;
 *   - a Byte Sequence must be canonical padded base64.
 */

import { SfToken, type SfBareItem, type SfInnerList, type SfItem, type SfParameters } from './http-signature';

export const MAX_FIELD_CHARS = 8 * 1024;
export const MAX_DICTIONARY_MEMBERS = 16;

/** The one error; `reason` is a short phrase for a log and never echoes the field. */
export class StructuredFieldParseError extends Error {
  constructor(readonly reason: string) {
    super(`structured field: ${reason}`);
    this.name = 'StructuredFieldParseError';
  }
}

export type SfMember = SfItem | SfInnerList;

export function isInnerList(member: SfMember): member is SfInnerList {
  return 'items' in member;
}

const LCALPHA = /[a-z]/;
const DIGIT = /[0-9]/;
const ALPHA = /[A-Za-z]/;
const KEY_CHAR = /[a-z0-9_\-.*]/;
const TOKEN_CHAR = /[A-Za-z0-9:/!#$%&'*+\-.^_`|~]/;
const BASE64 = /^[A-Za-z0-9+/]*={0,2}$/;

class Parser {
  private i = 0;

  constructor(private readonly s: string) {
    if (typeof s !== 'string') throw new StructuredFieldParseError('not a string');
    if (s.length > MAX_FIELD_CHARS) throw new StructuredFieldParseError('field too long');
  }

  private peek(): string {
    return this.s[this.i] ?? '';
  }

  private fail(reason: string): never {
    throw new StructuredFieldParseError(reason);
  }

  private skipSP(): void {
    while (this.peek() === ' ') this.i += 1;
  }

  private skipOWS(): void {
    while (this.peek() === ' ' || this.peek() === '\t') this.i += 1;
  }

  /** RFC 9651 4.2: leading SP is discarded, and nothing may follow the value but SP. */
  finish<T>(value: T): T {
    this.skipSP();
    if (this.i !== this.s.length) this.fail('trailing characters');
    return value;
  }

  start(): void {
    this.skipSP();
  }

  dictionary(): Array<[string, SfMember]> {
    const out: Array<[string, SfMember]> = [];
    const seen = new Set<string>();
    if (this.i >= this.s.length) return out;
    for (;;) {
      const key = this.key();
      if (seen.has(key)) this.fail('repeated dictionary key');
      seen.add(key);
      let member: SfMember;
      if (this.peek() === '=') {
        this.i += 1;
        member = this.itemOrInnerList();
      } else {
        member = { value: true, params: this.parameters() };
      }
      out.push([key, member]);
      if (out.length > MAX_DICTIONARY_MEMBERS) this.fail('too many dictionary members');
      this.skipOWS();
      if (this.i >= this.s.length) return out;
      if (this.peek() !== ',') this.fail('expected a comma');
      this.i += 1;
      this.skipOWS();
      if (this.i >= this.s.length) this.fail('trailing comma');
    }
  }

  list(): SfMember[] {
    const out: SfMember[] = [];
    if (this.i >= this.s.length) return out;
    for (;;) {
      out.push(this.itemOrInnerList());
      this.skipOWS();
      if (this.i >= this.s.length) return out;
      if (this.peek() !== ',') this.fail('expected a comma');
      this.i += 1;
      this.skipOWS();
      if (this.i >= this.s.length) this.fail('trailing comma');
    }
  }

  itemOrInnerList(): SfMember {
    return this.peek() === '(' ? this.innerList() : this.item();
  }

  innerList(): SfInnerList {
    this.i += 1;
    const items: SfItem[] = [];
    for (;;) {
      this.skipSP();
      if (this.peek() === ')') {
        this.i += 1;
        return { items, params: this.parameters() };
      }
      if (this.i >= this.s.length) this.fail('unterminated inner list');
      items.push(this.item());
      const next = this.peek();
      if (next !== ' ' && next !== ')') this.fail('bad inner list separator');
    }
  }

  item(): SfItem {
    const value = this.bareItem();
    return { value, params: this.parameters() };
  }

  parameters(): SfParameters {
    const out: Array<[string, SfBareItem]> = [];
    const seen = new Set<string>();
    while (this.peek() === ';') {
      this.i += 1;
      this.skipSP();
      const key = this.key();
      if (seen.has(key)) this.fail('repeated parameter key');
      seen.add(key);
      let value: SfBareItem = true;
      if (this.peek() === '=') {
        this.i += 1;
        value = this.bareItem();
      }
      out.push([key, value]);
    }
    return out;
  }

  key(): string {
    const first = this.peek();
    if (!(LCALPHA.test(first) || first === '*')) this.fail('bad key');
    let out = '';
    while (this.i < this.s.length && KEY_CHAR.test(this.peek())) {
      out += this.peek();
      this.i += 1;
    }
    return out;
  }

  bareItem(): SfBareItem {
    const c = this.peek();
    if (c === '-' || DIGIT.test(c)) return this.integer();
    if (c === '"') return this.string();
    if (c === ':') return this.bytes();
    if (c === '?') return this.boolean();
    if (ALPHA.test(c) || c === '*') return this.token();
    if (c === '@') this.fail('dates are not accepted');
    if (c === '%') this.fail('display strings are not accepted');
    return this.fail('bad item');
  }

  integer(): number {
    let sign = 1;
    if (this.peek() === '-') {
      sign = -1;
      this.i += 1;
    }
    if (!DIGIT.test(this.peek())) this.fail('bad number');
    let digits = '';
    while (DIGIT.test(this.peek())) {
      digits += this.peek();
      this.i += 1;
      if (digits.length > 15) this.fail('integer too long');
    }
    if (this.peek() === '.') this.fail('decimals are not accepted');
    return sign * Number(digits);
  }

  string(): string {
    this.i += 1;
    let out = '';
    for (;;) {
      if (this.i >= this.s.length) this.fail('unterminated string');
      const c = this.s[this.i];
      this.i += 1;
      if (c === '\\') {
        const next = this.s[this.i];
        if (next !== '"' && next !== '\\') this.fail('bad escape');
        out += next;
        this.i += 1;
        continue;
      }
      if (c === '"') return out;
      const code = c.charCodeAt(0);
      if (code < 0x20 || code > 0x7e) this.fail('string is not printable ASCII');
      out += c;
    }
  }

  token(): SfToken {
    let out = '';
    while (this.i < this.s.length && TOKEN_CHAR.test(this.peek())) {
      out += this.peek();
      this.i += 1;
    }
    return new SfToken(out);
  }

  bytes(): Uint8Array {
    this.i += 1;
    const end = this.s.indexOf(':', this.i);
    if (end < 0) this.fail('unterminated byte sequence');
    const text = this.s.slice(this.i, end);
    this.i = end + 1;
    if (!BASE64.test(text) || text.length % 4 !== 0) this.fail('byte sequence is not padded base64');
    let binary: string;
    try {
      binary = atob(text);
    } catch {
      return this.fail('byte sequence is not base64');
    }
    const bytes = Uint8Array.from(binary, (ch) => ch.charCodeAt(0));
    if (btoa(binary) !== text) this.fail('byte sequence is not canonical base64');
    return bytes;
  }

  boolean(): boolean {
    this.i += 1;
    const c = this.peek();
    if (c !== '0' && c !== '1') this.fail('bad boolean');
    this.i += 1;
    return c === '1';
  }
}

/** RFC 9651 4.2 Dictionary, members in wire order. */
export function parseDictionary(input: string): Array<[string, SfMember]> {
  const p = new Parser(input);
  p.start();
  return p.finish(p.dictionary());
}

/** RFC 9651 4.2 Item. */
export function parseItem(input: string): SfItem {
  const p = new Parser(input);
  p.start();
  return p.finish(p.item());
}

/** RFC 9651 4.2 List. */
export function parseList(input: string): SfMember[] {
  const p = new Parser(input);
  p.start();
  return p.finish(p.list());
}

/** A parameter's value by key, or undefined. */
export function paramOf(params: SfParameters | undefined, key: string): SfBareItem | undefined {
  return params?.find(([name]) => name === key)?.[1];
}
