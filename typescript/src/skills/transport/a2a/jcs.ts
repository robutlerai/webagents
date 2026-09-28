/**
 * RFC 8785 JSON Canonicalization Scheme (JCS), the form an A2A agent card is
 * signed over (A2A v1.0 section 8.4) and the form every verifier rebuilds.
 *
 * Written against the standard library only (plan item 1.3, 2026-09-26): a
 * signature that depends on a canonicaliser package pins the package's bugs
 * into every card, and the Python twin (`a2a/jcs.py`) has to produce the same
 * bytes from the same value, so the rules are spelled out here rather than
 * inherited:
 *
 *   * object keys sorted by UTF-16 code units (what `Array.prototype.sort`
 *     does to strings, and what Python has to emulate through UTF-16BE);
 *   * no whitespace;
 *   * numbers as ES6 `Number.prototype.toString` prints them, so `1.0` is `1`,
 *     `1e-6` is `0.000001` and `-0` is `0` (`JSON.stringify` already does
 *     this; the Python side has to reproduce it by hand);
 *   * strings with the short escapes (`\b \f \n \r \t \" \\`), other control
 *     characters as lowercase `\u00xx`, everything else literal UTF-8.
 *
 * `JSON.stringify` implements every string and number rule, so the only work
 * here is key order and the refusal of values JSON cannot carry (NaN,
 * Infinity, undefined, functions), which a canonical form must not silently
 * turn into `null`.
 */

export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

/** The canonical (RFC 8785) serialisation of `value`. */
export function canonicalize(value: unknown): string {
  return serialize(value, []);
}

/** The canonical bytes, what a signature covers. */
export function canonicalBytes(value: unknown): Uint8Array {
  return new TextEncoder().encode(canonicalize(value));
}

function serialize(value: unknown, path: string[]): string {
  if (value === null) return 'null';
  switch (typeof value) {
    case 'boolean':
      return value ? 'true' : 'false';
    case 'number':
      if (!Number.isFinite(value)) throw new Error(`JCS: ${describe(path)} is ${String(value)}, which JSON cannot carry`);
      return JSON.stringify(value);
    case 'string':
      return JSON.stringify(value);
    case 'object':
      if (Array.isArray(value)) {
        return `[${value.map((item, index) => serialize(item, [...path, String(index)])).join(',')}]`;
      }
      return serializeObject(value as Record<string, unknown>, path);
    default:
      throw new Error(`JCS: ${describe(path)} is a ${typeof value}, which JSON cannot carry`);
  }
}

function serializeObject(value: Record<string, unknown>, path: string[]): string {
  // Default string sort is by UTF-16 code unit, the order RFC 8785 asks for.
  const keys = Object.keys(value).sort();
  const members: string[] = [];
  for (const key of keys) {
    const member = value[key];
    if (member === undefined) continue; // `JSON.stringify` drops these too
    members.push(`${JSON.stringify(key)}:${serialize(member, [...path, key])}`);
  }
  return `{${members.join(',')}}`;
}

function describe(path: string[]): string {
  return path.length ? `"${path.join('.')}"` : 'the value';
}
