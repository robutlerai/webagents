/**
 * The two SDKs' credential floors are ONE decision written twice, so this test
 * reads the OTHER language's source and asserts the two copies still agree.
 *
 * Why it has to read the file rather than compare two TypeScript constants: the
 * drift this catches is somebody adding a billable path in Python and not in
 * TypeScript (or a header name in one and not the other). Nothing inside a
 * single language can see that. The Python half asserts the same thing from the
 * other side, in `python/tests/server/test_floor_parity.py`, so either suite
 * alone catches the drift.
 *
 * The pair matters because the SDKs serve the SAME endpoint for the same
 * platform. A caller probing both must not find `chat/completions` gated on one
 * and open on the other, and the observable refusal must be the same 401 with
 * the same message — the last time these disagreed, one answered 401 and the
 * other a JSON-parser 500 for the same anonymous request.
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  BILLABLE_METHODS,
  BILLABLE_PATHS,
  BILLABLE_WS_PATHS,
  CREDENTIAL_HEADERS,
  PUBLIC_SUBPATHS,
  PUBLIC_WS_SUBPATHS,
  UNAUTHORIZED_MESSAGE,
  WS_CREDENTIAL_QUERY_PARAMS,
} from '../../../src/server/credential-floor.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PYTHON_FLOOR = path.resolve(
  HERE,
  '../../../../python/webagents/server/core/credential_floor.py',
);

/** Every quoted string literal in the right-hand side of `NAME = ...`. */
function pythonLiterals(source: string, name: string): string[] {
  // Match the assignment through its closing bracket/paren, or to end of line
  // for a single-line value.
  const match = new RegExp(`^${name}\\s*=\\s*(\\([\\s\\S]*?\\)|.+)$`, 'm').exec(source);
  if (!match) throw new Error(`${name} not found in ${PYTHON_FLOOR}`);
  return [...match[1].matchAll(/"([^"]*)"|'([^']*)'/g)].map((m) => m[1] ?? m[2]);
}

describe('the Python and TypeScript credential floors do not drift apart', () => {
  const source = fs.readFileSync(PYTHON_FLOOR, 'utf-8');

  it('the Python module is where this test thinks it is', () => {
    // Guard the guard: a moved or renamed file must fail loudly rather than
    // silently stop comparing anything.
    expect(fs.existsSync(PYTHON_FLOOR)).toBe(true);
    expect(source).toContain('BILLABLE_PATHS');
  });

  it('agrees on which headers can carry a credential', () => {
    expect(pythonLiterals(source, 'CREDENTIAL_HEADERS')).toEqual([...CREDENTIAL_HEADERS]);
  });

  it('agrees on the billable path set', () => {
    expect(pythonLiterals(source, 'BILLABLE_PATHS')).toEqual([...BILLABLE_PATHS]);
  });

  it('agrees on which methods the floor applies to', () => {
    expect(pythonLiterals(source, 'BILLABLE_METHODS')).toEqual([...BILLABLE_METHODS]);
  });

  it('agrees on the billable WebSocket path set', () => {
    expect(pythonLiterals(source, 'BILLABLE_WS_PATHS')).toEqual([...BILLABLE_WS_PATHS]);
  });

  it('agrees on which subpaths are declared public', () => {
    // The allow-list is half of the classification, so it drifts the same way
    // the billable set does — and drift here is worse, because a path declared
    // public in one SDK and merely FORGOTTEN in the other looks identical from
    // inside either language.
    expect(pythonLiterals(source, 'PUBLIC_SUBPATHS')).toEqual([...PUBLIC_SUBPATHS]);
  });

  it('agrees on which WebSocket subpaths are declared public', () => {
    expect(pythonLiterals(source, 'PUBLIC_WS_SUBPATHS')).toEqual([...PUBLIC_WS_SUBPATHS]);
  });

  it('declares no subpath both billable and public', () => {
    // Classification means EXACTLY one of the two sets. A path in both would let
    // the enumerating test pass while the floor gates a route the allow-list
    // says is public — or the reverse, which is how a route ends up looking
    // classified and being open.
    const both = (BILLABLE_PATHS as readonly string[]).filter((p) =>
      (PUBLIC_SUBPATHS as readonly string[]).includes(p),
    );
    expect(both).toEqual([]);
    const bothWs = (BILLABLE_WS_PATHS as readonly string[]).filter((p) =>
      (PUBLIC_WS_SUBPATHS as readonly string[]).includes(p),
    );
    expect(bothWs).toEqual([]);
  });

  it('agrees on the WebSocket credential query parameters', () => {
    expect(pythonLiterals(source, 'WS_CREDENTIAL_QUERY_PARAMS')).toEqual([
      ...WS_CREDENTIAL_QUERY_PARAMS,
    ]);
  });

  it('answers with the same refusal message', () => {
    expect(pythonLiterals(source, 'UNAUTHORIZED_MESSAGE').join('')).toBe(UNAUTHORIZED_MESSAGE);
  });
});
