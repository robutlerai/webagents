/**
 * S-298 (2026-09-26): a caller principal with a comma is not a namespace, and
 * namespace lists travel to the portal as repeated parameters, never joined.
 * The twin of the Python `test_memory_namespace_comma_w2fix.py`; both read
 * `namespace_grammar` of the shared fixture `memory_tool/definition.json`.
 *
 *   - the grammar refuses the comma probe and `namespaceOf` gives such a
 *     caller no namespace (it reads `shared` alone);
 *   - the portal store sends one `namespace` parameter per value and one
 *     `excludeSource` per source, and post-filters what the platform answers
 *     to the namespaces it asked for.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { NAMESPACE_RE, namespaceOf, readableNamespaces } from '../../../../src/skills/memory/namespace';
import { PortalMemoryStore } from '../../../../src/skills/memory/portal-store';
import { fakePortal } from './fake-portal-w2mem';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../../python/tests/fixtures/memory_tool/definition.json'), 'utf8'));
const G = FIXTURE.namespace_grammar as {
  valid: string[];
  invalid: string[];
  probe: { principal: string; namespace: string; namespace_of: null };
  list_wire: { parameter: string; retired: string[]; exclude_parameter: string };
};

describe('the grammar (shared fixture)', () => {
  it('accepts the valid shapes and refuses every invalid one, the comma probe included', () => {
    for (const ns of G.valid) expect(NAMESPACE_RE.test(ns), ns).toBe(true);
    for (const ns of G.invalid) expect(NAMESPACE_RE.test(ns), ns).toBe(false);
  });

  it('the probe: a verified principal with a comma gets no namespace, so it reads shared alone and writes nothing', () => {
    const auth = { authenticated: true, provider: 'platform', scopes: [], principals: [G.probe.principal] };
    expect(namespaceOf(auth as never)).toBe(G.probe.namespace_of);
    expect(readableNamespaces(namespaceOf(auth as never))).toEqual(['shared']);
    // The same principal without the comma is a caller like any other.
    const clean = { ...auth, principals: [G.probe.principal.replace(',owner', '')] };
    expect(namespaceOf(clean as never)).toBe(`caller:${G.probe.principal.replace(',owner', '')}`);
  });
});

describe('the portal store', () => {
  function store(portal: ReturnType<typeof fakePortal>) {
    return new PortalMemoryStore({ portalUrl: 'https://portal.test', token: () => 'agent-key', agentId: portal.agentId, fetchImpl: portal.fetch });
  }

  it('sends one namespace parameter per value and one excludeSource per source, never a joined list', async () => {
    const portal = fakePortal();
    const s = store(portal);
    await s.list(['caller:user:alice', 'shared'], { excludeSources: ['compaction', 'sync'] });
    await s.search('short', ['caller:user:alice', 'shared'], 5);
    const [list, search] = portal.seen;
    expect(list.url.searchParams.getAll(G.list_wire.parameter)).toEqual(['caller:user:alice', 'shared']);
    expect(list.url.searchParams.getAll(G.list_wire.exclude_parameter)).toEqual(['compaction', 'sync']);
    expect(search.url.searchParams.getAll(G.list_wire.parameter)).toEqual(['caller:user:alice', 'shared']);
    for (const retired of G.list_wire.retired) {
      expect(list.url.searchParams.has(retired)).toBe(false);
      expect(search.url.searchParams.has(retired)).toBe(false);
    }
    // A null list asks for every namespace: no parameter at all.
    await s.list(null);
    expect(portal.seen[2].url.searchParams.has(G.list_wire.parameter)).toBe(false);
  });

  it('post-filters what the platform answers to the namespaces it asked for', async () => {
    const portal = fakePortal();
    // A platform that answered more than asked (the pre-fix split): the store drops the owner's row.
    const leaky = (async (input: string | URL | Request, init?: RequestInit) => {
      const res = await portal.fetch(input, init);
      const body = (await res.json()) as { entries?: Array<Record<string, unknown>> };
      if (body.entries) body.entries.push({ id: 'leak', namespace: 'owner', key: 'plan', content: 'October', source: 'owner', created_at: 'x', updated_at: 'x' });
      return new Response(JSON.stringify(body), { status: res.status, headers: { 'Content-Type': 'application/json' } });
    }) as unknown as typeof fetch;
    portal.put('caller:user:alice', 'preferences', 'short', 'tool');
    const s = new PortalMemoryStore({ portalUrl: 'https://portal.test', token: () => 'agent-key', agentId: portal.agentId, fetchImpl: leaky });
    const listed = await s.list(['caller:user:alice', 'shared']);
    expect(listed.map((e) => `${e.namespace}/${e.key}`)).toEqual(['caller:user:alice/preferences']);
    const found = await s.search('short', ['caller:user:alice', 'shared']);
    expect(found.map((e) => e.namespace)).toEqual(['caller:user:alice']);
    // Asking for everything keeps everything.
    expect((await s.list(null)).some((e) => e.namespace === 'owner')).toBe(true);
  });
});
