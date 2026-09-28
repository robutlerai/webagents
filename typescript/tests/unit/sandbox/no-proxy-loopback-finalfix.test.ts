/**
 * A `network:` host on loopback or a private range is reachable from a
 * confined command (2026-09-27, the final e2e re-run, `g1b-netdebug`). srt
 * exports `NO_PROXY=localhost,127.0.0.1,...` inside the sandbox, ordinary
 * clients therefore bypassed its proxy, and their direct socket was refused,
 * so a listed loopback host answered nothing. The command now starts with a
 * NO_PROXY from which every entry covering a `network:` host is removed,
 * pinned by `no_proxy` in the shared fixture `python/tests/fixtures/sandbox/
 * srt.json` (the Python suite reads the same cases), and proved against real
 * srt: a plain `curl`, no `--noproxy`, to a listed `127.0.0.1:<port>` and to a
 * listed `localhost:<port>` answers, while an unlisted loopback name still
 * does not.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  SRT_NO_PROXY,
  backendStatus,
  networkHost,
  noProxyEntryCovers,
  noProxyFor,
  parseSandboxDeclaration,
  policyFromDeclaration,
  runSandboxed,
  wrappedCommand,
} from '../../../src/sandbox/index';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/sandbox/srt.json'), 'utf8')) as {
  no_proxy: {
    srt_entries: string[];
    cases: Array<{ network: string[]; kept: string[] | null }>;
    wrapped: { command: string; path: string; network: string[]; text: string; unchanged_text: string; all_removed_text: string };
  };
};
const NO_PROXY = FIXTURE.no_proxy;
const tempDir = tempDirs();

const status = backendStatus();
const forReal = status.available ? it : it.skip;
if (!status.available) console.warn(`srt enforcement tests skipped: ${status.reason}`);

function policy(declared: Record<string, unknown>, cwd: string) {
  return policyFromDeclaration(parseSandboxDeclaration(declared), { cwd });
}

describe('the fixture is the contract', () => {
  it("pins srt's list", () => {
    expect([...SRT_NO_PROXY]).toEqual(NO_PROXY.srt_entries);
  });

  it.each(NO_PROXY.cases)('network $network keeps $kept', ({ network, kept }) => {
    expect(noProxyFor(network)).toEqual(kept ?? NO_PROXY.srt_entries);
    // The second line is NODE_USE_ENV_PROXY=1 since 2026-09-27 (fixture `node_proxy`).
    const wrapped = wrappedCommand('true', '/usr/bin', network);
    const head = 'export PATH=/usr/bin\nexport NODE_USE_ENV_PROXY=1\nexport npm_config_cache="${TMPDIR:-/tmp}/npm-cache"\n';
    if (kept === null) expect(wrapped).toBe(`${head}true`);
    else if (kept.length) expect(wrapped).toBe(`${head}export NO_PROXY=${kept.join(',')} no_proxy=${kept.join(',')}\ntrue`);
    else expect(wrapped).toBe(`${head}unset NO_PROXY no_proxy\ntrue`);
  });

  it('wraps the command as the fixture spells it', () => {
    const { command, path: userPath, network, text, unchanged_text, all_removed_text } = NO_PROXY.wrapped;
    expect(wrappedCommand(command, userPath, network)).toBe(text);
    expect(wrappedCommand(command, userPath)).toBe(unchanged_text);
    expect(wrappedCommand(command, userPath, ['github.com'])).toBe(unchanged_text);
    expect(wrappedCommand(command, userPath, NO_PROXY.cases[NO_PROXY.cases.length - 1].network)).toBe(all_removed_text);
  });

  it('reads the host of an entry as the fixture says', () => {
    expect(networkHost('*.Example.com')).toBe('example.com');
    expect(networkHost('127.0.0.1:8080')).toBe('127.0.0.1');
    expect(networkHost('[::1]:9000')).toBe('::1');
    expect(networkHost('[::1]')).toBe('::1');
    expect(networkHost('localhost.')).toBe('localhost');
  });

  it('covers by suffix for names, by equality for addresses, by membership for ranges', () => {
    expect(noProxyEntryCovers('localhost', 'api.localhost')).toBe(true);
    expect(noProxyEntryCovers('localhost', 'notlocalhost.example')).toBe(false);
    expect(noProxyEntryCovers('localhost', '127.0.0.1')).toBe(false);
    expect(noProxyEntryCovers('127.0.0.1', 'localhost')).toBe(false);
    expect(noProxyEntryCovers('10.0.0.0/8', 'ten.example')).toBe(false);
    expect(noProxyEntryCovers('10.0.0.0/8', '10.255.255.255')).toBe(true);
    expect(noProxyEntryCovers('10.0.0.0/8', '11.0.0.1')).toBe(false);
    expect(noProxyEntryCovers('::1', '[::1]:80')).toBe(true);
  });
});

describe('against real srt, a listed loopback host answers a plain client', () => {
  let server: http.Server;
  let port: number;
  beforeAll(async () => {
    server = http.createServer((_req, res) => {
      res.writeHead(200, { 'Content-Type': 'text/plain' });
      res.end('HELLO-FROM-SITE\n');
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    port = (server.address() as AddressInfo).port;
  });
  afterAll(async () => {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });

  function work(): string {
    const dir = path.join(tempDir('wa-noproxy-'), 'work');
    fs.mkdirSync(dir);
    return fs.realpathSync(dir);
  }

  forReal('127.0.0.1:<port> listed: curl with no --noproxy answers, and NO_PROXY inside no longer names it', async () => {
    const dir = work();
    const allowed = policy({ preset: 'development', allowed_folders: [dir], network: [`127.0.0.1:${port}`] }, dir);
    const inside = await runSandboxed('echo "np=[$NO_PROXY] lc=[$no_proxy]"', allowed, { timeout: 30 });
    expect(inside.stdout.trim(), inside.stderr).toBe(`np=[${noProxyFor([`127.0.0.1:${port}`]).join(',')}] lc=[${noProxyFor([`127.0.0.1:${port}`]).join(',')}]`);
    const served = await runSandboxed(`curl -sf -m 4 http://127.0.0.1:${port}/`, allowed, { timeout: 30 });
    expect(served.stdout.trim(), served.stderr).toBe('HELLO-FROM-SITE');
  }, 90_000);

  forReal('localhost:<port> listed: curl to localhost answers; an unlisted loopback name still does not', async () => {
    const dir = work();
    const byName = policy({ preset: 'development', allowed_folders: [dir], network: [`localhost:${port}`] }, dir);
    const served = await runSandboxed(`curl -sf -m 4 http://localhost:${port}/`, byName, { timeout: 30 });
    expect(served.stdout.trim(), served.stderr).toBe('HELLO-FROM-SITE');
    // Only the address is listed: the name stays in NO_PROXY, goes direct, and is refused.
    const byAddress = policy({ preset: 'development', allowed_folders: [dir], network: [`127.0.0.1:${port}`] }, dir);
    const refused = await runSandboxed(`curl -sf -m 4 http://localhost:${port}/ && echo NET`, byAddress, { timeout: 30 });
    expect(refused.stdout).not.toContain('NET');
  }, 90_000);

  forReal('nothing listed: the same server stays unreachable, direct and through the proxy', async () => {
    const dir = work();
    const denied = policy({ preset: 'development', allowed_folders: [dir] }, dir);
    const direct = await runSandboxed(`curl -sf -m 4 http://127.0.0.1:${port}/ && echo NET`, denied, { timeout: 30 });
    expect(direct.stdout).not.toContain('NET');
    const viaProxy = await runSandboxed(`curl -sf -m 4 --noproxy '' http://127.0.0.1:${port}/ && echo NET`, denied, { timeout: 30 });
    expect(viaProxy.stdout).not.toContain('NET');
  }, 90_000);
});
