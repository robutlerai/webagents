/**
 * One portal for `login` and every command after it (2026-09-24).
 *
 * `login --url` defaulted to a hardcoded `https://robutler.ai` and never read
 * `platform.url`; `platformAuth()` read `platform.url` and could never reach the
 * URL `login` saved, because `ConfigStore.get` returns the DEFAULT for an unset
 * key, which made the `auth.json` fallback behind it unreachable. So a login to a
 * local cluster stored that cluster's token, and the next `sync` sent it to
 * production. The Python CLI had the same split the other way round.
 *
 * `resolvePlatformUrl` is the shared answer, and these pin that it matches the
 * Python resolver's contract rather than only being self-consistent.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

import {
  ConfigStore,
  DEFAULTS,
  PLATFORM_URL_ENV_VAR,
  resolvePlatformUrl,
} from '../../../src/cli/config-store';
import { tempDirs, TSX_CLI } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

const REPO = path.resolve(__dirname, '../../../..');
const CLI_SOURCE = path.resolve(__dirname, '../../../src/cli/index.ts');

describe('resolvePlatformUrl', () => {
  const saved = {
    HOME: process.env.HOME,
    WEBAGENTS_PROFILE: process.env.WEBAGENTS_PROFILE,
    ROBUTLER_API_URL: process.env.ROBUTLER_API_URL,
  };
  let home: string;
  let cwd: string;

  beforeEach(() => {
    home = tempDir('wa-url-home-');
    cwd = tempDir('wa-url-cwd-');
    process.env.HOME = home;
    delete process.env.WEBAGENTS_PROFILE;
    delete process.env.ROBUTLER_API_URL;
  });

  afterEach(() => {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });

  it('falls back to the default, and says it is the default', () => {
    expect(resolvePlatformUrl(undefined, cwd)).toEqual([DEFAULTS['platform.url'], 'default']);
  });

  it('reads platform.url, which login used to ignore', () => {
    new ConfigStore({ cwd }).set('platform.url', 'https://cluster.example');
    expect(resolvePlatformUrl(undefined, cwd)).toEqual(['https://cluster.example', 'global']);
  });

  it('lets the environment outrank config', () => {
    new ConfigStore({ cwd }).set('platform.url', 'https://cluster.example');
    process.env.ROBUTLER_API_URL = 'https://from-env.example';
    expect(resolvePlatformUrl(undefined, cwd)).toEqual([
      'https://from-env.example',
      PLATFORM_URL_ENV_VAR,
    ]);
  });

  it('strips a trailing slash so routes never double it', () => {
    process.env.ROBUTLER_API_URL = 'https://cluster.example/';
    expect(resolvePlatformUrl(undefined, cwd)[0]).toBe('https://cluster.example');
  });

  it('keeps each profile on its own portal', () => {
    new ConfigStore({ cwd }).set('platform.url', 'https://production.example');
    process.env.WEBAGENTS_PROFILE = 'local';
    new ConfigStore({ cwd }).set('platform.url', 'https://cluster.example');

    expect(resolvePlatformUrl(undefined, cwd)[0]).toBe('https://cluster.example');
    delete process.env.WEBAGENTS_PROFILE;
    expect(resolvePlatformUrl(undefined, cwd)[0]).toBe('https://production.example');
  });
});

describe('the two SDKs agree on the contract', () => {
  it('reads the same environment variable as the Python resolver', () => {
    const py = fs.readFileSync(
      path.join(REPO, 'python/webagents/cli/config_store.py'),
      'utf-8',
    );
    expect(py).toContain(`PLATFORM_URL_ENV_VAR = "${PLATFORM_URL_ENV_VAR}"`);
  });
});

describe('login no longer carries its own answer', () => {
  const source = fs.readFileSync(CLI_SOURCE, 'utf-8');

  it('has no hardcoded default portal on --url', () => {
    // The old option read: .option('-u, --url <url>', 'Portal URL', 'https://robutler.ai')
    expect(source).not.toMatch(/--url <url>'[^)]*'https:\/\/robutler\.ai'\)/);
  });

  it('points at a page that exists', () => {
    // `/settings/api-keys` was never a portal page; keys are on the Developer tab.
    // Matched on the URL being BUILT, not the words: the comment that explains
    // the change names the old path, and should.
    expect(source).not.toContain('}/settings/api-keys');
    expect(source).toContain('}/settings?tab=developer');
  });

  it('no longer writes a URL file that nothing reads', () => {
    expect(source).not.toContain('saveAuth(');
    expect(source).not.toContain('loadAuth(');
  });
});

describe('link, run for real against a stub portal', () => {
  it('goes where the resolver says and finds the agent by username', async () => {
    const http = await import('node:http');
    const { spawn } = await import('node:child_process');

    const seen: { path?: string; auth?: string } = {};
    const server = http.createServer((req, res) => {
      seen.path = req.url;
      seen.auth = req.headers.authorization;
      res.setHeader('content-type', 'application/json');
      // The shape the real list route answers with: `name` is null.
      res.end(
        JSON.stringify({
          agents: [{ id: 'a-1', username: 'owner.first', displayName: 'First', name: null }],
        }),
      );
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = (server.address() as { port: number }).port;
    const home = tempDir('wa-link-home-');
    const cwd = tempDir('wa-link-cwd-');

    try {
      // spawn, not spawnSync: the stub server lives on THIS event loop, and a
      // synchronous child would block it from ever answering.
      const out = await new Promise<{ code: number | null; stdout: string }>((resolve) => {
        const child = spawn(process.execPath, [TSX_CLI, CLI_SOURCE, 'link', 'first'], {
          cwd,
          env: {
            ...process.env,
            HOME: home,
            WEBAGENTS_TOKEN: 'stub-token',
            ROBUTLER_API_URL: `http://127.0.0.1:${port}`,
            WEBAGENTS_PROFILE: '',
            WEBAGENTS_SECRETS_BACKEND: 'file',
          },
        });
        let stdout = '';
        child.stdout.on('data', (d) => (stdout += d));
        child.on('close', (code) => resolve({ code, stdout }));
      });

      expect(out.code).toBe(0);
      // It went to the resolved portal, with the bearer...
      expect(seen.path).toBe('/api/agents');
      expect(seen.auth).toBe('Bearer stub-token');
      // ...and matched the agent by its username, whose `name` is null.
      expect(out.stdout).toContain('Linked this folder to owner.first.');
    } finally {
      server.close();
    }
  }, 60_000);
});

describe('login --url, run for real against a stub token endpoint', () => {
  async function loginAgainstStub(extraEnv: Record<string, string> = {}) {
    const http = await import('node:http');
    const { spawn } = await import('node:child_process');

    const server = http.createServer((req, res) => {
      req.resume();
      req.on('end', () => {
        res.setHeader('content-type', 'application/json');
        // The shape `/api/auth/cli/token` answers with.
        res.end(JSON.stringify({ access_token: 'exchanged-dummy', username: 'owner' }));
      });
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const url = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
    const home = tempDir('wa-login-home-');

    try {
      const out = await new Promise<{ code: number | null; stdout: string }>((resolve) => {
        const child = spawn(process.execPath, [TSX_CLI, CLI_SOURCE, 'login', '--url', url, '--token', 'dummy-key'], {
          env: {
            ...process.env,
            HOME: home,
            WEBAGENTS_PROFILE: '',
            WEBAGENTS_SECRETS_BACKEND: 'file',
            ROBUTLER_API_URL: '',
            ...extraEnv,
          },
        });
        let stdout = '';
        child.stdout.on('data', (d) => (stdout += d));
        child.on('close', (code) => resolve({ code, stdout }));
      });
      return { ...out, url, home };
    } finally {
      server.close();
    }
  }

  it('records the portal it logged in to, so the next command goes there too', async () => {
    const { code, stdout, url, home } = await loginAgainstStub();
    expect(code).toBe(0);
    expect(stdout).toContain(`on ${url}`);

    // THE BUG: `--url` was written to auth.json, which nothing read, so the
    // next command sent this portal's token to production.
    const config = JSON.parse(
      fs.readFileSync(path.join(home, '.webagents', 'config.json'), 'utf-8'),
    ) as Record<string, string>;
    expect(config['platform.url']).toBe(url);
    expect(fs.existsSync(path.join(home, '.webagents', 'auth.json'))).toBe(false);
  }, 60_000);

  it('says so, rather than writing config, when the environment would outrank it', async () => {
    const { code, stdout, home } = await loginAgainstStub({
      ROBUTLER_API_URL: 'https://elsewhere.example',
    });
    expect(code).toBe(0);
    expect(stdout).toContain('ROBUTLER_API_URL=https://elsewhere.example is set');
    // Writing platform.url would change nothing while the variable is set.
    expect(fs.existsSync(path.join(home, '.webagents', 'config.json'))).toBe(false);
  }, 60_000);
});
