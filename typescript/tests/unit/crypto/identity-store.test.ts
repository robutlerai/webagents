/**
 * The persistent identity store: what it may never do is replace a key file.
 *
 * S-185 (found 2026-09-19): one `try` wrapped read + parse + import and its
 * `catch` stayed silent for an error with no `code`. A JSON SyntaxError has
 * none, so a truncated key file fell through to key generation and
 * `writeFile` overwrote the only copy of the agent's pinned private key.
 * These tests pin the rule that replaced it: a MISSING file is the only
 * reason to generate, every other failure throws naming the file, and what
 * is on disk is byte-identical afterwards.
 *
 * The second half pins rotation parity with the Python store: a
 * `<stem>.ed25519.previous.jwk.json` beside the current key is held and
 * co-signs, so the dual-signing rotation `AgentIdentity` implements is
 * reachable from `serve()`.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtemp, rm, readdir, readFile, writeFile, mkdir, stat, chmod } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { exportJWK, generateKeyPair, calculateJwkThumbprint } from 'jose';
import {
  AgentKeyFileError,
  agentKeyFileNames,
  loadOrCreateAgentIdentity,
} from '../../../src/crypto/identity-store';
import { MAX_HELD_KEYS } from '../../../src/crypto/identity';
import { signMessage } from '../../../src/crypto/http-signature';

const ISSUER = 'https://agent.example.com/agents/mini';
const posix = process.platform !== 'win32';

async function privateJwk(): Promise<{ jwk: Record<string, unknown>; kid: string }> {
  const { privateKey } = await generateKeyPair('EdDSA', { crv: 'Ed25519', extractable: true });
  const jwk = (await exportJWK(privateKey)) as Record<string, unknown>;
  const kid = await calculateJwkThumbprint({ kty: 'OKP', crv: 'Ed25519', x: jwk.x as string }, 'sha256');
  return { jwk, kid };
}

describe('loadOrCreateAgentIdentity', () => {
  let keysDir: string;
  let currentFile: string;
  let previousFile: string;
  let logs: ReturnType<typeof vi.spyOn>;
  let warns: ReturnType<typeof vi.spyOn>;

  beforeEach(async () => {
    keysDir = await mkdtemp(path.join(tmpdir(), 'webagents-store-'));
    const names = agentKeyFileNames('mini');
    currentFile = path.join(keysDir, names.current);
    previousFile = path.join(keysDir, names.previous);
    logs = vi.spyOn(console, 'log').mockImplementation(() => {});
    warns = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(async () => {
    logs.mockRestore();
    warns.mockRestore();
    await rm(keysDir, { recursive: true, force: true });
  });

  it('names its two files after the agent, the previous key beside the current one', () => {
    expect(agentKeyFileNames('mini')).toEqual({
      current: 'mini.ed25519.jwk.json',
      previous: 'mini.ed25519.previous.jwk.json',
    });
    expect(agentKeyFileNames('a/b:c').current).toBe('a_b_c.ed25519.jwk.json');
  });

  describe('a missing file is the only reason to generate (S-185)', () => {
    it('generates when the file is absent, owner-only, and leaves no temporary file behind', async () => {
      const identity = await loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir });
      expect(await readdir(keysDir)).toEqual(['mini.ed25519.jwk.json']);
      const stored = JSON.parse(await readFile(currentFile, 'utf8')) as { kty: string; crv: string; d?: string };
      expect(stored).toMatchObject({ kty: 'OKP', crv: 'Ed25519' });
      expect(typeof stored.d).toBe('string');
      expect(identity.getHeldKeys()).toHaveLength(1);
      if (posix) {
        expect((await stat(currentFile)).mode & 0o777).toBe(0o600);
        expect((await stat(keysDir)).mode & 0o777).toBe(0o700);
      }
    });

    it('refuses a truncated key file and leaves its bytes alone', async () => {
      const { jwk } = await privateJwk();
      const truncated = JSON.stringify(jwk).slice(0, 40);
      await writeFile(currentFile, truncated, { mode: 0o600 });

      const attempt = loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir });
      await expect(attempt).rejects.toBeInstanceOf(AgentKeyFileError);
      await expect(loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir })).rejects.toThrow(currentFile);
      await expect(loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir })).rejects.toMatchObject({ file: currentFile });

      // The only copy of the pinned key is exactly what it was, and nothing was written beside it.
      expect(await readFile(currentFile, 'utf8')).toBe(truncated);
      expect(await readdir(keysDir)).toEqual(['mini.ed25519.jwk.json']);
    });

    it('refuses an empty key file rather than treating it as absent', async () => {
      await writeFile(currentFile, '', { mode: 0o600 });
      await expect(loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir })).rejects.toBeInstanceOf(AgentKeyFileError);
      expect(await readFile(currentFile, 'utf8')).toBe('');
    });

    it('refuses well-formed JSON that is not an Ed25519 private JWK', async () => {
      const { jwk } = await privateJwk();
      const { d: _d, ...publicOnly } = jwk;
      for (const content of [JSON.stringify(publicOnly), JSON.stringify({ kty: 'RSA', n: 'x', e: 'AQAB', d: 'y' }), 'null', '[]']) {
        await writeFile(currentFile, content, { mode: 0o600 });
        await expect(loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir })).rejects.toThrow(/not an Ed25519 private JWK/);
        expect(await readFile(currentFile, 'utf8')).toBe(content);
      }
    });

    it('refuses a JWK whose key material does not import', async () => {
      const content = JSON.stringify({ kty: 'OKP', crv: 'Ed25519', x: 'AAAA', d: 'AAAA' });
      await writeFile(currentFile, content, { mode: 0o600 });
      await expect(loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir })).rejects.toBeInstanceOf(AgentKeyFileError);
      expect(await readFile(currentFile, 'utf8')).toBe(content);
    });

    it('refuses a read error that is not ENOENT (a directory where the key should be)', async () => {
      await mkdir(currentFile);
      await expect(loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir })).rejects.toThrow(/could not be read/);
      expect((await stat(currentFile)).isDirectory()).toBe(true);
    });

    it('two first boots at once end up holding the SAME key: the loser loads the file the winner created', async () => {
      const [a, b, c] = await Promise.all([
        loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir }),
        loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir }),
        loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir }),
      ]);
      expect(b.kid).toBe(a.kid);
      expect(c.kid).toBe(a.kid);
      // And it is the key on disk, so the next boot is the same identity too.
      const next = await loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir });
      expect(next.kid).toBe(a.kid);
      expect(await readdir(keysDir)).toEqual(['mini.ed25519.jwk.json']);
    });
  });

  it.skipIf(!posix)('repairs the permissions of an existing key and its directory on load', async () => {
    const { jwk, kid } = await privateJwk();
    await writeFile(currentFile, JSON.stringify(jwk));
    await chmod(currentFile, 0o644);
    await chmod(keysDir, 0o755);

    const identity = await loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir });
    expect(identity.kid).toBe(kid);
    expect((await stat(currentFile)).mode & 0o777).toBe(0o600);
    expect((await stat(keysDir)).mode & 0o777).toBe(0o700);
  });

  describe('rotation: the previous key file is held and co-signs', () => {
    it('holds <stem>.ed25519.previous.jwk.json beside the current key, current first', async () => {
      const current = await privateJwk();
      const previous = await privateJwk();
      await writeFile(currentFile, JSON.stringify(current.jwk), { mode: 0o600 });
      await writeFile(previousFile, JSON.stringify(previous.jwk), { mode: 0o600 });

      const identity = await loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir });
      expect(identity.kid).toBe(current.kid);
      expect(identity.getJwks().keys.map((k) => k.kid)).toEqual([current.kid, previous.kid]);
      expect(identity.getHeldKeys()).toHaveLength(MAX_HELD_KEYS);

      const signed = await signMessage(identity, { method: 'GET', url: 'https://robutler.ai/api/agents/me' });
      expect(signed.labels.map((l) => [l.label, l.kid])).toEqual([
        ['sig1', current.kid],
        ['sig2', previous.kid],
      ]);
      expect(signed.labels[0].nonce).not.toBe(signed.labels[1].nonce);
    });

    it('is the documented procedure: rename current to previous, restart, and the new key is co-signed by the old', async () => {
      const first = await loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir });
      const { rename } = await import('node:fs/promises');
      await rename(currentFile, previousFile);

      const rotated = await loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir });
      expect(rotated.kid).not.toBe(first.kid);
      expect(rotated.getJwks().keys.map((k) => k.kid)).toEqual([rotated.kid, first.kid]);
      expect((await readdir(keysDir)).sort()).toEqual(['mini.ed25519.jwk.json', 'mini.ed25519.previous.jwk.json']);

      // The next boot holds the same two keys.
      const again = await loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir });
      expect(again.getJwks().keys.map((k) => k.kid)).toEqual([rotated.kid, first.kid]);
    });

    it('holds one key when the previous file is a copy of the current one', async () => {
      const current = await privateJwk();
      await writeFile(currentFile, JSON.stringify(current.jwk), { mode: 0o600 });
      await writeFile(previousFile, JSON.stringify(current.jwk), { mode: 0o600 });
      const identity = await loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir });
      expect(identity.getHeldKeys().map((k) => k.kid)).toEqual([current.kid]);
    });

    it('refuses an unusable previous key BEFORE generating a current one', async () => {
      await writeFile(previousFile, '{"kty":"OKP"', { mode: 0o600 });
      await expect(loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir })).rejects.toMatchObject({
        name: 'AgentKeyFileError',
        file: previousFile,
      });
      expect(await readdir(keysDir)).toEqual(['mini.ed25519.previous.jwk.json']);
    });
  });
});
