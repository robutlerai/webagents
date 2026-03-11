/**
 * AgentIdentity Unit Tests
 *
 * Tests AOAuth key generation, JWKS export, OpenID configuration, and token minting.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { AgentIdentity } from '../../../src/crypto/identity.js';
import { jwtVerify, importJWK } from 'jose';

describe('AgentIdentity', () => {
  let identity: AgentIdentity;

  beforeEach(async () => {
    identity = new AgentIdentity({
      agentId: 'test-agent',
      issuer: 'https://example.com/agents/test-agent',
    });
    await identity.initialize();
  });

  describe('initialization', () => {
    it('generates a key pair on initialize', async () => {
      expect(identity.publicKey).toBeDefined();
      expect(identity.privateKey).toBeDefined();
    });

    it('sets agentId and issuer', () => {
      expect(identity.agentId).toBe('test-agent');
      expect(identity.issuer).toBe('https://example.com/agents/test-agent');
    });

    it('strips trailing slash from issuer', () => {
      const id = new AgentIdentity({
        agentId: 'a',
        issuer: 'https://example.com/agents/a/',
      });
      expect(id.issuer).toBe('https://example.com/agents/a');
    });

    it('defaults kid to agentId', () => {
      expect(identity.kid).toBe('test-agent');
    });

    it('accepts custom kid', async () => {
      const custom = new AgentIdentity({
        agentId: 'a',
        issuer: 'https://example.com',
        kid: 'custom-key-id',
      });
      await custom.initialize();
      expect(custom.kid).toBe('custom-key-id');
    });
  });

  describe('getJwks', () => {
    it('returns a JWKS with one key', () => {
      const jwks = identity.getJwks();
      expect(jwks.keys).toBeDefined();
      expect(Array.isArray(jwks.keys)).toBe(true);
      expect((jwks.keys as unknown[]).length).toBe(1);
    });

    it('key has correct metadata', () => {
      const jwks = identity.getJwks();
      const key = (jwks.keys as Record<string, string>[])[0];
      expect(key.kid).toBe('test-agent');
      expect(key.use).toBe('sig');
      expect(key.alg).toBe('EdDSA');
      expect(key.kty).toBe('OKP');
      expect(key.crv).toBe('Ed25519');
    });

    it('throws if not initialized', () => {
      const uninitialized = new AgentIdentity({
        agentId: 'x',
        issuer: 'https://example.com',
      });
      expect(() => uninitialized.getJwks()).toThrow('not initialized');
    });
  });

  describe('getOpenIdConfiguration', () => {
    it('returns correct issuer', () => {
      const config = identity.getOpenIdConfiguration();
      expect(config.issuer).toBe('https://example.com/agents/test-agent');
    });

    it('returns correct jwks_uri', () => {
      const config = identity.getOpenIdConfiguration();
      expect(config.jwks_uri).toBe('https://example.com/agents/test-agent/.well-known/jwks.json');
    });

    it('includes expected fields', () => {
      const config = identity.getOpenIdConfiguration();
      expect(config.response_types_supported).toBeDefined();
      expect(config.scopes_supported).toBeDefined();
      expect(config.grant_types_supported).toContain('client_credentials');
      expect((config.id_token_signing_alg_values_supported as string[])).toContain('EdDSA');
    });
  });

  describe('mintToken', () => {
    it('produces a valid JWT', async () => {
      const token = await identity.mintToken('https://target.com', 'read write');
      expect(typeof token).toBe('string');
      expect(token.split('.')).toHaveLength(3);
    });

    it('token is verifiable with the JWKS key', async () => {
      const token = await identity.mintToken('https://target.com', 'read');
      const jwks = identity.getJwks();
      const keyData = (jwks.keys as Record<string, unknown>[])[0];
      const publicKey = await importJWK(keyData, 'EdDSA');
      const { payload } = await jwtVerify(token, publicKey, {
        issuer: 'https://example.com/agents/test-agent',
        audience: 'https://target.com',
      });
      expect(payload.scope).toBe('read');
      expect(payload.sub).toBe('test-agent');
      expect(payload.client_id).toBe('test-agent');
      expect(payload.iss).toBe('https://example.com/agents/test-agent');
    });

    it('respects TTL', async () => {
      const token = await identity.mintToken('https://target.com', 'read', 60);
      const jwks = identity.getJwks();
      const keyData = (jwks.keys as Record<string, unknown>[])[0];
      const publicKey = await importJWK(keyData, 'EdDSA');
      const { payload } = await jwtVerify(token, publicKey);
      const exp = payload.exp!;
      const iat = payload.iat!;
      expect(exp - iat).toBe(60);
    });

    it('includes agent_path when configured', async () => {
      const idWithPath = new AgentIdentity({
        agentId: 'agent-x',
        issuer: 'https://example.com',
        agentPath: '/agents/agent-x',
      });
      await idWithPath.initialize();
      const token = await idWithPath.mintToken('https://target.com', 'read');
      const jwks = idWithPath.getJwks();
      const keyData = (jwks.keys as Record<string, unknown>[])[0];
      const publicKey = await importJWK(keyData, 'EdDSA');
      const { payload } = await jwtVerify(token, publicKey);
      expect(payload.agent_path).toBe('/agents/agent-x');
    });

    it('includes unique jti', async () => {
      const token1 = await identity.mintToken('https://a.com', 'read');
      const token2 = await identity.mintToken('https://a.com', 'read');
      const jwks = identity.getJwks();
      const keyData = (jwks.keys as Record<string, unknown>[])[0];
      const publicKey = await importJWK(keyData, 'EdDSA');
      const { payload: p1 } = await jwtVerify(token1, publicKey);
      const { payload: p2 } = await jwtVerify(token2, publicKey);
      expect(p1.jti).toBeDefined();
      expect(p2.jti).toBeDefined();
      expect(p1.jti).not.toBe(p2.jti);
    });

    it('throws if not initialized', async () => {
      const uninitialized = new AgentIdentity({
        agentId: 'x',
        issuer: 'https://example.com',
      });
      await expect(uninitialized.mintToken('a', 'b')).rejects.toThrow('not initialized');
    });
  });
});
