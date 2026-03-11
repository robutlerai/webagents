/**
 * Agent Identity — AOAuth keypair management and JWKS serving.
 *
 * Each standalone agent generates or loads an Ed25519 key pair on startup.
 * The public portion is served at /.well-known/jwks.json so other agents
 * can verify signed responses and AOAuth tokens per the spec.
 */

import { generateKeyPair, exportJWK, SignJWT, type KeyLike } from 'jose';

export interface AgentIdentityConfig {
  /** Agent ID (sub claim in minted tokens) */
  agentId: string;
  /** Issuer URL (iss claim — the agent's public URL) */
  issuer: string;
  /** Stable key ID for JWKS (defaults to agentId) */
  kid?: string;
  /** Pre-generated private key (PEM or KeyLike). If omitted, a new key is generated. */
  privateKey?: KeyLike;
  /** Pre-generated public key */
  publicKey?: KeyLike;
  /** Agent hosting prefix path (agent_path claim) */
  agentPath?: string;
}

export class AgentIdentity {
  readonly agentId: string;
  readonly issuer: string;
  readonly kid: string;
  readonly agentPath?: string;

  private _privateKey: KeyLike | null = null;
  private _publicKey: KeyLike | null = null;
  private _jwksJson: Record<string, unknown> | null = null;

  constructor(config: AgentIdentityConfig) {
    this.agentId = config.agentId;
    this.issuer = config.issuer.replace(/\/$/, '');
    this.kid = config.kid ?? config.agentId;
    this.agentPath = config.agentPath;
    if (config.privateKey) this._privateKey = config.privateKey;
    if (config.publicKey) this._publicKey = config.publicKey;
  }

  async initialize(): Promise<void> {
    if (!this._privateKey || !this._publicKey) {
      const { privateKey, publicKey } = await generateKeyPair('EdDSA', { crv: 'Ed25519' });
      this._privateKey = privateKey;
      this._publicKey = publicKey;
    }

    const jwk = await exportJWK(this._publicKey);
    this._jwksJson = {
      keys: [{
        ...jwk,
        kid: this.kid,
        use: 'sig',
        alg: 'EdDSA',
      }],
    };
  }

  /** JWKS document for /.well-known/jwks.json */
  getJwks(): Record<string, unknown> {
    if (!this._jwksJson) throw new Error('AgentIdentity not initialized');
    return this._jwksJson;
  }

  /** OpenID configuration for /.well-known/openid-configuration */
  getOpenIdConfiguration(): Record<string, unknown> {
    return {
      issuer: this.issuer,
      jwks_uri: `${this.issuer}/.well-known/jwks.json`,
      response_types_supported: ['token'],
      subject_types_supported: ['public'],
      id_token_signing_alg_values_supported: ['EdDSA', 'RS256'],
      scopes_supported: ['read', 'write', 'admin', 'namespace:*', 'tools:*'],
      token_endpoint_auth_methods_supported: ['client_secret_basic', 'client_secret_post'],
      grant_types_supported: ['client_credentials'],
    };
  }

  /**
   * Mint an AOAuth JWT for agent-to-agent communication.
   *
   * @param audience - Target agent URL or identifier
   * @param scopes - Space-separated scope string
   * @param ttlSeconds - Token TTL (default 300s)
   */
  async mintToken(
    audience: string,
    scopes: string,
    ttlSeconds = 300,
  ): Promise<string> {
    if (!this._privateKey) throw new Error('AgentIdentity not initialized');

    const now = Math.floor(Date.now() / 1000);
    const builder = new SignJWT({
      scope: scopes,
      client_id: this.agentId,
      token_type: 'Bearer',
      ...(this.agentPath ? { agent_path: this.agentPath } : {}),
    })
      .setProtectedHeader({ alg: 'EdDSA', kid: this.kid })
      .setIssuedAt(now)
      .setNotBefore(now)
      .setExpirationTime(now + ttlSeconds)
      .setSubject(this.agentId)
      .setIssuer(this.issuer)
      .setAudience(audience)
      .setJti(crypto.randomUUID());

    return builder.sign(this._privateKey);
  }

  /**
   * Sign arbitrary content (for NLI response signing).
   * Returns a base64-encoded Ed25519 signature.
   */
  async sign(data: Uint8Array): Promise<string> {
    if (!this._privateKey) throw new Error('AgentIdentity not initialized');

    const sig = await crypto.subtle.sign(
      'Ed25519',
      this._privateKey as CryptoKey,
      data as unknown as ArrayBuffer,
    );
    return btoa(String.fromCharCode(...new Uint8Array(sig)));
  }

  get publicKey(): KeyLike | null {
    return this._publicKey;
  }

  get privateKey(): KeyLike | null {
    return this._privateKey;
  }
}
