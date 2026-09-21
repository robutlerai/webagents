export { JWKSManager, type JWKSManagerConfig } from './jwks';
export {
  AgentIdentity,
  type AgentIdentityConfig,
  type HeldKey,
  type HeldKeyPair,
  type PublishedJwk,
} from './identity';
// The Web Bot Auth request signer (ADR 0038 step 5, 2026-09-17): what
// `registerWithPlatform` signs with, exported so an agent can sign its own
// calls to the platform the same way.
export {
  signRequest,
  signedFetch,
  signMessage,
  contentDigest,
  randomNonce,
  buildSignatureBase,
  signatureAgentValue,
  assertSignableAgentUrl,
  normalizeCoveredHeaders,
  COVERED_HEADERS_RESERVED,
  SIGNATURE_LIFETIME_SECONDS,
  SIGNATURE_AGENT_FORMS,
  type SignatureAgentForm,
  type SignRequestOptions,
  type SigningIdentity,
  type SignedHeaders,
  type SignedMessage,
  type MessageToSign,
} from './http-signature';
