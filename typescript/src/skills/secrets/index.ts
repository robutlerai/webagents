export { SecretsSkill } from './skill';
export type { SecretsSkillConfig } from './skill';
export {
  SecretStore,
  KeystoreUnavailableError,
  legacyServiceKey,
  openSecretStore,
  serviceKey,
} from './store';
// Keychain dialogs on macOS (2026-09-27): each SDK's own item names, and the
// switch a server throws so nothing waits on a dialog nobody can answer.
export { KeychainDialogBlocked, forbidKeychainDialogs } from './keychain-ux';
export type {
  SecretBackendKind,
  SecretBackendStatus,
  SecretStoreLike,
  SecretStoreOptions,
} from './store';
// `${secret:NAME}` / `${env:NAME}` references (S-292): the grammar, the
// resolver and the masking, for a host that reads or reports MCP configs.
export {
  REFERENCE_NAME,
  REFERENCE_SENTENCES,
  SECRET_LOOKING_PATTERNS,
  SECRET_MASK,
  SecretReferenceError,
  expandReferences,
  hasReference,
  looksLikeSecret,
  maskMap,
  maskText,
  maskUrl,
  maskValue,
  tokenizeReferences,
} from './references';
export type { ExpandedValue, ReferenceLookup, ReferenceScheme, SecretReference } from './references';
