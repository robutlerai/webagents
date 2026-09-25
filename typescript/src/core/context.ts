/**
 * Request Context
 * 
 * Context provides access to session state, authentication, and other
 * request-scoped data for tool and hook handlers.
 */

import type {
  Context,
  AuthInfo,
  PaymentInfo,
  SessionState,
} from './types';
import { callerScopes, scopeAllows } from './scopes';
import type { Capabilities } from '../uamp/types';

/**
 * Create a new session state
 */
export function createSessionState(id?: string): SessionState {
  return {
    id: id || crypto.randomUUID(),
    created_at: Date.now(),
    last_activity: Date.now(),
    data: {},
  };
}

/**
 * Create default authentication info (unauthenticated)
 */
export function createDefaultAuthInfo(): AuthInfo {
  return {
    authenticated: false,
  };
}

/**
 * Create default payment info (invalid)
 */
export function createDefaultPaymentInfo(): PaymentInfo {
  return {
    valid: false,
  };
}

/**
 * Context implementation
 */
export class ContextImpl implements Context {
  session: SessionState;
  auth: AuthInfo;
  payment: PaymentInfo;
  client_capabilities?: Capabilities;
  agent_capabilities?: Capabilities;
  metadata: Record<string, unknown>;
  signal?: AbortSignal;
  
  constructor(options: Partial<Context> = {}) {
    this.session = options.session || createSessionState();
    this.auth = options.auth || createDefaultAuthInfo();
    this.payment = options.payment || createDefaultPaymentInfo();
    this.client_capabilities = options.client_capabilities;
    this.agent_capabilities = options.agent_capabilities;
    this.metadata = options.metadata || {};
    this.signal = options.signal;
  }
  
  /**
   * Get session data by key
   */
  get<T = unknown>(key: string): T | undefined {
    return this.session.data[key] as T | undefined;
  }
  
  /**
   * Set session data
   */
  set<T = unknown>(key: string, value: T): void {
    this.session.data[key] = value;
    this.session.last_activity = Date.now();
  }
  
  /**
   * Delete session data
   */
  delete(key: string): void {
    delete this.session.data[key];
    this.session.last_activity = Date.now();
  }
  
  /**
   * Whether the caller may use something declared with `scope`: the one rule
   * both SDKs share (`./scopes`, ADR-0045). An admin passes `owner`, the owner
   * and an admin pass every `group:` scope, and an unknown scope needs the
   * caller to hold it.
   */
  hasScope(scope: string): boolean {
    return scopeAllows(scope, callerScopes(this.auth));
  }
  
  /**
   * Check if user has all specified scopes. Declared scope LISTS are any-of
   * (the agent's gates call `hasScope` per entry); this is the AND a caller
   * may still want for its own checks.
   */
  hasScopes(scopes: string[]): boolean {
    return scopes.every(scope => this.hasScope(scope));
  }
  
  /**
   * Create a copy of this context with updated values
   */
  with(updates: Partial<Context>): ContextImpl {
    return new ContextImpl({
      session: updates.session || this.session,
      auth: updates.auth || this.auth,
      payment: updates.payment || this.payment,
      client_capabilities: updates.client_capabilities || this.client_capabilities,
      agent_capabilities: updates.agent_capabilities || this.agent_capabilities,
      metadata: { ...this.metadata, ...updates.metadata },
      signal: updates.signal || this.signal,
    });
  }
  
  /**
   * Update authentication info
   */
  setAuth(auth: AuthInfo): void {
    this.auth = auth;
    this.session.last_activity = Date.now();
  }
  
  /**
   * Update payment info
   */
  setPayment(payment: PaymentInfo): void {
    this.payment = payment;
    this.session.last_activity = Date.now();
  }
  
  /**
   * Update client capabilities
   */
  setClientCapabilities(capabilities: Capabilities): void {
    this.client_capabilities = capabilities;
    this.session.last_activity = Date.now();
  }
}

/**
 * Create a new context
 */
export function createContext(options: Partial<Context> = {}): Context {
  return new ContextImpl(options);
}
