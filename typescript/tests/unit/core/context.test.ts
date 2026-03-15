/**
 * Context Unit Tests
 */

import { describe, it, expect } from 'vitest';
import {
  ContextImpl,
  createContext,
  createSessionState,
  createDefaultAuthInfo,
  createDefaultPaymentInfo,
} from '../../../src/core/context.js';

describe('Context', () => {
  describe('createSessionState', () => {
    it('creates session with unique ID', () => {
      const session1 = createSessionState();
      const session2 = createSessionState();
      
      expect(session1.id).toBeDefined();
      expect(session2.id).toBeDefined();
      expect(session1.id).not.toBe(session2.id);
    });
    
    it('accepts custom ID', () => {
      const session = createSessionState('custom-123');
      expect(session.id).toBe('custom-123');
    });
    
    it('has timestamps', () => {
      const before = Date.now();
      const session = createSessionState();
      const after = Date.now();
      
      expect(session.created_at).toBeGreaterThanOrEqual(before);
      expect(session.created_at).toBeLessThanOrEqual(after + 1);
      expect(session.last_activity).toBeGreaterThanOrEqual(session.created_at);
      expect(session.last_activity).toBeLessThanOrEqual(session.created_at + 1);
    });
  });
  
  describe('createDefaultAuthInfo', () => {
    it('creates unauthenticated state', () => {
      const auth = createDefaultAuthInfo();
      expect(auth.authenticated).toBe(false);
    });
  });
  
  describe('createDefaultPaymentInfo', () => {
    it('creates invalid payment state', () => {
      const payment = createDefaultPaymentInfo();
      expect(payment.valid).toBe(false);
    });
  });
  
  describe('createContext', () => {
    it('creates context with defaults', () => {
      const ctx = createContext();
      
      expect(ctx.session).toBeDefined();
      expect(ctx.auth).toBeDefined();
      expect(ctx.payment).toBeDefined();
      expect(ctx.metadata).toEqual({});
    });
    
    it('accepts custom values', () => {
      const ctx = createContext({
        auth: { authenticated: true, user_id: 'user-123' },
        metadata: { custom: 'value' },
      });
      
      expect(ctx.auth.authenticated).toBe(true);
      expect(ctx.auth.user_id).toBe('user-123');
      expect(ctx.metadata.custom).toBe('value');
    });
  });
  
  describe('ContextImpl', () => {
    describe('get/set/delete', () => {
      it('stores and retrieves session data', () => {
        const ctx = new ContextImpl();
        
        ctx.set('key', 'value');
        expect(ctx.get('key')).toBe('value');
        
        ctx.delete('key');
        expect(ctx.get('key')).toBeUndefined();
      });
      
      it('supports typed get', () => {
        const ctx = new ContextImpl();
        
        ctx.set('count', 42);
        const count = ctx.get<number>('count');
        expect(count).toBe(42);
      });
      
      it('updates last_activity on set', () => {
        const ctx = new ContextImpl();
        const initialActivity = ctx.session.last_activity;
        
        // Small delay
        return new Promise<void>(resolve => {
          setTimeout(() => {
            ctx.set('key', 'value');
            expect(ctx.session.last_activity).toBeGreaterThanOrEqual(initialActivity);
            resolve();
          }, 1);
        });
      });
    });
    
    describe('hasScope/hasScopes', () => {
      it('returns false when not authenticated', () => {
        const ctx = new ContextImpl();
        expect(ctx.hasScope('read')).toBe(false);
      });
      
      it('returns false when no scopes', () => {
        const ctx = new ContextImpl({
          auth: { authenticated: true },
        });
        expect(ctx.hasScope('read')).toBe(false);
      });
      
      it('checks scope membership', () => {
        const ctx = new ContextImpl({
          auth: {
            authenticated: true,
            scopes: ['read', 'write'],
          },
        });
        
        expect(ctx.hasScope('read')).toBe(true);
        expect(ctx.hasScope('write')).toBe(true);
        expect(ctx.hasScope('admin')).toBe(false);
      });
      
      it('checks multiple scopes', () => {
        const ctx = new ContextImpl({
          auth: {
            authenticated: true,
            scopes: ['read', 'write'],
          },
        });
        
        expect(ctx.hasScopes(['read', 'write'])).toBe(true);
        expect(ctx.hasScopes(['read', 'admin'])).toBe(false);
      });
    });
    
    describe('with', () => {
      it('creates copy with updates', () => {
        const ctx = new ContextImpl({
          metadata: { original: true },
        });
        
        const newCtx = ctx.with({
          metadata: { updated: true },
        });
        
        expect(ctx.metadata.original).toBe(true);
        expect(newCtx.metadata.updated).toBe(true);
        expect(newCtx.metadata.original).toBe(true);
      });
    });
    
    describe('setAuth', () => {
      it('updates authentication info', () => {
        const ctx = new ContextImpl();
        expect(ctx.auth.authenticated).toBe(false);
        
        ctx.setAuth({ authenticated: true, user_id: 'user-1' });
        expect(ctx.auth.authenticated).toBe(true);
        expect(ctx.auth.user_id).toBe('user-1');
      });
    });
    
    describe('setPayment', () => {
      it('updates payment info', () => {
        const ctx = new ContextImpl();
        expect(ctx.payment.valid).toBe(false);
        
        ctx.setPayment({ valid: true, balance: 100 });
        expect(ctx.payment.valid).toBe(true);
        expect(ctx.payment.balance).toBe(100);
      });
    });
    
    describe('setClientCapabilities', () => {
      it('updates client capabilities', () => {
        const ctx = new ContextImpl();
        expect(ctx.client_capabilities).toBeUndefined();
        
        ctx.setClientCapabilities({
          id: 'client',
          provider: 'test',
          modalities: ['text'],
          supports_streaming: true,
          supports_thinking: false,
          supports_caching: false,
        });
        
        expect(ctx.client_capabilities?.id).toBe('client');
      });
    });
  });
});
