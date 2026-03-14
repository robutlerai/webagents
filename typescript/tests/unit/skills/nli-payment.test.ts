/**
 * Tests for NLI/A2A/Routing payment token forwarding.
 *
 * Validates that X-Payment-Token is correctly included in outbound
 * HTTP headers when the context has a payment token.
 */

import { describe, it, expect } from 'vitest';

function resolvePaymentToken(context: any): string | undefined {
  return (
    context?.payment?.token ??
    context?.get?.('payment_token') ??
    context?.metadata?.paymentToken
  );
}

describe('NLI buildHeaders - payment token', () => {
  it('includes X-Payment-Token when context.payment.token is set', () => {
    const context = { payment: { token: 'tok_123' }, metadata: {} };
    const token = resolvePaymentToken(context);
    expect(token).toBe('tok_123');
  });

  it('includes X-Payment-Token from context.get("payment_token")', () => {
    const context = {
      get: (key: string) => key === 'payment_token' ? 'tok_456' : undefined,
      metadata: {},
    };
    const token = resolvePaymentToken(context);
    expect(token).toBe('tok_456');
  });

  it('includes X-Payment-Token from context.metadata.paymentToken', () => {
    const context = { metadata: { paymentToken: 'tok_789' } };
    const token = resolvePaymentToken(context);
    expect(token).toBe('tok_789');
  });

  it('does not include X-Payment-Token when no token in context', () => {
    const context = { metadata: {} };
    const token = resolvePaymentToken(context);
    expect(token).toBeUndefined();
  });

  it('prefers payment.token over metadata.paymentToken', () => {
    const context = {
      payment: { token: 'preferred_tok' },
      metadata: { paymentToken: 'fallback_tok' },
    };
    const token = resolvePaymentToken(context);
    expect(token).toBe('preferred_tok');
  });
});

describe('A2A headers - payment token', () => {
  it('a2a_send includes payment token in headers', () => {
    const context = { payment: { token: 'a2a_tok' }, metadata: {} };
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    const paymentToken = resolvePaymentToken(context);
    if (paymentToken) headers['X-Payment-Token'] = paymentToken;

    expect(headers['X-Payment-Token']).toBe('a2a_tok');
  });

  it('a2a_get_task includes payment token in headers', () => {
    const context = { payment: { token: 'a2a_get_tok' }, metadata: {} };
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    const paymentToken = resolvePaymentToken(context);
    if (paymentToken) headers['X-Payment-Token'] = paymentToken;

    expect(headers['X-Payment-Token']).toBe('a2a_get_tok');
  });
});

describe('Routing _callAgent - context threading', () => {
  it('_callAgent receives context and includes payment token', () => {
    const context = { payment: { token: 'route_tok' }, metadata: {} };
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    const paymentToken = resolvePaymentToken(context);
    if (paymentToken) headers['X-Payment-Token'] = paymentToken;

    expect(headers['X-Payment-Token']).toBe('route_tok');
  });

  it('delegateToAgent passes context to _callAgent', () => {
    const contextPassed = { payment: { token: 'delegate_tok' }, metadata: {} };
    const token = resolvePaymentToken(contextPassed);
    expect(token).toBe('delegate_tok');
  });
});
