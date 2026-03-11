/**
 * PaymentX402Skill Unit Tests - Transport-Agnostic Payment
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PaymentX402Skill, PaymentRequiredError } from '../../../../src/skills/payments/x402.js';

describe('PaymentRequiredError', () => {
  it('has status_code 402', () => {
    const err = new PaymentRequiredError();
    expect(err.status_code).toBe(402);
    expect(err.name).toBe('PaymentRequiredError');
  });

  it('carries accepts array in context', () => {
    const err = new PaymentRequiredError('Payment required', {
      accepts: [{ scheme: 'token', amount: '0.01' }],
    });
    expect(err.accepts).toHaveLength(1);
    expect(err.context.accepts).toEqual([{ scheme: 'token', amount: '0.01' }]);
  });

  it('carries maxAmountRequired in context', () => {
    const err = new PaymentRequiredError('Pay up', { maxAmountRequired: 5 });
    expect(err.context.maxAmountRequired).toBe(5);
  });

  it('instanceof Error', () => {
    const err = new PaymentRequiredError();
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(PaymentRequiredError);
  });
});

describe('PaymentSkill x402 methods', () => {
  it('createX402Requirements returns correct structure', async () => {
    const { PaymentSkill } = await import('../../../../src/skills/payments/skill.js');
    const skill = new PaymentSkill({
      enableBilling: true,
      agentId: 'test-agent',
      acceptedSchemes: [{ scheme: 'token', network: 'robutler' }],
    });

    const reqs = skill.createX402Requirements(0.05, '/api/search');
    expect(reqs.version).toBe('1.0');
    expect(reqs.accepts).toHaveLength(1);
    expect(reqs.accepts[0].scheme).toBe('token');
    expect(reqs.accepts[0].network).toBe('robutler');
    expect(reqs.accepts[0].maxAmountRequired).toBe('0.05');
    expect(reqs.accepts[0].payTo).toBe('test-agent');
    expect(reqs.accepts[0].resource).toBe('/api/search');
  });

  it('verifyX402Payment rejects amounts over maxPayment', async () => {
    const { PaymentSkill } = await import('../../../../src/skills/payments/skill.js');
    const skill = new PaymentSkill({
      enableBilling: true,
      maxPayment: 1.0,
    });

    const result = await skill.verifyX402Payment('fake-token', 5.0, '/api/test');
    expect(result.valid).toBe(false);
    expect(result.error).toContain('exceeds max payment limit');
  });
});

describe('PaymentX402Skill', () => {
  let skill: PaymentX402Skill;

  beforeEach(() => {
    skill = new PaymentX402Skill({ facilitatorUrl: 'https://test.robutler.ai' });
  });

  describe('constructor', () => {
    it('strips trailing slash from facilitatorUrl', () => {
      const s = new PaymentX402Skill({ facilitatorUrl: 'https://robutler.ai/' });
      // Access private for testing
      expect((s as any).facilitatorUrl).toBe('https://robutler.ai');
    });

    it('defaults to https://robutler.ai', () => {
      const s = new PaymentX402Skill();
      expect((s as any).facilitatorUrl).toBe('https://robutler.ai');
    });
  });

  describe('checkPayment hook', () => {
    it('reads payment_token from context first (transport-agnostic)', async () => {
      const context = {
        get: vi.fn((key: string) => {
          if (key === 'payment_token') return 'tok_from_context';
          if (key === 'payment_required') return false;
          return undefined;
        }),
        set: vi.fn(),
        metadata: {},
      };

      // Mock verifyPaymentToken to succeed
      vi.spyOn(skill, 'verifyPaymentToken').mockResolvedValue({ valid: true, balance: 10 });

      await skill.checkPayment({} as any, context as any);

      // Should have called verifyPaymentToken with context token
      expect(skill.verifyPaymentToken).toHaveBeenCalledWith('tok_from_context');
    });

    it('falls back to X-PAYMENT metadata header', async () => {
      const context = {
        get: vi.fn((key: string) => {
          if (key === 'payment_token') return undefined;
          if (key === 'payment_required') return false;
          return undefined;
        }),
        set: vi.fn(),
        metadata: { 'X-PAYMENT': 'tok_from_header' },
      };

      vi.spyOn(skill, 'verifyPaymentToken').mockResolvedValue({ valid: true, balance: 5 });

      await skill.checkPayment({} as any, context as any);
      expect(skill.verifyPaymentToken).toHaveBeenCalledWith('tok_from_header');
    });

    it('throws PaymentRequiredError when payment_required flag is set and no token', async () => {
      const context = {
        get: vi.fn((key: string) => {
          if (key === 'payment_token') return undefined;
          if (key === 'payment_required') return true;
          if (key === 'payment_accepts') return [{ scheme: 'token' }];
          if (key === 'payment_max_amount_required') return 1;
          return undefined;
        }),
        set: vi.fn(),
        metadata: {},
      };

      await expect(skill.checkPayment({} as any, context as any)).rejects.toThrow(
        PaymentRequiredError
      );
    });

    it('does nothing when no token and payment not required', async () => {
      const context = {
        get: vi.fn(() => undefined),
        set: vi.fn(),
        metadata: {},
      };

      const result = await skill.checkPayment({} as any, context as any);
      expect(result).toBeUndefined();
    });
  });
});
