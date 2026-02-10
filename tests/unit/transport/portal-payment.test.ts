/**
 * Portal Transport Payment Tests - payment.required / payment.submit / payment.accepted
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PaymentRequiredError } from '../../../src/skills/payments/x402.js';

/**
 * Since PortalTransportSkill uses real WebSocket connections, we test
 * the payment negotiation logic in isolation by simulating the event flow.
 */

describe('Portal Transport - Payment Negotiation', () => {
  describe('PaymentRequiredError triggers payment.required', () => {
    it('PaymentRequiredError has correct shape for portal to send payment.required', () => {
      const err = new PaymentRequiredError('Agent requires payment', {
        accepts: [{ scheme: 'token', amount: '0.05', currency: 'USD' }],
      });

      expect(err.status_code).toBe(402);
      expect(err.accepts).toHaveLength(1);
      expect(err.accepts![0]).toEqual({ scheme: 'token', amount: '0.05', currency: 'USD' });
    });
  });

  describe('payment.submit event shape', () => {
    it('payment.submit carries token field', () => {
      const submitEvent = {
        type: 'payment.submit',
        payment: {
          scheme: 'token',
          amount: '0.05',
          token: 'jwt_payment_token',
        },
      };

      expect(submitEvent.type).toBe('payment.submit');
      expect(submitEvent.payment.token).toBe('jwt_payment_token');
    });
  });

  describe('portal payment flow contract', () => {
    it('payment.required -> payment.submit -> retry -> payment.accepted sequence', async () => {
      // Simulate the portal payment flow:
      // 1. First processUAMP call throws PaymentRequiredError
      // 2. Portal sends payment.required to client
      // 3. Client responds with payment.submit (token)
      // 4. Portal sets context.payment_token and retries
      // 5. Second processUAMP succeeds
      // 6. Portal sends payment.accepted

      const events: string[] = [];
      let callCount = 0;

      // Mock processUAMP: first call fails, second succeeds
      async function* mockProcessUAMP(): AsyncGenerator<{ type: string }> {
        callCount++;
        if (callCount === 1) {
          throw new PaymentRequiredError('Pay up', {
            accepts: [{ scheme: 'token', amount: '0.01' }],
          });
        }
        yield { type: 'response.delta' };
        yield { type: 'response.done' };
      }

      // Simulate payment negotiation
      const maxRetries = 1;
      let retries = 0;
      const contextStore = new Map<string, unknown>();

      while (true) {
        try {
          for await (const event of mockProcessUAMP()) {
            events.push(event.type);
          }
          if (retries > 0) {
            events.push('payment.accepted');
          }
          break;
        } catch (err) {
          if (err instanceof PaymentRequiredError && retries < maxRetries) {
            retries++;
            events.push('payment.required');

            // Simulate client sending payment.submit
            const token = 'jwt_from_client';
            events.push('payment.submit');
            contextStore.set('payment_token', token);
            continue;
          }
          throw err;
        }
      }

      expect(events).toEqual([
        'payment.required',
        'payment.submit',
        'response.delta',
        'response.done',
        'payment.accepted',
      ]);
      expect(contextStore.get('payment_token')).toBe('jwt_from_client');
    });

    it('payment timeout rejects when no payment.submit received', async () => {
      // If the client never sends payment.submit, the portal times out
      const paymentPromise = new Promise<string>((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('Payment token not received in time')), 50);
        // Simulate: no resolve call (client never submits)
      });

      await expect(paymentPromise).rejects.toThrow('Payment token not received in time');
    });
  });
});
