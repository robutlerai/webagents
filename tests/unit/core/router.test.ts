/**
 * MessageRouter Unit Tests
 * 
 * Tests for the UAMP message router including:
 * - Auto-wiring based on capabilities
 * - Priority-based handler selection
 * - Observer notification
 * - Loop prevention (source, seen, TTL)
 * - System events
 * - Extensibility hooks
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  MessageRouter,
  SystemEvents,
  matchesSubscription,
  type UAMPEvent,
  type Handler,
  type TransportSink,
  type RouterContext,
} from '../../../src/core/router.js';
import { CallbackSink, BufferSink } from '../../../src/core/transport.js';

describe('MessageRouter', () => {
  let router: MessageRouter;

  beforeEach(() => {
    router = new MessageRouter();
  });

  describe('handler registration', () => {
    it('should register a handler', () => {
      const handler: Handler = {
        name: 'test-handler',
        subscribes: ['input.text'],
        produces: ['response.delta'],
        priority: 0,
        process: async function* () {},
      };

      router.registerHandler(handler);
      expect(router.getHandlers().has('test-handler')).toBe(true);
    });

    it('should unregister a handler', () => {
      const handler: Handler = {
        name: 'test-handler',
        subscribes: ['input.text'],
        produces: ['response.delta'],
        priority: 0,
        process: async function* () {},
      };

      router.registerHandler(handler);
      router.unregisterHandler('test-handler');
      expect(router.getHandlers().has('test-handler')).toBe(false);
    });

    it('should set default handler', () => {
      const handler: Handler = {
        name: 'default-handler',
        subscribes: ['input.text'],
        produces: ['response.delta'],
        priority: 0,
        process: async function* () {},
      };

      router.registerHandler(handler);
      router.setDefault('default-handler');
      expect(router.defaultHandler?.name).toBe('default-handler');
    });

    it('should throw when setting non-existent handler as default', () => {
      expect(() => router.setDefault('non-existent')).toThrow('Handler \'non-existent\' not found');
    });
  });

  describe('auto-wiring', () => {
    it('should route to handler by event type', async () => {
      const processed: UAMPEvent[] = [];
      
      const handler: Handler = {
        name: 'text-handler',
        subscribes: ['input.text'],
        produces: ['response.delta'],
        priority: 0,
        process: async function* (event) {
          processed.push(event);
        },
      };

      router.registerHandler(handler);
      
      await router.send({
        id: 'test-1',
        type: 'input.text',
        payload: { text: 'Hello' },
      });

      expect(processed).toHaveLength(1);
      expect(processed[0].payload).toEqual({ text: 'Hello' });
    });

    it('should route to default handler when no match', async () => {
      const processed: UAMPEvent[] = [];
      
      const handler: Handler = {
        name: 'default',
        subscribes: ['input.text'],
        produces: ['response.delta'],
        priority: 0,
        process: async function* (event) {
          processed.push(event);
        },
      };

      router.registerHandler(handler);
      router.setDefault('default');
      
      await router.send({
        id: 'test-1',
        type: 'unknown.type',
        payload: {},
      });

      expect(processed).toHaveLength(1);
    });
  });

  describe('priority-based selection', () => {
    it('should select higher priority handler', async () => {
      const results: string[] = [];

      const lowPriority: Handler = {
        name: 'low',
        subscribes: ['input.text'],
        produces: [],
        priority: 0,
        process: async function* () {
          results.push('low');
        },
      };

      const highPriority: Handler = {
        name: 'high',
        subscribes: ['input.text'],
        produces: [],
        priority: 100,
        process: async function* () {
          results.push('high');
        },
      };

      router.registerHandler(lowPriority);
      router.registerHandler(highPriority);

      await router.send({
        id: 'test-1',
        type: 'input.text',
        payload: {},
      });

      expect(results).toEqual(['high']);
    });
  });

  describe('regex pattern matching', () => {
    it('should match regex patterns', () => {
      expect(matchesSubscription('translate.en', [/^translate\..+$/])).toBe(true);
      expect(matchesSubscription('translate.fr', [/^translate\..+$/])).toBe(true);
      expect(matchesSubscription('input.text', [/^translate\..+$/])).toBe(false);
    });

    it('should match wildcard', () => {
      expect(matchesSubscription('anything', ['*'])).toBe(true);
      expect(matchesSubscription('input.text', ['*'])).toBe(true);
    });

    it('should match exact strings', () => {
      expect(matchesSubscription('input.text', ['input.text'])).toBe(true);
      expect(matchesSubscription('input.audio', ['input.text'])).toBe(false);
    });

    it('should route to regex handler', async () => {
      const processed: UAMPEvent[] = [];
      
      const handler: Handler = {
        name: 'translate',
        subscribes: [/^translate\..+$/],
        produces: [],
        priority: 0,
        process: async function* (event) {
          processed.push(event);
        },
      };

      router.registerHandler(handler);

      await router.send({
        id: 'test-1',
        type: 'translate.en',
        payload: {},
      });

      expect(processed).toHaveLength(1);
    });
  });

  describe('observers', () => {
    it('should notify observers', async () => {
      const observed: UAMPEvent[] = [];
      
      router.registerObserver({
        name: 'logger',
        subscribes: ['input.text'],  // Only observe input.text, not system events
        handler: async (event) => {
          observed.push(event);
        },
      });

      await router.send({
        id: 'test-1',
        type: 'input.text',
        payload: { text: 'Hello' },
      });

      expect(observed).toHaveLength(1);
    });

    it('should not block on observer errors', async () => {
      let handlerCalled = false;
      
      router.registerObserver({
        name: 'failing-observer',
        subscribes: ['*'],
        handler: async () => {
          throw new Error('Observer error');
        },
      });

      router.registerHandler({
        name: 'handler',
        subscribes: ['input.text'],
        produces: [],
        priority: 0,
        process: async function* () {
          handlerCalled = true;
        },
      });

      // Should not throw
      await router.send({
        id: 'test-1',
        type: 'input.text',
        payload: {},
      });

      expect(handlerCalled).toBe(true);
    });

    it('should filter observers by subscription pattern', async () => {
      const observed: string[] = [];
      
      router.registerObserver({
        name: 'text-only',
        subscribes: ['input.text'],
        handler: async () => {
          observed.push('text');
        },
      });

      router.registerObserver({
        name: 'audio-only',
        subscribes: ['input.audio'],
        handler: async () => {
          observed.push('audio');
        },
      });

      await router.send({
        id: 'test-1',
        type: 'input.text',
        payload: {},
      });

      expect(observed).toEqual(['text']);
    });
  });

  describe('loop prevention', () => {
    describe('source tracking', () => {
      it('should not route back to source handler', async () => {
        const callCount = { handler: 0 };

        const handler: Handler = {
          name: 'echo-handler',
          subscribes: ['input.text', 'response.delta'],
          produces: ['response.delta'],
          priority: 0,
          process: async function* (event) {
            callCount.handler++;
            yield {
              id: 'resp-1',
              type: 'response.delta',
              payload: { text: 'Echo' },
            };
          },
        };

        router.registerHandler(handler);

        await router.send({
          id: 'test-1',
          type: 'input.text',
          payload: {},
        });

        // Should only be called once (not infinitely)
        expect(callCount.handler).toBe(1);
      });
    });

    describe('seen set', () => {
      it('should not process message twice by same handler', async () => {
        const callCount = { handler: 0 };

        const handler: Handler = {
          name: 'handler',
          subscribes: ['input.text'],
          produces: [],
          priority: 0,
          process: async function* () {
            callCount.handler++;
          },
        };

        router.registerHandler(handler);

        // Send with handler already in seen set
        await router.send({
          id: 'test-1',
          type: 'input.text',
          payload: {},
          seen: ['handler'],
        });

        expect(callCount.handler).toBe(0);
      });
    });

    describe('TTL', () => {
      it('should emit error when TTL exceeds', async () => {
        let errorReceived = false;

        router.registerObserver({
          name: 'error-observer',
          subscribes: ['system.error'],
          handler: async (event) => {
            errorReceived = true;
            expect(event.payload).toHaveProperty('error', 'TTL exceeded');
          },
        });

        await router.send({
          id: 'test-1',
          type: 'input.text',
          payload: {},
          ttl: 0,
        });

        expect(errorReceived).toBe(true);
      });

      it('should decrement TTL on each hop', async () => {
        const receivedTTL: number[] = [];

        const handler1: Handler = {
          name: 'handler1',
          subscribes: ['input.text'],
          produces: ['intermediate'],
          priority: 0,
          process: async function* (event) {
            receivedTTL.push(event.ttl ?? 10);
            yield {
              id: 'int-1',
              type: 'intermediate',
              payload: {},
            };
          },
        };

        const handler2: Handler = {
          name: 'handler2',
          subscribes: ['intermediate'],
          produces: [],
          priority: 0,
          process: async function* (event) {
            receivedTTL.push(event.ttl ?? 10);
          },
        };

        router.registerHandler(handler1);
        router.registerHandler(handler2);

        await router.send({
          id: 'test-1',
          type: 'input.text',
          payload: {},
          ttl: 5,
        });

        expect(receivedTTL[0]).toBe(5);
        expect(receivedTTL[1]).toBe(4);
      });
    });
  });

  describe('system events', () => {
    it('should handle system.stop', async () => {
      const events: UAMPEvent[] = [];
      const sink = new BufferSink();
      
      router.registerSink(sink);
      router.setActiveSink(sink.id);

      await router.send({
        id: 'test-1',
        type: SystemEvents.STOP,
        payload: {},
      });

      expect(sink.getEvents()).toHaveLength(1);
      expect(sink.getEvents()[0].type).toBe(SystemEvents.STOP);
    });

    it('should respond to system.ping with pong', async () => {
      const sink = new BufferSink();
      router.registerSink(sink);
      router.setActiveSink(sink.id);

      await router.send({
        id: 'test-1',
        type: SystemEvents.PING,
        payload: {},
      });

      expect(sink.getEvents()).toHaveLength(1);
      expect(sink.getEvents()[0].type).toBe(SystemEvents.PONG);
    });
  });

  describe('transport sinks', () => {
    it('should register and set active sink', () => {
      const sink = new BufferSink({ id: 'test-sink' });
      
      router.registerSink(sink);
      router.setActiveSink('test-sink');

      expect(router.activeSink).toBe(sink);
    });

    it('should throw when setting non-existent sink as active', () => {
      expect(() => router.setActiveSink('non-existent')).toThrow('Sink \'non-existent\' not registered');
    });

    it('should unregister sink', () => {
      const sink = new BufferSink({ id: 'test-sink' });
      
      router.registerSink(sink);
      router.unregisterSink('test-sink');

      expect(() => router.setActiveSink('test-sink')).toThrow();
    });

    it('should register default sink with wildcard', async () => {
      const sink = new BufferSink({ id: 'default-sink' });
      
      router.registerDefaultSink(sink);
      router.setActiveSink('default-sink');

      // Unhandled event should go to default sink
      await router.send({
        id: 'test-1',
        type: 'unhandled.type',
        payload: { data: 'test' },
      });

      expect(sink.getEvents()).toHaveLength(1);
    });
  });

  describe('extensibility hooks', () => {
    it('should call onUnroutable for unhandled events', async () => {
      let unroutableCalled = false;
      let unroutableEvent: UAMPEvent | null = null;

      router.onUnroutable(async (event) => {
        unroutableCalled = true;
        unroutableEvent = event;
      });

      await router.send({
        id: 'test-1',
        type: 'unhandled.type',
        payload: {},
      });

      expect(unroutableCalled).toBe(true);
      expect(unroutableEvent?.type).toBe('unhandled.type');
    });

    it('should call onError when handler throws', async () => {
      let errorCalled = false;
      let caughtError: Error | null = null;

      router.onError(async (error) => {
        errorCalled = true;
        caughtError = error;
      });

      router.registerHandler({
        name: 'failing-handler',
        subscribes: ['input.text'],
        produces: [],
        priority: 0,
        process: async function* () {
          throw new Error('Handler error');
        },
      });

      await router.send({
        id: 'test-1',
        type: 'input.text',
        payload: {},
      });

      expect(errorCalled).toBe(true);
      expect(caughtError?.message).toBe('Handler error');
    });

    it('should allow beforeRoute to modify events', async () => {
      const processed: UAMPEvent[] = [];

      router.beforeRoute(async (event) => {
        return {
          ...event,
          payload: { ...event.payload, modified: true },
        };
      });

      router.registerHandler({
        name: 'handler',
        subscribes: ['input.text'],
        produces: [],
        priority: 0,
        process: async function* (event) {
          processed.push(event);
        },
      });

      await router.send({
        id: 'test-1',
        type: 'input.text',
        payload: { original: true },
      });

      expect(processed[0].payload).toEqual({ original: true, modified: true });
    });

    it('should allow beforeRoute to block events', async () => {
      let handlerCalled = false;

      router.beforeRoute(async () => null); // Block all

      router.registerHandler({
        name: 'handler',
        subscribes: ['input.text'],
        produces: [],
        priority: 0,
        process: async function* () {
          handlerCalled = true;
        },
      });

      await router.send({
        id: 'test-1',
        type: 'input.text',
        payload: {},
      });

      expect(handlerCalled).toBe(false);
    });

    it('should call afterRoute after processing', async () => {
      let afterRouteCalled = false;
      let handlerName: string | undefined;

      router.afterRoute(async (event, handler) => {
        afterRouteCalled = true;
        handlerName = handler?.name;
        return event;
      });

      router.registerHandler({
        name: 'test-handler',
        subscribes: ['input.text'],
        produces: [],
        priority: 0,
        process: async function* () {},
      });

      await router.send({
        id: 'test-1',
        type: 'input.text',
        payload: {},
      });

      expect(afterRouteCalled).toBe(true);
      expect(handlerName).toBe('test-handler');
    });
  });

  describe('explicit routing', () => {
    it('should allow explicit route override', async () => {
      const results: string[] = [];

      router.registerHandler({
        name: 'default-handler',
        subscribes: ['custom.event'],
        produces: [],
        priority: 50,
        process: async function* () {
          results.push('default');
        },
      });

      router.registerHandler({
        name: 'special-handler',
        subscribes: ['other.event'],
        produces: [],
        priority: 0,
        process: async function* () {
          results.push('special');
        },
      });

      // Override routing
      router.route('custom.event', 'special-handler', 100);

      await router.send({
        id: 'test-1',
        type: 'custom.event',
        payload: {},
      });

      expect(results).toEqual(['special']);
    });
  });
});

describe('Transport Sinks', () => {
  describe('CallbackSink', () => {
    it('should call callback on send', async () => {
      const events: any[] = [];
      const sink = new CallbackSink((event) => events.push(event));

      await sink.send({ type: 'test', payload: {} });

      expect(events).toHaveLength(1);
    });

    it('should not send when closed', async () => {
      const events: any[] = [];
      const sink = new CallbackSink((event) => events.push(event));

      sink.close();
      await sink.send({ type: 'test', payload: {} });

      expect(events).toHaveLength(0);
    });
  });

  describe('BufferSink', () => {
    it('should buffer events', async () => {
      const sink = new BufferSink();

      await sink.send({ type: 'test1', payload: {} });
      await sink.send({ type: 'test2', payload: {} });

      expect(sink.getEvents()).toHaveLength(2);
      expect(sink.length).toBe(2);
    });

    it('should clear buffer', async () => {
      const sink = new BufferSink();

      await sink.send({ type: 'test', payload: {} });
      sink.clear();

      expect(sink.getEvents()).toHaveLength(0);
    });

    it('should respect maxSize', async () => {
      const sink = new BufferSink({ maxSize: 2 });

      await sink.send({ type: 'test1', payload: {} });
      await sink.send({ type: 'test2', payload: {} });
      await sink.send({ type: 'test3', payload: {} });

      expect(sink.length).toBe(2);
      expect(sink.getEvents()[0].type).toBe('test2');
    });
  });
});
