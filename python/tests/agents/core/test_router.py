"""
MessageRouter Unit Tests

Tests for the UAMP message router including:
- Auto-wiring based on capabilities
- Priority-based handler selection
- Observer notification
- Loop prevention (source, seen, TTL)
- System events
- Extensibility hooks
"""

import re
import pytest
import asyncio
from typing import List, Dict, Any

from webagents.agents.core.router import (
    MessageRouter,
    UAMPEvent,
    RouterContext,
    Handler,
    Observer,
    SystemEvents,
    CallbackSink,
    BufferSink,
    matches_subscription,
    matches_scope,
)


class TestMatchesSubscription:
    """Tests for the matches_subscription utility function."""
    
    def test_exact_string_match(self):
        assert matches_subscription('input.text', ['input.text']) is True
        assert matches_subscription('input.audio', ['input.text']) is False
    
    def test_wildcard_match(self):
        assert matches_subscription('anything', ['*']) is True
        assert matches_subscription('input.text', ['*']) is True
    
    def test_regex_match(self):
        pattern = re.compile(r'^translate\..+$')
        assert matches_subscription('translate.en', [pattern]) is True
        assert matches_subscription('translate.fr', [pattern]) is True
        assert matches_subscription('input.text', [pattern]) is False
    
    def test_multiple_patterns(self):
        assert matches_subscription('input.text', ['input.audio', 'input.text']) is True
        assert matches_subscription('input.image', ['input.audio', 'input.text']) is False


class TestMessageRouter:
    """Tests for the MessageRouter class."""
    
    @pytest.fixture
    def router(self):
        return MessageRouter()
    
    @pytest.fixture
    def simple_handler(self):
        async def process(event, context):
            yield UAMPEvent(
                id='resp-1',
                type='response.delta',
                payload={'text': 'Hello'}
            )
        
        return Handler(
            name='simple-handler',
            subscribes=['input.text'],
            produces=['response.delta'],
            priority=0,
            process=process
        )


class TestHandlerRegistration(TestMessageRouter):
    """Tests for handler registration."""
    
    @pytest.mark.asyncio
    async def test_register_handler(self, router, simple_handler):
        router.register_handler(simple_handler)
        assert 'simple-handler' in router.get_handlers()
    
    @pytest.mark.asyncio
    async def test_unregister_handler(self, router, simple_handler):
        router.register_handler(simple_handler)
        router.unregister_handler('simple-handler')
        assert 'simple-handler' not in router.get_handlers()
    
    @pytest.mark.asyncio
    async def test_set_default_handler(self, router, simple_handler):
        router.register_handler(simple_handler)
        router.set_default('simple-handler')
        assert router.default_handler.name == 'simple-handler'
    
    @pytest.mark.asyncio
    async def test_set_nonexistent_default_raises(self, router):
        with pytest.raises(ValueError, match="Handler 'nonexistent' not found"):
            router.set_default('nonexistent')


class TestAutoWiring(TestMessageRouter):
    """Tests for auto-wiring based on capabilities."""
    
    @pytest.mark.asyncio
    async def test_route_to_handler_by_event_type(self, router):
        processed = []
        
        async def process(event, context):
            processed.append(event)
            return
            yield  # Make it a generator
        
        handler = Handler(
            name='text-handler',
            subscribes=['input.text'],
            produces=['response.delta'],
            priority=0,
            process=process
        )
        
        router.register_handler(handler)
        
        await router.send(UAMPEvent(
            id='test-1',
            type='input.text',
            payload={'text': 'Hello'}
        ))
        
        assert len(processed) == 1
        assert processed[0].payload == {'text': 'Hello'}
    
    @pytest.mark.asyncio
    async def test_route_to_default_handler_when_no_match(self, router, simple_handler):
        processed = []
        
        async def process(event, context):
            processed.append(event)
            return
            yield
        
        handler = Handler(
            name='default',
            subscribes=['input.text'],
            produces=[],
            priority=0,
            process=process
        )
        
        router.register_handler(handler)
        router.set_default('default')
        
        await router.send(UAMPEvent(
            id='test-1',
            type='unknown.type',
            payload={}
        ))
        
        assert len(processed) == 1


class TestPriorityBasedSelection(TestMessageRouter):
    """Tests for priority-based handler selection."""
    
    @pytest.mark.asyncio
    async def test_select_higher_priority_handler(self, router):
        results = []
        
        async def low_process(event, context):
            results.append('low')
            return
            yield
        
        async def high_process(event, context):
            results.append('high')
            return
            yield
        
        low = Handler(
            name='low',
            subscribes=['input.text'],
            produces=[],
            priority=0,
            process=low_process
        )
        
        high = Handler(
            name='high',
            subscribes=['input.text'],
            produces=[],
            priority=100,  # Higher priority
            process=high_process
        )
        
        router.register_handler(low)
        router.register_handler(high)
        
        await router.send(UAMPEvent(
            id='test-1',
            type='input.text',
            payload={}
        ))
        
        assert results == ['high']


class TestRegexPatternRouting(TestMessageRouter):
    """Tests for regex pattern matching in routing."""
    
    @pytest.mark.asyncio
    async def test_route_to_regex_handler(self, router):
        processed = []
        
        async def process(event, context):
            processed.append(event)
            return
            yield
        
        handler = Handler(
            name='translator',
            subscribes=[re.compile(r'^translate\..+$')],
            produces=[],
            priority=0,
            process=process
        )
        
        router.register_handler(handler)
        
        await router.send(UAMPEvent(
            id='test-1',
            type='translate.en',
            payload={}
        ))
        
        assert len(processed) == 1


class TestObservers(TestMessageRouter):
    """Tests for observer notification."""
    
    @pytest.mark.asyncio
    async def test_notify_observers(self, router):
        observed = []
        
        async def observe(event, context):
            observed.append(event)
        
        router.register_observer(Observer(
            name='logger',
            subscribes=['input.text'],  # Only observe input.text, not system events
            handler=observe
        ))
        
        await router.send(UAMPEvent(
            id='test-1',
            type='input.text',
            payload={'text': 'Hello'}
        ))
        
        assert len(observed) == 1
    
    @pytest.mark.asyncio
    async def test_observer_errors_dont_block(self, router):
        handler_called = False
        
        async def failing_observe(event, context):
            raise RuntimeError("Observer error")
        
        async def process(event, context):
            nonlocal handler_called
            handler_called = True
            return
            yield
        
        router.register_observer(Observer(
            name='failing',
            subscribes=['*'],
            handler=failing_observe
        ))
        
        router.register_handler(Handler(
            name='handler',
            subscribes=['input.text'],
            produces=[],
            priority=0,
            process=process
        ))
        
        # Should not raise
        await router.send(UAMPEvent(
            id='test-1',
            type='input.text',
            payload={}
        ))
        
        assert handler_called is True
    
    @pytest.mark.asyncio
    async def test_filter_observers_by_subscription(self, router):
        observed = []
        
        async def text_observe(event, context):
            observed.append('text')
        
        async def audio_observe(event, context):
            observed.append('audio')
        
        router.register_observer(Observer(
            name='text-only',
            subscribes=['input.text'],
            handler=text_observe
        ))
        
        router.register_observer(Observer(
            name='audio-only',
            subscribes=['input.audio'],
            handler=audio_observe
        ))
        
        await router.send(UAMPEvent(
            id='test-1',
            type='input.text',
            payload={}
        ))
        
        assert observed == ['text']


class TestLoopPrevention(TestMessageRouter):
    """Tests for loop prevention mechanisms."""
    
    @pytest.mark.asyncio
    async def test_source_tracking_prevents_loop(self, router):
        call_count = 0
        
        async def process(event, context):
            nonlocal call_count
            call_count += 1
            yield UAMPEvent(
                id='resp-1',
                type='input.text',  # Same as input
                payload={}
            )
        
        router.register_handler(Handler(
            name='echo-handler',
            subscribes=['input.text'],
            produces=['input.text'],
            priority=0,
            process=process
        ))
        
        await router.send(UAMPEvent(
            id='test-1',
            type='input.text',
            payload={}
        ))
        
        # Should only be called once (not infinitely)
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_seen_set_prevents_duplicate_processing(self, router):
        call_count = 0
        
        async def process(event, context):
            nonlocal call_count
            call_count += 1
            return
            yield
        
        router.register_handler(Handler(
            name='handler',
            subscribes=['input.text'],
            produces=[],
            priority=0,
            process=process
        ))
        
        # Send with handler already in seen set
        await router.send(UAMPEvent(
            id='test-1',
            type='input.text',
            payload={},
            seen={'handler'}
        ))
        
        assert call_count == 0
    
    @pytest.mark.asyncio
    async def test_ttl_expiry_emits_error(self, router):
        error_received = False
        
        async def observe(event, context):
            nonlocal error_received
            if event.type == SystemEvents.ERROR:
                error_received = True
        
        router.register_observer(Observer(
            name='error-observer',
            subscribes=['system.error'],
            handler=observe
        ))
        
        await router.send(UAMPEvent(
            id='test-1',
            type='input.text',
            payload={},
            ttl=0
        ))
        
        assert error_received is True
    
    @pytest.mark.asyncio
    async def test_ttl_decrements_on_each_hop(self, router):
        received_ttls = []
        
        async def process1(event, context):
            received_ttls.append(event.ttl)
            yield UAMPEvent(
                id='int-1',
                type='intermediate',
                payload={}
            )
        
        async def process2(event, context):
            received_ttls.append(event.ttl)
            return
            yield
        
        router.register_handler(Handler(
            name='handler1',
            subscribes=['input.text'],
            produces=['intermediate'],
            priority=0,
            process=process1
        ))
        
        router.register_handler(Handler(
            name='handler2',
            subscribes=['intermediate'],
            produces=[],
            priority=0,
            process=process2
        ))
        
        await router.send(UAMPEvent(
            id='test-1',
            type='input.text',
            payload={},
            ttl=5
        ))
        
        assert received_ttls[0] == 5
        assert received_ttls[1] == 4


class TestSystemEvents(TestMessageRouter):
    """Tests for system event handling."""
    
    @pytest.mark.asyncio
    async def test_handle_stop_event(self, router):
        sink = BufferSink()
        router.register_sink(sink)
        router.set_active_sink(sink.id)
        
        await router.send(UAMPEvent(
            id='test-1',
            type=SystemEvents.STOP,
            payload={}
        ))
        
        assert len(sink) == 1
        assert sink.get_events()[0]['type'] == SystemEvents.STOP
    
    @pytest.mark.asyncio
    async def test_ping_responds_with_pong(self, router):
        sink = BufferSink()
        router.register_sink(sink)
        router.set_active_sink(sink.id)
        
        await router.send(UAMPEvent(
            id='test-1',
            type=SystemEvents.PING,
            payload={}
        ))
        
        assert len(sink) == 1
        assert sink.get_events()[0]['type'] == SystemEvents.PONG


class TestTransportSinks(TestMessageRouter):
    """Tests for transport sinks."""
    
    @pytest.mark.asyncio
    async def test_register_and_set_active_sink(self, router):
        sink = BufferSink(sink_id='test-sink')
        
        router.register_sink(sink)
        router.set_active_sink('test-sink')
        
        assert router.active_sink is sink
    
    @pytest.mark.asyncio
    async def test_set_nonexistent_sink_raises(self, router):
        with pytest.raises(ValueError, match="Sink 'nonexistent' not registered"):
            router.set_active_sink('nonexistent')
    
    @pytest.mark.asyncio
    async def test_unregister_sink(self, router):
        sink = BufferSink(sink_id='test-sink')
        
        router.register_sink(sink)
        router.unregister_sink('test-sink')
        
        with pytest.raises(ValueError):
            router.set_active_sink('test-sink')
    
    @pytest.mark.asyncio
    async def test_default_sink_catches_unhandled(self, router):
        sink = BufferSink(sink_id='default-sink')
        
        router.register_default_sink(sink)
        router.set_active_sink('default-sink')
        
        await router.send(UAMPEvent(
            id='test-1',
            type='unhandled.type',
            payload={'data': 'test'}
        ))
        
        assert len(sink) == 1


class TestExtensibilityHooks(TestMessageRouter):
    """Tests for extensibility hooks."""
    
    @pytest.mark.asyncio
    async def test_on_unroutable_called(self, router):
        unroutable_called = False
        unroutable_event = None
        
        async def handle(event, context):
            nonlocal unroutable_called, unroutable_event
            unroutable_called = True
            unroutable_event = event
        
        router.on_unroutable(handle)
        
        await router.send(UAMPEvent(
            id='test-1',
            type='unhandled.type',
            payload={}
        ))
        
        assert unroutable_called is True
        assert unroutable_event.type == 'unhandled.type'
    
    @pytest.mark.asyncio
    async def test_on_error_called(self, router):
        error_called = False
        caught_error = None
        
        async def handle(error, event, handler, context):
            nonlocal error_called, caught_error
            error_called = True
            caught_error = error
        
        async def failing_process(event, context):
            raise RuntimeError("Handler error")
            yield
        
        router.on_error(handle)
        
        router.register_handler(Handler(
            name='failing',
            subscribes=['input.text'],
            produces=[],
            priority=0,
            process=failing_process
        ))
        
        await router.send(UAMPEvent(
            id='test-1',
            type='input.text',
            payload={}
        ))
        
        assert error_called is True
        assert str(caught_error) == "Handler error"
    
    @pytest.mark.asyncio
    async def test_before_route_can_block(self, router):
        handler_called = False
        
        async def intercept(event, handler, context):
            return None  # Block
        
        async def process(event, context):
            nonlocal handler_called
            handler_called = True
            return
            yield
        
        router.before_route(intercept)
        
        router.register_handler(Handler(
            name='handler',
            subscribes=['input.text'],
            produces=[],
            priority=0,
            process=process
        ))
        
        await router.send(UAMPEvent(
            id='test-1',
            type='input.text',
            payload={}
        ))
        
        assert handler_called is False
    
    @pytest.mark.asyncio
    async def test_after_route_called(self, router):
        after_route_called = False
        handler_name = None
        
        async def intercept(event, handler, context):
            nonlocal after_route_called, handler_name
            after_route_called = True
            handler_name = handler.name if handler else None
            return event
        
        async def process(event, context):
            return
            yield
        
        router.after_route(intercept)
        
        router.register_handler(Handler(
            name='test-handler',
            subscribes=['input.text'],
            produces=[],
            priority=0,
            process=process
        ))
        
        await router.send(UAMPEvent(
            id='test-1',
            type='input.text',
            payload={}
        ))
        
        assert after_route_called is True
        assert handler_name == 'test-handler'


class TestExplicitRouting(TestMessageRouter):
    """Tests for explicit route override."""
    
    @pytest.mark.asyncio
    async def test_explicit_route_override(self, router):
        results = []
        
        async def default_process(event, context):
            results.append('default')
            return
            yield
        
        async def special_process(event, context):
            results.append('special')
            return
            yield
        
        router.register_handler(Handler(
            name='default-handler',
            subscribes=['custom.event'],
            produces=[],
            priority=50,
            process=default_process
        ))
        
        router.register_handler(Handler(
            name='special-handler',
            subscribes=['other.event'],
            produces=[],
            priority=0,
            process=special_process
        ))
        
        # Override routing
        router.route('custom.event', 'special-handler', 100)
        
        await router.send(UAMPEvent(
            id='test-1',
            type='custom.event',
            payload={}
        ))
        
        assert results == ['special']


class TestTransportSinkImplementations:
    """Tests for transport sink implementations."""
    
    @pytest.mark.asyncio
    async def test_callback_sink(self):
        events = []
        
        async def callback(event):
            events.append(event)
        
        sink = CallbackSink(callback)
        
        await sink.send({'type': 'test', 'payload': {}})
        
        assert len(events) == 1
    
    @pytest.mark.asyncio
    async def test_callback_sink_not_active_after_close(self):
        events = []
        
        async def callback(event):
            events.append(event)
        
        sink = CallbackSink(callback)
        sink.close()
        
        await sink.send({'type': 'test', 'payload': {}})
        
        assert len(events) == 0
    
    @pytest.mark.asyncio
    async def test_buffer_sink(self):
        sink = BufferSink()
        
        await sink.send({'type': 'test1', 'payload': {}})
        await sink.send({'type': 'test2', 'payload': {}})
        
        assert len(sink) == 2
        assert len(sink.get_events()) == 2
    
    @pytest.mark.asyncio
    async def test_buffer_sink_clear(self):
        sink = BufferSink()
        
        await sink.send({'type': 'test', 'payload': {}})
        sink.clear()
        
        assert len(sink) == 0
    
    @pytest.mark.asyncio
    async def test_buffer_sink_max_size(self):
        sink = BufferSink(max_size=2)
        
        await sink.send({'type': 'test1', 'payload': {}})
        await sink.send({'type': 'test2', 'payload': {}})
        await sink.send({'type': 'test3', 'payload': {}})
        
        assert len(sink) == 2
        assert sink.get_events()[0]['type'] == 'test2'


class TestBackwardCompatibility:
    """Tests for backward compatibility."""
    
    def test_handoff_decorator_defaults(self):
        """Test that @handoff decorator has correct defaults."""
        from webagents.agents.tools.decorators import handoff
        
        @handoff(name='test-handler')
        async def test_func(self, messages, **kwargs):
            pass
        
        assert test_func._handoff_subscribes == ['input.text']
        assert test_func._handoff_produces == ['response.delta']
    
    def test_handoff_decorator_with_custom_values(self):
        """Test that @handoff decorator accepts custom values."""
        from webagents.agents.tools.decorators import handoff
        
        @handoff(
            name='audio-handler',
            subscribes=['input.audio'],
            produces=['input.text']
        )
        async def test_func(self, messages, **kwargs):
            pass
        
        assert test_func._handoff_subscribes == ['input.audio']
        assert test_func._handoff_produces == ['input.text']
    
    def test_observe_decorator(self):
        """Test that @observe decorator works."""
        from webagents.agents.tools.decorators import observe
        
        @observe(subscribes=['*'], name='test-observer')
        async def test_func(self, event, context=None):
            pass
        
        assert test_func._webagents_is_observer is True
        assert test_func._observer_name == 'test-observer'
        assert test_func._observer_subscribes == ['*']


class TestMatchesScope:
    """Tests for the matches_scope utility function."""
    
    def test_all_scope_on_handler_allows_access(self):
        """Handler with 'all' scope is accessible to everyone."""
        assert matches_scope(['user'], ['all']) is True
        assert matches_scope(['admin'], ['all']) is True
        assert matches_scope(['custom'], ['all']) is True
    
    def test_all_scope_on_request_allows_access(self):
        """Request with 'all' scope (superuser) can access anything."""
        assert matches_scope(['all'], ['admin']) is True
        assert matches_scope(['all'], ['owner']) is True
        assert matches_scope(['all'], ['restricted']) is True
    
    def test_empty_scopes_treated_as_all(self):
        """Empty or None scopes should be treated as 'all'."""
        assert matches_scope([], ['admin']) is True
        assert matches_scope(['admin'], []) is True
        assert matches_scope([], []) is True
    
    def test_empty_string_scope_treated_as_all(self):
        """Empty string '' scope should be treated as 'all'."""
        assert matches_scope([''], ['admin']) is True
        assert matches_scope(['admin'], ['']) is True
        assert matches_scope([''], ['']) is True
    
    def test_exact_scope_match(self):
        """Exact string scope matching."""
        assert matches_scope(['admin'], ['admin']) is True
        assert matches_scope(['user'], ['admin']) is False
        assert matches_scope(['admin', 'user'], ['admin']) is True
    
    def test_multiple_scopes(self):
        """Multiple scopes - any match is allowed."""
        assert matches_scope(['user', 'viewer'], ['admin', 'user']) is True
        assert matches_scope(['viewer'], ['admin', 'user']) is False
    
    def test_regex_handler_scope(self):
        """Handler scope can be a regex pattern."""
        admin_pattern = re.compile(r'^admin.*$')
        assert matches_scope(['admin'], [admin_pattern]) is True
        assert matches_scope(['admin_super'], [admin_pattern]) is True
        assert matches_scope(['user'], [admin_pattern]) is False
    
    def test_regex_request_scope(self):
        """Request scope can be a regex pattern."""
        admin_pattern = re.compile(r'^admin.*$')
        assert matches_scope([admin_pattern], ['admin']) is True
        assert matches_scope([admin_pattern], ['admin_super']) is True
        assert matches_scope([admin_pattern], ['user']) is False
    
    def test_regex_both_scopes(self):
        """Both request and handler scopes can be regex."""
        pattern1 = re.compile(r'^admin.*$')
        pattern2 = re.compile(r'^admin.*$')  # Same pattern
        pattern3 = re.compile(r'^user.*$')  # Different pattern
        assert matches_scope([pattern1], [pattern2]) is True
        assert matches_scope([pattern1], [pattern3]) is False


class TestScopeBasedRouting:
    """Tests for scope-based routing in the router."""
    
    @pytest.fixture
    def router(self):
        return MessageRouter()
    
    @pytest.mark.asyncio
    async def test_handler_with_scope_restriction(self, router):
        """Handler with specific scope should only be accessible to matching requests."""
        processed = []
        
        async def admin_process(event, context):
            processed.append(('admin', event.type))
            yield UAMPEvent(id='resp', type='response.delta', payload={})
        
        router.register_handler(Handler(
            name='admin-handler',
            subscribes=['admin.action'],
            produces=['response.delta'],
            priority=100,
            scopes=['admin'],  # Admin only
            process=admin_process,
        ))
        
        # Request with admin scope should succeed
        await router.send(
            UAMPEvent(id='1', type='admin.action', payload={}),
            RouterContext(scopes=['admin'])
        )
        assert len(processed) == 1
        
        # Request with user scope should NOT route to this handler
        processed.clear()
        await router.send(
            UAMPEvent(id='2', type='admin.action', payload={}),
            RouterContext(scopes=['user'])
        )
        assert len(processed) == 0  # Handler wasn't called
    
    @pytest.mark.asyncio
    async def test_handler_with_all_scope_accessible_by_anyone(self, router):
        """Handler with 'all' scope should be accessible by any request."""
        processed = []
        
        async def public_process(event, context):
            processed.append(event.type)
            yield UAMPEvent(id='resp', type='response.delta', payload={})
        
        router.register_handler(Handler(
            name='public-handler',
            subscribes=['public.action'],
            produces=['response.delta'],
            scopes=['all'],  # Accessible to everyone
            process=public_process,
        ))
        
        # Any scope should work
        await router.send(
            UAMPEvent(id='1', type='public.action', payload={}),
            RouterContext(scopes=['user'])
        )
        await router.send(
            UAMPEvent(id='2', type='public.action', payload={}),
            RouterContext(scopes=['admin'])
        )
        await router.send(
            UAMPEvent(id='3', type='public.action', payload={}),
            RouterContext(scopes=['custom_role'])
        )
        assert len(processed) == 3
    
    @pytest.mark.asyncio
    async def test_observer_scope_filtering(self, router):
        """Observers should respect scopes too."""
        observed = []
        
        async def admin_observer(event, context):
            observed.append(event.type)
        
        router.register_observer(Observer(
            name='admin-observer',
            subscribes=['test.event'],  # Only observe test.event, not system events
            scopes=['admin'],  # Only observe for admin requests
            handler=admin_observer,
        ))
        
        # Admin request should trigger observer
        await router.send(
            UAMPEvent(id='1', type='test.event', payload={}),
            RouterContext(scopes=['admin'])
        )
        assert len(observed) == 1
        
        # User request should NOT trigger observer (scope doesn't match)
        observed.clear()
        await router.send(
            UAMPEvent(id='2', type='test.event', payload={}),
            RouterContext(scopes=['user'])
        )
        assert len(observed) == 0
    
    @pytest.mark.asyncio
    async def test_priority_with_different_scopes(self, router):
        """Higher priority handler should be selected if scope matches."""
        processed = []
        
        async def high_prio_admin(event, context):
            processed.append('high-admin')
            yield UAMPEvent(id='resp', type='response.delta', payload={})
        
        async def low_prio_all(event, context):
            processed.append('low-all')
            yield UAMPEvent(id='resp', type='response.delta', payload={})
        
        router.register_handler(Handler(
            name='high-prio-admin',
            subscribes=['action'],
            produces=['response.delta'],
            priority=100,
            scopes=['admin'],
            process=high_prio_admin,
        ))
        router.register_handler(Handler(
            name='low-prio-all',
            subscribes=['action'],
            produces=['response.delta'],
            priority=50,
            scopes=['all'],
            process=low_prio_all,
        ))
        
        # Admin should get high priority handler
        await router.send(
            UAMPEvent(id='1', type='action', payload={}),
            RouterContext(scopes=['admin'])
        )
        assert processed == ['high-admin']
        
        # User should fallback to lower priority handler
        processed.clear()
        await router.send(
            UAMPEvent(id='2', type='action', payload={}),
            RouterContext(scopes=['user'])
        )
        assert processed == ['low-all']
    
    @pytest.mark.asyncio
    async def test_regex_scope_in_handler(self, router):
        """Handler with regex scope should match request scopes."""
        processed = []
        
        async def admin_process(event, context):
            processed.append(event.type)
            yield UAMPEvent(id='resp', type='response.delta', payload={})
        
        admin_pattern = re.compile(r'^admin.*$')
        router.register_handler(Handler(
            name='admin-wildcard',
            subscribes=['action'],
            produces=['response.delta'],
            scopes=[admin_pattern],  # Regex scope
            process=admin_process,
        ))
        
        # admin should match
        await router.send(
            UAMPEvent(id='1', type='action', payload={}),
            RouterContext(scopes=['admin'])
        )
        assert len(processed) == 1
        
        # admin_super should also match
        processed.clear()
        await router.send(
            UAMPEvent(id='2', type='action', payload={}),
            RouterContext(scopes=['admin_super'])
        )
        assert len(processed) == 1
        
        # user should NOT match
        processed.clear()
        await router.send(
            UAMPEvent(id='3', type='action', payload={}),
            RouterContext(scopes=['user'])
        )
        assert len(processed) == 0
