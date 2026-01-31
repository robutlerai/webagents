"""
UAMP Message Router

Central hub for capability-based message routing with auto-wiring,
observers, loop prevention, and system events.
"""

import re
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Callable,
    AsyncGenerator,
    Optional,
    Union,
    Set,
    Any,
    Awaitable,
    Pattern,
)
from abc import ABC, abstractmethod


# ============================================================================
# Types
# ============================================================================

@dataclass
class UAMPEvent:
    """UAMP Event with loop prevention metadata."""
    id: str
    type: str
    payload: Dict[str, Any]
    source: Optional[str] = None  # Handler that produced this message
    ttl: Optional[int] = None  # Time-to-live (max hops)
    seen: Optional[Set[str]] = None  # Handler IDs that have processed this message


@dataclass
class RouterContext:
    """Execution context passed to handlers."""
    cancelled: bool = False
    auth_token: Optional[str] = None
    session_id: Optional[str] = None
    scopes: List[str] = field(default_factory=lambda: ['all'])  # Request scopes for access control
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Handler:
    """Handler for processing events."""
    name: str
    subscribes: List[Union[str, Pattern]]  # Event types/patterns
    produces: List[str]  # Event types this handler emits
    priority: int = 0
    scopes: List[str] = field(default_factory=lambda: ['all'])  # Handler's required scopes
    process: Callable[[UAMPEvent, RouterContext], AsyncGenerator[UAMPEvent, None]] = None


@dataclass
class Observer:
    """Observer for non-consuming message listening."""
    name: str
    subscribes: List[Union[str, Pattern]]  # Event types/patterns
    scopes: List[str] = field(default_factory=lambda: ['all'])  # Observer's required scopes
    handler: Callable[[UAMPEvent, Optional[RouterContext]], Awaitable[None]] = None


class TransportSink(ABC):
    """Abstract base class for transport sinks."""
    
    @property
    @abstractmethod
    def id(self) -> str:
        """Unique sink ID."""
        pass
    
    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Whether sink is active."""
        pass
    
    @abstractmethod
    async def send(self, event: Dict[str, Any]) -> None:
        """Send event to transport."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the sink."""
        pass


# ============================================================================
# Constants
# ============================================================================

class SystemEvents:
    """System event types for control flow."""
    ERROR = 'system.error'
    STOP = 'system.stop'
    CANCEL = 'system.cancel'
    PING = 'system.ping'
    PONG = 'system.pong'
    UNROUTABLE = 'system.unroutable'


class UAMPEventTypes:
    """UAMP Event Types."""
    # Input events
    INPUT_TEXT = 'input.text'
    INPUT_AUDIO = 'input.audio'
    INPUT_IMAGE = 'input.image'
    # Response events
    RESPONSE_DELTA = 'response.delta'
    RESPONSE_DONE = 'response.done'
    # Audio events
    AUDIO_DELTA = 'audio.delta'
    TRANSCRIPT_DELTA = 'transcript.delta'
    # System events
    SYSTEM_ERROR = 'system.error'
    SYSTEM_STOP = 'system.stop'
    SYSTEM_CANCEL = 'system.cancel'
    SYSTEM_PING = 'system.ping'
    SYSTEM_PONG = 'system.pong'


DEFAULT_TTL = 10
DEFAULT_SINK_PRIORITY = -1000


# ============================================================================
# Utility Functions
# ============================================================================

def generate_id() -> str:
    """Generate a UUID v4."""
    return str(uuid.uuid4())


def matches_subscription(event_type: str, patterns: List[Union[str, Pattern]]) -> bool:
    """Check if an event type matches a subscription pattern."""
    for pattern in patterns:
        if pattern == '*':
            return True
        if isinstance(pattern, Pattern):
            if pattern.search(event_type):
                return True
        elif pattern == event_type:
            return True
    return False


def matches_scope(request_scopes: List[Union[str, Pattern]], handler_scopes: List[Union[str, Pattern]]) -> bool:
    """Check if request scopes match handler's required scopes.
    
    Args:
        request_scopes: Scopes from the current request/context (strings or regex patterns)
        handler_scopes: Scopes required by the handler (strings or regex patterns)
        
    Returns:
        True if access is allowed:
        - Empty/None scopes or 'all' or '' means no restriction (accessible to everyone)
        - Any request scope matches any handler scope (supports regex)
    """
    # Normalize empty/None to ['all']
    if not handler_scopes:
        handler_scopes = ['all']
    if not request_scopes:
        request_scopes = ['all']
    
    # Filter out empty strings and treat them as 'all'
    handler_scopes = [s if s != '' else 'all' for s in handler_scopes]
    request_scopes = [s if s != '' else 'all' for s in request_scopes]
    
    # 'all' scope on handler means accessible to everyone
    if 'all' in handler_scopes:
        return True
    # 'all' scope on request means superuser/admin access
    if 'all' in request_scopes:
        return True
    
    # Check for matching scopes (supports regex patterns)
    for req_scope in request_scopes:
        for handler_scope in handler_scopes:
            # Both are regex patterns
            if isinstance(req_scope, Pattern) and isinstance(handler_scope, Pattern):
                # If patterns are the same, consider it a match
                if req_scope.pattern == handler_scope.pattern:
                    return True
            # Request scope is regex
            elif isinstance(req_scope, Pattern):
                if isinstance(handler_scope, str) and req_scope.search(handler_scope):
                    return True
            # Handler scope is regex
            elif isinstance(handler_scope, Pattern):
                if isinstance(req_scope, str) and handler_scope.search(req_scope):
                    return True
            # Both are strings - exact match
            elif req_scope == handler_scope:
                return True
    
    return False


# ============================================================================
# Transport Sink Implementations
# ============================================================================

class CallbackSink(TransportSink):
    """Callback-based sink for testing and programmatic event handling."""
    
    def __init__(self, callback: Callable[[Dict[str, Any]], Awaitable[None]], sink_id: Optional[str] = None):
        self._id = sink_id or f"callback-{generate_id()[:8]}"
        self._callback = callback
        self._is_active = True
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def is_active(self) -> bool:
        return self._is_active
    
    async def send(self, event: Dict[str, Any]) -> None:
        if self.is_active:
            await self._callback(event)
    
    def close(self) -> None:
        self._is_active = False


class BufferSink(TransportSink):
    """Buffer sink that collects events for later retrieval."""
    
    def __init__(self, sink_id: Optional[str] = None, max_size: int = 1000):
        self._id = sink_id or f"buffer-{generate_id()[:8]}"
        self._events: List[Dict[str, Any]] = []
        self._max_size = max_size
        self._is_active = True
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def is_active(self) -> bool:
        return self._is_active
    
    async def send(self, event: Dict[str, Any]) -> None:
        if self.is_active:
            self._events.append(event)
            if len(self._events) > self._max_size:
                self._events = self._events[-self._max_size:]
    
    def close(self) -> None:
        self._is_active = False
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get all collected events."""
        return list(self._events)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._events = []
    
    def __len__(self) -> int:
        return len(self._events)


# ============================================================================
# MessageRouter Class
# ============================================================================

# Type aliases for hooks
UnroutableHandler = Callable[[UAMPEvent, Optional[RouterContext]], Awaitable[None]]
ErrorHandler = Callable[[Exception, UAMPEvent, Handler, Optional[RouterContext]], Awaitable[None]]
RouteInterceptor = Callable[[UAMPEvent, Optional[Handler], Optional[RouterContext]], Awaitable[Optional[UAMPEvent]]]


class MessageRouter:
    """Central message router with capability-based routing."""
    
    def __init__(self):
        self._routes: Dict[str, List[Dict[str, Any]]] = {}  # eventType -> [(handler, priority)]
        self._handlers: Dict[str, Handler] = {}
        self._observers: List[Observer] = []
        self._sinks: Dict[str, TransportSink] = {}
        self._default_handler: Optional[Handler] = None
        self._active_sink_id: Optional[str] = None
        
        # Extensibility hooks
        self._on_unroutable: Optional[UnroutableHandler] = None
        self._on_error: Optional[ErrorHandler] = None
        self._before_route: Optional[RouteInterceptor] = None
        self._after_route: Optional[RouteInterceptor] = None
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    async def send(self, event: UAMPEvent, context: Optional[RouterContext] = None) -> None:
        """Send a message into the router (main entry point for transports)."""
        # Ensure event has an ID
        if not event.id:
            event.id = generate_id()
        
        # 1. TTL check - prevent deep chains
        ttl = event.ttl if event.ttl is not None else DEFAULT_TTL
        if ttl <= 0:
            await self._emit_system_event(SystemEvents.ERROR, {
                'error': 'TTL exceeded',
                'original_event': {
                    'id': event.id,
                    'type': event.type,
                }
            })
            return
        
        # 2. Notify all observers (non-consuming, parallel)
        await self._notify_observers(event, context)
        
        # 3. Handle system events
        if self._is_system_event(event):
            await self._handle_system_event(event, context)
            return
        
        # 4. Find handler for event type (respecting scopes)
        request_scopes = context.scopes if context else ['all']
        handler = self._get_handler(event.type, request_scopes)
        
        # 5. Apply beforeRoute interceptor
        if self._before_route:
            intercepted = await self._before_route(event, handler, context)
            if intercepted is None:
                return  # Interceptor blocked the event
            event = intercepted
        
        # 6. Use default handler if no specific handler found
        if handler is None:
            handler = self._default_handler
        
        # 7. Check for wildcard handler if still no handler
        if handler is None:
            wildcard_routes = self._routes.get('*', [])
            if wildcard_routes:
                handler = wildcard_routes[0]['handler']
        
        # 8. If no handler found, emit unroutable event
        if handler is None:
            if self._on_unroutable:
                await self._on_unroutable(event, context)
            await self._emit_system_event(SystemEvents.UNROUTABLE, {
                'original_event': {
                    'id': event.id,
                    'type': event.type,
                }
            })
            return
        
        # 9. Source check - don't route back to producer
        if event.source == handler.name:
            return  # Skip - would create loop
        
        # 10. Seen check - don't process same message twice by same handler
        seen = event.seen or set()
        if handler.name in seen:
            return  # Already processed by this handler
        
        # 11. Process through handler
        await self._process_handler_output(handler, event, context)
        
        # 12. Apply afterRoute interceptor
        if self._after_route:
            await self._after_route(event, handler, context)
    
    def register_handler(self, handler: Handler) -> None:
        """Register a handler (auto-wires based on capabilities)."""
        self._handlers[handler.name] = handler
        self._build_routing_table()
    
    def unregister_handler(self, name: str) -> None:
        """Unregister a handler."""
        self._handlers.pop(name, None)
        self._build_routing_table()
    
    def register_observer(self, observer: Observer) -> None:
        """Register a non-consuming observer."""
        self._observers.append(observer)
    
    def unregister_observer(self, name: str) -> None:
        """Unregister an observer."""
        self._observers = [o for o in self._observers if o.name != name]
    
    def route(self, event_type: str, handler_name: str, priority: Optional[int] = None) -> None:
        """Add explicit route (overrides auto-wiring)."""
        handler = self._handlers.get(handler_name)
        if handler is None:
            raise ValueError(f"Handler '{handler_name}' not found")
        
        routes = self._routes.get(event_type, [])
        
        # Remove existing route for this handler
        routes = [r for r in routes if r['handler'].name != handler_name]
        
        # Add new route
        routes.append({
            'event_type': event_type,
            'handler': handler,
            'priority': priority if priority is not None else handler.priority,
        })
        
        # Sort by priority (higher first)
        routes.sort(key=lambda r: r['priority'], reverse=True)
        
        self._routes[event_type] = routes
    
    def register_sink(self, sink: TransportSink) -> None:
        """Register a transport sink for responses."""
        self._sinks[sink.id] = sink
    
    def register_default_sink(self, sink: TransportSink) -> None:
        """Register a default sink that catches all unhandled response events."""
        # Register as handler with lowest priority and wildcard pattern
        async def default_sink_process(event: UAMPEvent, context: RouterContext) -> AsyncGenerator[UAMPEvent, None]:
            await sink.send({
                'type': event.type,
                'payload': event.payload,
            })
            return
            yield  # Make it a generator
        
        self.register_handler(Handler(
            name=f'default-sink-{sink.id}',
            subscribes=['*'],
            produces=[],
            priority=DEFAULT_SINK_PRIORITY,
            process=default_sink_process,
        ))
        self._sinks[sink.id] = sink
    
    def unregister_sink(self, sink_id: str) -> None:
        """Unregister a sink."""
        self._sinks.pop(sink_id, None)
        # Also remove any default sink handler
        self._handlers.pop(f'default-sink-{sink_id}', None)
        self._build_routing_table()
    
    def set_active_sink(self, sink_id: str) -> None:
        """Set active sink (receives responses)."""
        if sink_id not in self._sinks:
            raise ValueError(f"Sink '{sink_id}' not registered")
        self._active_sink_id = sink_id
    
    @property
    def active_sink(self) -> Optional[TransportSink]:
        """Get the active sink."""
        if self._active_sink_id:
            return self._sinks.get(self._active_sink_id)
        return None
    
    def set_default(self, handler_name: str) -> None:
        """Set default handler (fallback when no capability match)."""
        handler = self._handlers.get(handler_name)
        if handler is None:
            raise ValueError(f"Handler '{handler_name}' not found")
        self._default_handler = handler
    
    @property
    def default_handler(self) -> Optional[Handler]:
        """Get default handler."""
        return self._default_handler
    
    def get_handlers(self) -> Dict[str, Handler]:
        """Get all registered handlers."""
        return dict(self._handlers)
    
    def get_observers(self) -> List[Observer]:
        """Get all registered observers."""
        return list(self._observers)
    
    # =========================================================================
    # Extensibility Hooks
    # =========================================================================
    
    def on_unroutable(self, handler: UnroutableHandler) -> None:
        """Set handler for unroutable events."""
        self._on_unroutable = handler
    
    def on_error(self, handler: ErrorHandler) -> None:
        """Set handler for errors during processing."""
        self._on_error = handler
    
    def before_route(self, interceptor: RouteInterceptor) -> None:
        """Set interceptor called before routing."""
        self._before_route = interceptor
    
    def after_route(self, interceptor: RouteInterceptor) -> None:
        """Set interceptor called after routing."""
        self._after_route = interceptor
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _build_routing_table(self) -> None:
        """Build routing table from handler capabilities."""
        self._routes.clear()
        
        for handler in self._handlers.values():
            for pattern in handler.subscribes:
                # For string patterns, use exact match key
                if isinstance(pattern, Pattern):
                    key = pattern.pattern
                else:
                    key = pattern
                
                routes = self._routes.get(key, [])
                routes.append({
                    'event_type': key,
                    'handler': handler,
                    'priority': handler.priority,
                })
                
                # Sort by priority (higher first)
                routes.sort(key=lambda r: r['priority'], reverse=True)
                
                self._routes[key] = routes
    
    def _get_handler(self, event_type: str, request_scopes: Optional[List[str]] = None) -> Optional[Handler]:
        """Get the best handler for an event type, respecting scopes.
        
        Args:
            event_type: The event type to match
            request_scopes: Scopes from the current request (default: ['all'])
            
        Returns:
            Best matching handler that the request has access to, or None
        """
        scopes = request_scopes or ['all']
        
        # 1. Check for exact match
        exact_routes = self._routes.get(event_type)
        if exact_routes:
            # Find first handler that matches scopes
            for route in exact_routes:
                handler = route['handler']
                if matches_scope(scopes, handler.scopes):
                    return handler
        
        # 2. Check for regex matches
        for key, routes in self._routes.items():
            if not routes:
                continue
            
            for route in routes:
                handler = route['handler']
                # Check scope first
                if not matches_scope(scopes, handler.scopes):
                    continue
                # Then check pattern match
                for pattern in handler.subscribes:
                    if isinstance(pattern, Pattern) and pattern.search(event_type):
                        return handler
        
        return None
    
    async def _notify_observers(self, event: UAMPEvent, context: Optional[RouterContext]) -> None:
        """Notify all observers (non-consuming), respecting scopes."""
        tasks = []
        request_scopes = context.scopes if context else ['all']
        
        for observer in self._observers:
            # Check scope access
            if not matches_scope(request_scopes, observer.scopes):
                continue
            # Check subscription match
            if matches_subscription(event.type, observer.subscribes):
                tasks.append(self._safe_observer_call(observer, event, context))
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def _safe_observer_call(
        self,
        observer: Observer,
        event: UAMPEvent,
        context: Optional[RouterContext]
    ) -> None:
        """Call observer safely, catching errors."""
        try:
            await observer.handler(event, context)
        except Exception as e:
            # Observers should not break the flow
            print(f"Observer {observer.name} error: {e}")
    
    async def _process_handler_output(
        self,
        handler: Handler,
        event: UAMPEvent,
        context: Optional[RouterContext]
    ) -> None:
        """Process handler output and route produced events."""
        seen = set(event.seen) if event.seen else set()
        seen.add(handler.name)
        
        try:
            async for output in handler.process(event, context or RouterContext()):
                # Route handler output back through router
                await self.send(
                    UAMPEvent(
                        id=output.id or generate_id(),
                        type=output.type,
                        payload=output.payload,
                        source=handler.name,
                        ttl=(event.ttl if event.ttl is not None else DEFAULT_TTL) - 1,
                        seen=set(seen),
                    ),
                    context
                )
        except Exception as e:
            if self._on_error:
                await self._on_error(e, event, handler, context)
            else:
                raise
    
    def _is_system_event(self, event: UAMPEvent) -> bool:
        """Check if event is a system event."""
        return event.type.startswith('system.')
    
    async def _handle_system_event(self, event: UAMPEvent, context: Optional[RouterContext]) -> None:
        """Handle system events."""
        if event.type in (SystemEvents.STOP, SystemEvents.CANCEL):
            # Signal handlers to stop via context
            if context:
                context.cancelled = True
            # Deliver to active sink
            await self._deliver_to_active_sink(event)
        
        elif event.type == SystemEvents.ERROR:
            # Deliver error to sink
            await self._deliver_to_active_sink(event)
        
        elif event.type == SystemEvents.PING:
            # Respond with pong
            await self._deliver_to_active_sink(UAMPEvent(
                id=generate_id(),
                type=SystemEvents.PONG,
                payload={},
            ))
        
        elif event.type == SystemEvents.UNROUTABLE:
            # Deliver unroutable notification
            await self._deliver_to_active_sink(event)
        
        else:
            # Unknown system event - deliver as-is
            await self._deliver_to_active_sink(event)
    
    async def _deliver_to_active_sink(self, event: UAMPEvent) -> None:
        """Deliver event to active sink."""
        sink = self.active_sink
        if sink is not None and sink.is_active:
            await sink.send({
                'type': event.type,
                'payload': event.payload,
            })
    
    async def _emit_system_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Emit a system event."""
        event = UAMPEvent(
            id=generate_id(),
            type=event_type,
            payload=payload,
        )
        
        # Notify observers of system event
        await self._notify_observers(event, None)
        
        # Handle the system event
        await self._handle_system_event(event, None)
