"""
FastAPI Middleware - WebAgents V2.0

Request logging and rate limiting middleware for the WebAgents server.
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Get logger for middleware
middleware_logger = logging.getLogger("webagents.server.middleware")


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests"""
    
    def __init__(self, app, timeout: float = 300.0):
        super().__init__(app)
        self.timeout = timeout

    async def __call__(self, scope, receive, send):
        # BaseHTTPMiddleware does not support WebSocket -- pass through
        if scope["type"] == "websocket":
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request using proper logging (INFO level for important requests)
        middleware_logger.info(f"{request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        duration = time.time() - start_time
        
        # Use different log levels based on status code
        if response.status_code >= 400:
            middleware_logger.warning(f"{response.status_code} - {duration:.3f}s")
        else:
            middleware_logger.info(f"{response.status_code} - {duration:.3f}s")
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests"""
    
    def __init__(self, app, default_rule: RateLimitRule, user_rules: Dict[str, RateLimitRule]):
        super().__init__(app)
        self.default_rule = default_rule
        self.user_rules = user_rules
        self.request_counts = {}

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)
    
    async def dispatch(self, request: Request, call_next):
        # Simple rate limiting (in production, use Redis or similar)
        client_ip = request.client.host if request.client else "unknown"
        
        # For now, just pass through - rate limiting would be implemented here
        response = await call_next(request)
        return response


class WorkingDirMiddleware(BaseHTTPMiddleware):
    """Middleware for extracting X-Working-Dir header and storing in request state.
    
    This is used to pass the client's working directory to the server,
    which is important for embedded agents like robutler that need to
    operate in the directory where the command was invoked.
    """

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)
    
    async def dispatch(self, request: Request, call_next):
        # Extract working dir from header and store in request state
        working_dir = request.headers.get("X-Working-Dir")
        
        if working_dir:
            request.state.working_dir = working_dir
            middleware_logger.debug(f"Set working_dir in request state: {working_dir}")
        
        response = await call_next(request)
        return response