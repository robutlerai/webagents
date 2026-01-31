"""
Server Middleware - WebAgents V2.0

Production-ready middleware for request timeout, rate limiting, 
and comprehensive request management.
"""

import time
import asyncio
from typing import Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
import uuid

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .context.context_vars import create_context, set_context


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Max requests in burst window
    burst_window_seconds: int = 1  # Burst window duration


@dataclass
class ClientUsage:
    """Track client usage for rate limiting"""
    client_id: str
    minute_count: int = 0
    hour_count: int = 0
    day_count: int = 0
    burst_count: int = 0
    last_minute_reset: datetime = None
    last_hour_reset: datetime = None
    last_day_reset: datetime = None
    last_burst_reset: datetime = None
    
    def __post_init__(self):
        now = datetime.utcnow()
        if self.last_minute_reset is None:
            self.last_minute_reset = now
        if self.last_hour_reset is None:
            self.last_hour_reset = now
        if self.last_day_reset is None:
            self.last_day_reset = now
        if self.last_burst_reset is None:
            self.last_burst_reset = now


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Request timeout middleware"""
    
    def __init__(self, app, timeout_seconds: float = 300.0):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
        
    async def dispatch(self, request: Request, call_next):
        try:
            # Apply timeout to request processing
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds
            )
            return response
            
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=408,
                content={
                    "error": {
                        "type": "timeout_error",
                        "message": f"Request timeout after {self.timeout_seconds} seconds",
                        "code": "request_timeout"
                    }
                }
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Comprehensive rate limiting middleware"""
    
    def __init__(
        self,
        app,
        default_rule: RateLimitRule = None,
        user_rules: Dict[str, RateLimitRule] = None,
        enable_rate_limiting: bool = True
    ):
        super().__init__(app)
        self.default_rule = default_rule or RateLimitRule()
        self.user_rules = user_rules or {}
        self.enable_rate_limiting = enable_rate_limiting
        
        # In-memory storage for client usage (production should use Redis)
        self.client_usage: Dict[str, ClientUsage] = {}
        
        # Exempt paths from rate limiting
        self.exempt_paths = {'/health', '/health/detailed', '/'}
        
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request"""
        # Priority order for client identification:
        # 1. User ID header (if authenticated)
        # 2. API key (if provided)
        # 3. IP address (fallback)
        
        user_id = request.headers.get("x-user-id")
        if user_id and user_id != "anonymous":
            return f"user:{user_id}"
        
        api_key = request.headers.get("authorization")
        if api_key:
            # Hash for privacy but keep unique
            return f"api:{hash(api_key) % 1000000}"
        
        # Fallback to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    def _get_rate_limit_rule(self, client_id: str, request: Request) -> RateLimitRule:
        """Get rate limit rule for client"""
        
        # Check for user-specific rules
        user_id = request.headers.get("x-user-id")
        if user_id and user_id in self.user_rules:
            return self.user_rules[user_id]
        
        # Check for agent owner rules
        agent_owner = request.headers.get("x-agent-owner-id")
        if agent_owner and agent_owner in self.user_rules:
            return self.user_rules[agent_owner]
        
        # Default rule
        return self.default_rule
    
    def _reset_counters_if_needed(self, usage: ClientUsage) -> None:
        """Reset usage counters if time windows have expired"""
        now = datetime.utcnow()
        
        # Reset burst counter
        if now - usage.last_burst_reset >= timedelta(seconds=1):
            usage.burst_count = 0
            usage.last_burst_reset = now
        
        # Reset minute counter
        if now - usage.last_minute_reset >= timedelta(minutes=1):
            usage.minute_count = 0
            usage.last_minute_reset = now
        
        # Reset hour counter
        if now - usage.last_hour_reset >= timedelta(hours=1):
            usage.hour_count = 0
            usage.last_hour_reset = now
        
        # Reset day counter
        if now - usage.last_day_reset >= timedelta(days=1):
            usage.day_count = 0
            usage.last_day_reset = now
    
    def _check_rate_limits(self, client_id: str, rule: RateLimitRule) -> Tuple[bool, str, int]:
        """
        Check if client has exceeded rate limits
        
        Returns:
            Tuple of (allowed: bool, error_message: str, retry_after_seconds: int)
        """
        
        # Get or create client usage
        if client_id not in self.client_usage:
            self.client_usage[client_id] = ClientUsage(client_id=client_id)
        
        usage = self.client_usage[client_id]
        self._reset_counters_if_needed(usage)
        
        # Check burst limit (most restrictive)
        if usage.burst_count >= rule.burst_limit:
            return False, f"Burst limit exceeded ({rule.burst_limit} requests per second)", 1
        
        # Check per-minute limit
        if usage.minute_count >= rule.requests_per_minute:
            remaining_seconds = 60 - (datetime.utcnow() - usage.last_minute_reset).seconds
            return False, f"Rate limit exceeded ({rule.requests_per_minute} requests per minute)", remaining_seconds
        
        # Check per-hour limit
        if usage.hour_count >= rule.requests_per_hour:
            remaining_seconds = 3600 - (datetime.utcnow() - usage.last_hour_reset).seconds
            return False, f"Rate limit exceeded ({rule.requests_per_hour} requests per hour)", remaining_seconds
        
        # Check per-day limit
        if usage.day_count >= rule.requests_per_day:
            remaining_seconds = 86400 - (datetime.utcnow() - usage.last_day_reset).seconds
            return False, f"Rate limit exceeded ({rule.requests_per_day} requests per day)", remaining_seconds
        
        return True, "", 0
    
    def _increment_counters(self, client_id: str) -> None:
        """Increment usage counters for client"""
        usage = self.client_usage[client_id]
        usage.burst_count += 1
        usage.minute_count += 1
        usage.hour_count += 1
        usage.day_count += 1
    
    def _cleanup_old_entries(self) -> None:
        """Clean up old client usage entries to prevent memory leaks"""
        now = datetime.utcnow()
        cutoff = now - timedelta(days=2)  # Keep 2 days of history
        
        # Remove entries older than cutoff
        old_clients = [
            client_id for client_id, usage in self.client_usage.items()
            if usage.last_day_reset < cutoff
        ]
        
        for client_id in old_clients:
            del self.client_usage[client_id]
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting if disabled
        if not self.enable_rate_limiting:
            return await call_next(request)
        
        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)
        
        # Get client ID and rate limit rule
        client_id = self._get_client_id(request)
        rule = self._get_rate_limit_rule(client_id, request)
        
        # Check rate limits
        allowed, error_message, retry_after = self._check_rate_limits(client_id, rule)
        
        if not allowed:
            # Return rate limit error
            response = JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "type": "rate_limit_exceeded",
                        "message": error_message,
                        "code": "too_many_requests",
                        "retry_after": retry_after
                    }
                },
                headers={"Retry-After": str(retry_after)}
            )
            return response
        
        # Increment counters and continue
        self._increment_counters(client_id)
        
        # Periodic cleanup (every 1000 requests)
        if len(self.client_usage) % 1000 == 0:
            self._cleanup_old_entries()
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        # Add rate limit info headers
        usage = self.client_usage[client_id]
        response.headers["X-RateLimit-Limit-Minute"] = str(rule.requests_per_minute)
        response.headers["X-RateLimit-Remaining-Minute"] = str(max(0, rule.requests_per_minute - usage.minute_count))
        response.headers["X-RateLimit-Limit-Hour"] = str(rule.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Hour"] = str(max(0, rule.requests_per_hour - usage.hour_count))
        response.headers["X-RateLimit-Limit-Day"] = str(rule.requests_per_day)
        response.headers["X-RateLimit-Remaining-Day"] = str(max(0, rule.requests_per_day - usage.day_count))
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging and metrics middleware"""
    
    def __init__(self, app, enable_logging: bool = True):
        super().__init__(app)
        self.enable_logging = enable_logging
        self.request_count = 0
        self.error_count = 0
        
    async def dispatch(self, request: Request, call_next):
        if not self.enable_logging:
            return await call_next(request)
        
        start_time = time.time()
        self.request_count += 1
        request_id = str(uuid.uuid4())[:8]
        
        # Log request start
        print(f"[{request_id}] {request.method} {request.url.path} - Started")
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log successful request
            print(f"[{request_id}] {request.method} {request.url.path} - {response.status_code} ({duration_ms:.1f}ms)")
            
            # Add request ID header
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            self.error_count += 1
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            print(f"[{request_id}] {request.method} {request.url.path} - ERROR ({duration_ms:.1f}ms): {str(e)}")
            
            # Re-raise exception
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "success_rate": (self.request_count - self.error_count) / max(1, self.request_count)
        } 