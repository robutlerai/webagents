"""
Monitoring & Observability - WebAgents V2.0

Comprehensive monitoring system with Prometheus metrics, structured logging,
and request tracing for production deployments.
"""

import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Counter as CounterType
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextvars import ContextVar
from collections import defaultdict, Counter

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, CollectorRegistry, 
        generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus_client is not installed
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def time(self):
            return self
        def labels(self, **kwargs):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def dec(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
    
    class Info:
        def __init__(self, *args, **kwargs):
            pass
        def info(self, *args, **kwargs):
            pass
    
    def generate_latest(*args, **kwargs):
        return b"# Prometheus not available\n"
    
    CONTENT_TYPE_LATEST = "text/plain"
    CollectorRegistry = None


@dataclass
class RequestMetrics:
    """Metrics for individual requests"""
    request_id: str
    method: str
    path: str
    agent_name: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    tokens_used: int = 0
    stream: bool = False
    
    def finish(self, status_code: int, error: Optional[str] = None):
        """Mark request as finished"""
        self.end_time = time.time()
        self.status_code = status_code
        self.error = error
        self.duration_ms = (self.end_time - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
            "agent_name": self.agent_name,
            "duration_ms": self.duration_ms,
            "status_code": self.status_code,
            "error": self.error,
            "tokens_used": self.tokens_used,
            "stream": self.stream,
            "timestamp": datetime.fromtimestamp(self.start_time).isoformat()
        }


class PrometheusMetrics:
    """Prometheus metrics collection for WebAgents server"""
    
    def __init__(self, registry: CollectorRegistry = None):
        self.registry = registry
        
        if not PROMETHEUS_AVAILABLE:
            logging.warning("Prometheus client not available - metrics will be mocked")
            return
        
        # HTTP Request metrics
        self.http_requests_total = Counter(
            'webagents_http_requests_total',
            'Total HTTP requests',
            ['method', 'path', 'status_code', 'agent_name'],
            registry=registry
        )
        
        self.http_request_duration = Histogram(
            'webagents_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'path', 'agent_name'],
            registry=registry
        )
        
        self.http_requests_in_progress = Gauge(
            'webagents_http_requests_in_progress',
            'HTTP requests currently in progress',
            ['method', 'path', 'agent_name'],
            registry=registry
        )
        
        # Agent metrics
        self.agent_requests_total = Counter(
            'webagents_agent_requests_total',
            'Total requests per agent',
            ['agent_name', 'stream'],
            registry=registry
        )
        
        self.agent_request_duration = Histogram(
            'webagents_agent_request_duration_seconds',
            'Agent request processing duration',
            ['agent_name', 'stream'],
            registry=registry
        )
        
        self.agent_errors_total = Counter(
            'webagents_agent_errors_total',
            'Total agent processing errors',
            ['agent_name', 'error_type'],
            registry=registry
        )
        
        # Token usage metrics
        self.tokens_used_total = Counter(
            'webagents_tokens_used_total',
            'Total tokens used',
            ['agent_name', 'model'],
            registry=registry
        )
        
        self.credits_spent_total = Counter(
            'webagents_credits_spent_total',
            'Total credits spent',
            ['agent_name', 'user_id'],
            registry=registry
        )
        
        # System metrics
        self.active_agents = Gauge(
            'webagents_active_agents',
            'Number of active agents',
            registry=registry
        )
        
        self.dynamic_agents_cache_size = Gauge(
            'webagents_dynamic_agents_cache_size',
            'Dynamic agents cache size',
            registry=registry
        )
        
        self.rate_limit_exceeded_total = Counter(
            'webagents_rate_limit_exceeded_total',
            'Total rate limit violations',
            ['client_type', 'limit_type'],
            registry=registry
        )
        
        # Server info
        self.server_info = Info(
            'webagents_server_info',
            'Server information',
            registry=registry
        )
    
    def record_http_request_start(self, method: str, path: str, agent_name: Optional[str] = None):
        """Record start of HTTP request"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.http_requests_in_progress.labels(
            method=method,
            path=path,
            agent_name=agent_name or "unknown"
        ).inc()
    
    def record_http_request_finish(
        self, 
        method: str, 
        path: str, 
        status_code: int,
        duration: float,
        agent_name: Optional[str] = None
    ):
        """Record completion of HTTP request"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        labels = {
            "method": method,
            "path": path,
            "agent_name": agent_name or "unknown"
        }
        
        # Record request completion
        self.http_requests_total.labels(
            **labels,
            status_code=str(status_code)
        ).inc()
        
        # Record duration
        self.http_request_duration.labels(**labels).observe(duration)
        
        # Decrement in-progress counter
        self.http_requests_in_progress.labels(**labels).dec()
    
    def record_agent_request(
        self,
        agent_name: str,
        duration: float,
        stream: bool = False,
        error: Optional[str] = None
    ):
        """Record agent request completion"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Count request
        self.agent_requests_total.labels(
            agent_name=agent_name,
            stream=str(stream).lower()
        ).inc()
        
        # Record duration
        self.agent_request_duration.labels(
            agent_name=agent_name,
            stream=str(stream).lower()
        ).observe(duration)
        
        # Record error if any
        if error:
            self.agent_errors_total.labels(
                agent_name=agent_name,
                error_type=error
            ).inc()
    
    def record_token_usage(self, agent_name: str, model: str, tokens: int):
        """Record token usage"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.tokens_used_total.labels(
            agent_name=agent_name,
            model=model
        ).inc(tokens)
    
    def record_credit_usage(self, agent_name: str, user_id: str, credits: float):
        """Record credit spending"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.credits_spent_total.labels(
            agent_name=agent_name,
            user_id=user_id
        ).inc(credits)
    
    def update_active_agents(self, count: int):
        """Update active agents count"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.active_agents.set(count)
    
    def update_dynamic_cache_size(self, size: int):
        """Update dynamic agents cache size"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.dynamic_agents_cache_size.set(size)
    
    def record_rate_limit_exceeded(self, client_type: str, limit_type: str):
        """Record rate limit violation"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.rate_limit_exceeded_total.labels(
            client_type=client_type,
            limit_type=limit_type
        ).inc()
    
    def set_server_info(self, version: str, agents_count: int, **kwargs):
        """Set server information"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        info_dict = {
            "version": version,
            "agents_count": str(agents_count),
            **{k: str(v) for k, v in kwargs.items()}
        }
        
        self.server_info.info(info_dict)


class StructuredLogger:
    """Structured logging with JSON output and performance tracking"""
    
    def __init__(self, name: str = "webagents", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create JSON formatter
        formatter = logging.Formatter(
            json.dumps({
                "timestamp": "%(asctime)s",
                "level": "%(levelname)s", 
                "name": "%(name)s",
                "message": "%(message)s"
            })
        )
        
        # Console handler with JSON formatting
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data"""
        self._log(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging with structured data"""
        log_data = {
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        
        # Use the message as the log message, but include structured data
        extra_data = json.dumps(log_data)
        self.logger.log(level, extra_data)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        # Try to parse the message as JSON (if it's already structured)
        try:
            if isinstance(record.msg, str) and record.msg.startswith('{'):
                log_data = json.loads(record.msg)
            else:
                log_data = {"message": str(record.msg)}
        except (json.JSONDecodeError, TypeError):
            log_data = {"message": str(record.msg)}
        
        # Add standard fields
        log_data.update({
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno
        })
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class MonitoringSystem:
    """Comprehensive monitoring system coordinator"""
    
    def __init__(
        self,
        enable_prometheus: bool = True,
        enable_structured_logging: bool = True,
        metrics_port: int = 9090
    ):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_structured_logging = enable_structured_logging
        self.metrics_port = metrics_port
        
        # Initialize components
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self.prometheus = PrometheusMetrics(self.registry)
        else:
            self.registry = None
            self.prometheus = PrometheusMetrics(None)  # Mock metrics
        
        if self.enable_structured_logging:
            self.logger = StructuredLogger("webagents.monitoring")
        else:
            self.logger = None
        
        # Request tracking
        self.active_requests: Dict[str, RequestMetrics] = {}
        self.recent_requests: List[RequestMetrics] = []
        self.request_history_limit = 1000
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "total_errors": 0,
            "avg_response_time": 0.0,
            "requests_per_minute": 0
        }
        
        self.last_stats_update = time.time()
    
    def start_request(
        self,
        request_id: str,
        method: str,
        path: str,
        agent_name: Optional[str] = None,
        **kwargs
    ) -> RequestMetrics:
        """Start tracking a request"""
        
        metrics = RequestMetrics(
            request_id=request_id,
            method=method,
            path=path,
            agent_name=agent_name,
            start_time=time.time()
        )
        
        self.active_requests[request_id] = metrics
        
        # Record in Prometheus
        self.prometheus.record_http_request_start(method, path, agent_name)
        
        # Log request start
        if self.logger:
            self.logger.info(
                "Request started",
                request_id=request_id,
                method=method,
                path=path,
                agent_name=agent_name,
                **kwargs
            )
        
        return metrics
    
    def finish_request(
        self,
        request_id: str,
        status_code: int,
        error: Optional[str] = None,
        tokens_used: int = 0,
        **kwargs
    ):
        """Finish tracking a request"""
        
        if request_id not in self.active_requests:
            return
        
        metrics = self.active_requests.pop(request_id)
        metrics.finish(status_code, error)
        metrics.tokens_used = tokens_used
        
        # Record in Prometheus
        self.prometheus.record_http_request_finish(
            metrics.method,
            metrics.path,
            status_code,
            metrics.duration_ms / 1000,  # Convert to seconds
            metrics.agent_name
        )
        
        # Record agent-specific metrics
        if metrics.agent_name:
            self.prometheus.record_agent_request(
                metrics.agent_name,
                metrics.duration_ms / 1000,
                metrics.stream,
                error
            )
        
        # Log request completion
        if self.logger:
            self.logger.info(
                "Request completed",
                **metrics.to_dict(),
                **kwargs
            )
        
        # Add to recent requests history
        self.recent_requests.append(metrics)
        if len(self.recent_requests) > self.request_history_limit:
            self.recent_requests.pop(0)
        
        # Update performance stats
        self._update_performance_stats(metrics)
    
    def record_token_usage(self, agent_name: str, model: str, tokens: int):
        """Record token usage"""
        self.prometheus.record_token_usage(agent_name, model, tokens)
        
        if self.logger:
            self.logger.info(
                "Token usage recorded",
                agent_name=agent_name,
                model=model,
                tokens=tokens
            )
    
    def record_credit_usage(self, agent_name: str, user_id: str, credits: float):
        """Record credit spending"""
        self.prometheus.record_credit_usage(agent_name, user_id, credits)
        
        if self.logger:
            self.logger.info(
                "Credit usage recorded",
                agent_name=agent_name,
                user_id=user_id,
                credits=credits
            )
    
    def update_system_metrics(
        self,
        active_agents: int,
        dynamic_cache_size: Optional[int] = None
    ):
        """Update system-level metrics"""
        self.prometheus.update_active_agents(active_agents)
        
        if dynamic_cache_size is not None:
            self.prometheus.update_dynamic_cache_size(dynamic_cache_size)
    
    def get_metrics_response(self) -> bytes:
        """Get Prometheus metrics response"""
        if not self.enable_prometheus:
            return b"# Prometheus metrics not enabled\n"
        
        return generate_latest(self.registry)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        now = time.time()
        
        # Calculate requests per minute
        minute_ago = now - 60
        recent_requests = [
            r for r in self.recent_requests 
            if r.start_time > minute_ago
        ]
        
        return {
            "total_requests": len(self.recent_requests),
            "requests_last_minute": len(recent_requests),
            "active_requests": len(self.active_requests),
            "average_response_time_ms": self._calculate_avg_response_time(),
            "error_rate": self._calculate_error_rate(),
            "last_updated": now
        }
    
    def _update_performance_stats(self, metrics: RequestMetrics):
        """Update internal performance statistics"""
        self.performance_stats["total_requests"] += 1
        
        if metrics.error:
            self.performance_stats["total_errors"] += 1
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from recent requests"""
        if not self.recent_requests:
            return 0.0
        
        total_time = sum(r.duration_ms or 0 for r in self.recent_requests[-100:])
        count = len(self.recent_requests[-100:])
        
        return total_time / count if count > 0 else 0.0
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate from recent requests"""
        if not self.recent_requests:
            return 0.0
        
        recent = self.recent_requests[-100:]
        errors = sum(1 for r in recent if r.error is not None)
        
        return errors / len(recent) if recent else 0.0


# Global monitoring instance (initialized by server)
monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system() -> Optional[MonitoringSystem]:
    """Get global monitoring system instance"""
    return monitoring_system


def initialize_monitoring(
    enable_prometheus: bool = True,
    enable_structured_logging: bool = True,
    metrics_port: int = 9090
) -> MonitoringSystem:
    """Initialize global monitoring system"""
    global monitoring_system
    
    monitoring_system = MonitoringSystem(
        enable_prometheus=enable_prometheus,
        enable_structured_logging=enable_structured_logging,
        metrics_port=metrics_port
    )
    
    return monitoring_system 