"""
Test suite for Monitoring & Observability System - WebAgents V2.0
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from webagents.server.core.app import create_server
from webagents.server.monitoring import (
    PrometheusMetrics, StructuredLogger, MonitoringSystem,
    RequestMetrics
)
from webagents.agents.core.base_agent import BaseAgent


class TestPrometheusMetrics:
    """Test suite for Prometheus metrics collection"""
    
    def test_prometheus_metrics_initialization(self):
        """Test Prometheus metrics initialization"""
        metrics = PrometheusMetrics()
        
        # Should initialize without errors
        assert metrics is not None
        
        # Should have all expected metrics
        assert hasattr(metrics, 'http_requests_total')
        assert hasattr(metrics, 'http_request_duration')
        assert hasattr(metrics, 'agent_requests_total')
        assert hasattr(metrics, 'tokens_used_total')
    
    def test_http_request_recording(self):
        """Test HTTP request metrics recording"""
        metrics = PrometheusMetrics()
        
        # Record request start (should not raise errors)
        metrics.record_http_request_start("POST", "/assistant/chat/completions", "assistant")
        
        # Record request finish
        metrics.record_http_request_finish(
            method="POST",
            path="/assistant/chat/completions",
            status_code=200,
            duration=0.5,
            agent_name="assistant"
        )
        
        # Should complete without errors
        assert True
    
    def test_agent_request_recording(self):
        """Test agent request metrics recording"""
        metrics = PrometheusMetrics()
        
        # Record successful agent request
        metrics.record_agent_request(
            agent_name="assistant",
            duration=0.3,
            stream=False
        )
        
        # Record agent request with error
        metrics.record_agent_request(
            agent_name="assistant", 
            duration=0.1,
            stream=True,
            error="timeout"
        )
        
        # Should complete without errors
        assert True


class TestMonitoringSystem:
    """Test suite for the monitoring system coordinator"""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create monitoring system for testing"""
        return MonitoringSystem(
            enable_prometheus=True,
            enable_structured_logging=True
        )
    
    def test_monitoring_system_initialization(self, monitoring_system):
        """Test monitoring system initialization"""
        assert monitoring_system is not None
        assert monitoring_system.enable_prometheus is True
        assert monitoring_system.enable_structured_logging is True
        assert monitoring_system.prometheus is not None
        assert isinstance(monitoring_system.active_requests, dict)
        assert isinstance(monitoring_system.recent_requests, list)
    
    def test_request_tracking_lifecycle(self, monitoring_system):
        """Test complete request tracking lifecycle"""
        # Start tracking a request
        metrics = monitoring_system.start_request(
            request_id="test_req_123",
            method="POST",
            path="/assistant/chat/completions",
            agent_name="assistant"
        )
        
        # Verify request is being tracked
        assert "test_req_123" in monitoring_system.active_requests
        assert metrics.request_id == "test_req_123"
        
        # Finish the request
        monitoring_system.finish_request(
            request_id="test_req_123",
            status_code=200,
            tokens_used=150
        )
        
        # Verify request is no longer active but in history
        assert "test_req_123" not in monitoring_system.active_requests
        assert len(monitoring_system.recent_requests) == 1
        assert monitoring_system.recent_requests[0].request_id == "test_req_123"


class TestServerMonitoringIntegration:
    """Test suite for server monitoring integration"""
    
    def test_server_with_monitoring_enabled(self):
        """Test server creation with monitoring enabled"""
        server = create_server(
            agents=[],
            enable_monitoring=True,
            enable_prometheus=True,
            enable_structured_logging=True
        )
        
        # Should have monitoring system
        assert hasattr(server, 'monitoring')
        assert server.monitoring is not None
        assert server.enable_monitoring is True
    
    def test_health_endpoints_with_monitoring(self):
        """Test health check endpoints work with monitoring"""
        server = create_server(
            agents=[],
            enable_monitoring=True
        )
        
        client = TestClient(server.app)
        
        # Test basic health check
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        
        # Test detailed health check
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "agents" in data
    
    def test_readiness_endpoint(self):
        """Test Kubernetes readiness probe endpoint"""
        server = create_server(
            agents=[],
            enable_monitoring=True
        )
        
        client = TestClient(server.app)
        
        response = client.get("/ready")
        assert response.status_code in [200, 503]  # Could be either depending on agent health
        data = response.json()
        assert "status" in data
        assert "details" in data
    
    def test_liveness_endpoint(self):
        """Test Kubernetes liveness probe endpoint"""
        server = create_server(
            agents=[],
            enable_monitoring=True
        )
        
        client = TestClient(server.app)
        
        response = client.get("/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "uptime_seconds" in data
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        server = create_server(
            agents=[],
            enable_monitoring=True,
            enable_prometheus=True
        )
        
        client = TestClient(server.app)
        
        response = client.get("/metrics")
        assert response.status_code == 200
        
        # Should return Prometheus format (text/plain)
        assert "text/plain" in response.headers["content-type"]
        
        # Content should contain metric names or placeholder
        content = response.content.decode()
        assert len(content) > 0 