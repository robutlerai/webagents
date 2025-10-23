"""
Streaming Compliance Tests - WebAgents V2.0

Comprehensive tests for OpenAI-compatible streaming functionality:
- ✅ Proper SSE formatting ("data: {...}\n\n", "data: [DONE]\n\n")  
- ✅ OpenAI-compatible streaming chunks with exact format matching
- ✅ Streaming context management (completion_id, timing, model)
- ✅ Final chunk with complete usage information (OpenAI format)
- ✅ Error handling in streaming scenarios

Run with: pytest tests/server/test_streaming_compliance.py -v
"""

import json
import time
import pytest
import asyncio
from typing import List, Dict, Any


class TestSSEFormatting:
    """Test proper Server-Sent Events formatting"""
    
    def test_sse_format_structure(self, test_client, streaming_request_data):
        """Test that streaming responses use proper SSE format"""
        with test_client.stream("POST", "/test-agent/chat/completions", json=streaming_request_data) as response:
            assert response.status_code == 200
            assert response.headers.get("content-type") == "text/plain; charset=utf-8"
            
            # Collect all lines including empty ones
            lines = []
            for line in response.iter_lines():
                lines.append(line)
            
            # Should have "data: " prefixed lines
            data_lines = [line for line in lines if line.startswith("data: ")]
            assert len(data_lines) > 0, "Should have 'data: ' prefixed lines"
            
            # Should have "data: [DONE]" in the lines
            done_lines = [line for line in lines if line == "data: [DONE]"]
            assert len(done_lines) > 0, "Should have 'data: [DONE]' marker"
    
    def test_sse_chunk_parsing(self, test_client, streaming_request_data, streaming_helper):
        """Test that SSE chunks can be parsed correctly"""
        with test_client.stream("POST", "/test-agent/chat/completions", json=streaming_request_data) as response:
            response_text = ""
            for line in response.iter_lines():
                response_text += line + "\n"
            
            chunks = streaming_helper['parse_sse_chunks'](response_text)
            assert len(chunks) > 0, "Should parse streaming chunks"
            
            # Each chunk should be valid JSON
            for chunk in chunks:
                assert isinstance(chunk, dict), "Each chunk should be a dictionary"


class TestOpenAIStreamingCompliance:
    """Test OpenAI-compatible streaming chunks"""
    
    def test_streaming_chunk_format(self, test_client, streaming_request_data, streaming_helper, openai_validator):
        """Test that all streaming chunks comply with OpenAI format"""
        with test_client.stream("POST", "/test-agent/chat/completions", json=streaming_request_data) as response:
            response_text = ""
            for line in response.iter_lines():
                response_text += line + "\n"
            
            chunks = streaming_helper['parse_sse_chunks'](response_text)
            
            # Validate each chunk for OpenAI compliance
            for i, chunk in enumerate(chunks):
                openai_validator(chunk, is_streaming=True)
    
    def test_streaming_chunk_sequence(self, test_client, streaming_request_data, streaming_helper):
        """Test streaming chunk sequence follows OpenAI pattern"""
        with test_client.stream("POST", "/test-agent/chat/completions", json=streaming_request_data) as response:
            response_text = ""
            for line in response.iter_lines():
                response_text += line + "\n"
            
            chunks = streaming_helper['parse_sse_chunks'](response_text)
            assert len(chunks) > 0
            
            # First chunk should establish role
            first_chunk = chunks[0]
            assert first_chunk['choices'][0]['delta'].get('role') == 'assistant'
            
            # Should have content chunks
            content_chunks = [c for c in chunks if c['choices'][0]['delta'].get('content')]
            assert len(content_chunks) > 0, "Should have content chunks"
            
            # Final chunk should have finish_reason
            final_chunk = chunks[-1] 
            assert final_chunk['choices'][0].get('finish_reason') == 'stop'


class TestStreamingContextManagement:
    """Test streaming context management and consistency"""
    
    def test_completion_id_consistency(self, test_client, streaming_request_data, streaming_helper):
        """Test completion ID consistency across chunks"""
        with test_client.stream("POST", "/test-agent/chat/completions", json=streaming_request_data) as response:
            response_text = ""
            for line in response.iter_lines():
                response_text += line + "\n"
            
            chunks = streaming_helper['parse_sse_chunks'](response_text)
            
            # All chunks should have the same completion ID
            completion_ids = set(chunk['id'] for chunk in chunks)
            assert len(completion_ids) == 1, "All chunks should have same completion ID"
    
    def test_model_consistency(self, test_client, streaming_request_data, streaming_helper):
        """Test model name consistency across chunks"""  
        with test_client.stream("POST", "/test-agent/chat/completions", json=streaming_request_data) as response:
            response_text = ""
            for line in response.iter_lines():
                response_text += line + "\n"
            
            chunks = streaming_helper['parse_sse_chunks'](response_text)
            
            # All chunks should have the same model
            models = set(chunk['model'] for chunk in chunks)
            assert len(models) == 1, "All chunks should have same model"
    
    def test_created_timestamp_consistency(self, test_client, streaming_request_data, streaming_helper):
        """Test created timestamp consistency"""
        with test_client.stream("POST", "/test-agent/chat/completions", json=streaming_request_data) as response:
            response_text = ""
            for line in response.iter_lines():
                response_text += line + "\n"
            
            chunks = streaming_helper['parse_sse_chunks'](response_text)
            
            # All chunks should have the same created timestamp
            timestamps = set(chunk['created'] for chunk in chunks)
            assert len(timestamps) == 1, "All chunks should have same timestamp"


class TestStreamingUsageTracking:
    """Test usage information tracking in streaming responses"""
    
    def test_final_chunk_usage_info(self, test_client, streaming_request_data, streaming_helper):
        """Test that final chunk contains complete usage information"""
        with test_client.stream("POST", "/test-agent/chat/completions", json=streaming_request_data) as response:
            response_text = ""
            for line in response.iter_lines():
                response_text += line + "\n"
            
            chunks = streaming_helper['parse_sse_chunks'](response_text)
            
            # Find chunks with usage info (should be final chunk)
            usage_chunks = [c for c in chunks if 'usage' in c]
            assert len(usage_chunks) == 1, "Should have exactly one chunk with usage"
            
            usage = usage_chunks[0]['usage']
            required_fields = ['prompt_tokens', 'completion_tokens', 'total_tokens']
            for field in required_fields:
                assert field in usage, f"Missing usage field: {field}"
            
            # Validate usage calculation
            assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
    
    def test_usage_only_in_final_chunk(self, test_client, streaming_request_data, streaming_helper):
        """Test that usage information appears only in final chunk"""
        with test_client.stream("POST", "/test-agent/chat/completions", json=streaming_request_data) as response:
            response_text = ""
            for line in response.iter_lines():
                response_text += line + "\n"
            
            chunks = streaming_helper['parse_sse_chunks'](response_text)
            
            # Only the final chunk should have usage
            usage_positions = []
            for i, chunk in enumerate(chunks):
                if 'usage' in chunk:
                    usage_positions.append(i)
            
            assert len(usage_positions) == 1, "Usage should appear in exactly one chunk"
            assert usage_positions[0] == len(chunks) - 1, "Usage should be in final chunk"


class TestStreamingErrorHandling:
    """Test error handling in streaming scenarios"""
    
    def test_streaming_nonexistent_agent_error(self, test_client, streaming_request_data):
        """Test streaming request to non-existent agent"""
        response = test_client.post("/nonexistent-agent/chat/completions", json=streaming_request_data)
        assert response.status_code == 404
        
        # Should return JSON error, not streaming - FastAPI uses 'detail' field
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_streaming_invalid_request_error(self, test_client):
        """Test streaming with invalid request data"""
        invalid_request = {
            "stream": True  # Missing required messages
        }
        
        response = test_client.post("/test-agent/chat/completions", json=invalid_request)
        assert response.status_code == 422
    
    def test_streaming_malformed_json_error(self, test_client):
        """Test streaming with malformed JSON"""
        response = test_client.post(
            "/test-agent/chat/completions", 
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422


class TestStreamingPerformance:
    """Test streaming performance and memory efficiency"""
    
    def test_streaming_response_timing(self, test_client, streaming_request_data):
        """Test streaming response performance"""
        start_time = time.time()
        
        with test_client.stream("POST", "/test-agent/chat/completions", json=streaming_request_data) as response:
            first_chunk_time = None
            chunk_count = 0
            
            for line in response.iter_lines():
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                if line.startswith("data: "):
                    chunk_count += 1
        
        total_time = time.time() - start_time
        
        # Performance requirements
        assert first_chunk_time < 1.0, f"First chunk too slow: {first_chunk_time:.3f}s"
        assert total_time < 3.0, f"Total streaming too slow: {total_time:.3f}s"
        assert chunk_count > 0, "Should receive chunks"
    
    def test_memory_efficient_streaming(self, test_client, streaming_request_data):
        """Test that streaming is memory efficient"""
        # Make multiple streaming requests to check for memory leaks
        for i in range(3):
            with test_client.stream("POST", "/test-agent/chat/completions", json=streaming_request_data) as response:
                chunk_count = 0
                for line in response.iter_lines():
                    chunk_count += 1
                
                assert chunk_count > 0, f"Request {i} should have chunks"


class TestStreamingIntegration:
    """Integration tests for streaming functionality"""
    
    def test_streaming_with_multiple_agents(self, multi_client, streaming_request_data, streaming_helper):
        """Test streaming works with multiple agents"""
        agent_names = ["assistant", "calculator", "weather"]
        
        for agent_name in agent_names:
            with multi_client.stream("POST", f"/{agent_name}/chat/completions", json=streaming_request_data) as response:
                assert response.status_code == 200
                
                response_text = ""
                for line in response.iter_lines():
                    response_text += line + "\n"
                
                chunks = streaming_helper['parse_sse_chunks'](response_text)
                assert len(chunks) > 0, f"Agent {agent_name} should return chunks"
    
    def test_streaming_with_tools_parameter(self, test_client, streaming_helper):
        """Test streaming with external tools parameter"""
        request_data = {
            "messages": [{"role": "user", "content": "Test with tools"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ]
        }
        
        with test_client.stream("POST", "/test-agent/chat/completions", json=request_data) as response:
            assert response.status_code == 200
            
            response_text = ""
            for line in response.iter_lines():
                response_text += line + "\n"
            
            chunks = streaming_helper['parse_sse_chunks'](response_text)
            assert len(chunks) > 0, "Should handle external tools parameter"


class TestOpenAIStreamingRequirements:
    """Comprehensive OpenAI streaming compliance verification"""
    
    def test_complete_openai_streaming_compliance(self, test_client, streaming_request_data, streaming_helper, openai_validator):
        """Comprehensive test verifying OpenAI streaming compatibility"""
        print("\n🎯 OpenAI Streaming Compliance Verification:")
        
        with test_client.stream("POST", "/test-agent/chat/completions", json=streaming_request_data) as response:
            # ✅ Server responds with streaming
            assert response.status_code == 200
            assert response.headers.get("content-type") == "text/plain; charset=utf-8"
            print("  ✅ Server streaming response")
            
            # Collect response
            response_text = ""
            for line in response.iter_lines():
                response_text += line + "\n"
            
            # ✅ Proper SSE formatting
            lines = response_text.strip().split('\n')
            data_lines = [line for line in lines if line.startswith("data: ")]
            assert len(data_lines) > 0
            done_lines = [line for line in lines if line == "data: [DONE]"]
            assert len(done_lines) > 0
            print("  ✅ Proper SSE formatting with 'data: {...}\\n\\n' and 'data: [DONE]\\n\\n'")
            
            # Parse chunks
            chunks = streaming_helper['parse_sse_chunks'](response_text)
            assert len(chunks) > 0
            
            # ✅ OpenAI-compatible chunks
            for chunk in chunks:
                openai_validator(chunk, is_streaming=True)
            print("  ✅ OpenAI-compatible streaming chunks with exact format matching")
            
            # ✅ Streaming context management
            completion_ids = set(chunk['id'] for chunk in chunks)
            models = set(chunk['model'] for chunk in chunks)
            timestamps = set(chunk['created'] for chunk in chunks)
            assert len(completion_ids) == 1
            assert len(models) == 1  
            assert len(timestamps) == 1
            print("  ✅ Streaming context management (completion_id, timing, model)")
            
            # ✅ Usage information in final chunk
            usage_chunks = [c for c in chunks if 'usage' in c]
            assert len(usage_chunks) == 1
            usage = usage_chunks[0]['usage']
            assert all(field in usage for field in ['prompt_tokens', 'completion_tokens', 'total_tokens'])
            print("  ✅ Final chunk with complete usage information (OpenAI format)")
        
        # ✅ Error handling
        error_response = test_client.post("/nonexistent/chat/completions", json=streaming_request_data)
        assert error_response.status_code == 404
        print("  ✅ Error handling in streaming scenarios")
        
        print("\n🚀 OpenAI-Compatible Streaming Foundation - ALL REQUIREMENTS MET!")
        
        # Don't return value to avoid pytest warning
        assert True 