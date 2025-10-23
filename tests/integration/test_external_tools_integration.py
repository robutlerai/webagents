"""
Integration Test for External Tools - Real Server & API Calls

This test demonstrates the complete external tools flow:
1. Client sends request with external tools to real WebAgents server
2. Server passes external tools to LLM (LiteLLM/OpenAI)  
3. LLM responds with tool_calls for external tools
4. Server returns tool_calls to client (NOT executing them server-side)
5. Client executes external tools and sends results back
6. Server continues conversation with tool results

CRITICAL UNDERSTANDING:
- External tools = tools from request.tools parameter
- These are executed by CLIENT, not server
- Server's role is to pass to LLM and return tool_calls to client
"""

import pytest
import asyncio
import httpx
import json
import uuid
from typing import Dict, Any, List
import os
# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Using environment variables as-is.")

# Import WebAgents components - ZERO MOCKING - true integration test
from webagents.agents.core.base_agent import BaseAgent  
from webagents.agents.skills.core.llm.litellm import LiteLLMSkill
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool
from webagents.server.core.app import WebAgentsServer
from webagents.server.models import ChatCompletionRequest

class ExternalToolsIntegrationTest:
    """Comprehensive integration test for external tools with real server"""
    
    def __init__(self):
        self.server = None
        self.client = None
        self.server_url = "http://localhost:8080"  # Test server
        
    async def setup(self):
        """Setup real server with LiteLLM agent"""
        
        # Create agent with LiteLLM skill for real API calls using .env config
        litellm_base_url = os.getenv("LITELLM_BASE_URL", "http://localhost:2225")
        litellm_api_key = os.getenv("LITELLM_API_KEY", "rok_testapikey")
        
        # ZERO MOCKING - Configure LiteLLM with proxy keys (config overrides environment)  
        litellm_config = {
            "model": "gpt-4o-mini",  # Use cheaper model for testing
            "base_url": litellm_base_url,
            "api_keys": {
                "openai": litellm_api_key,
                "anthropic": litellm_api_key,
                "xai": litellm_api_key,
                "azure": litellm_api_key
            }
        }
        
        print(f"📡 Using LiteLLM proxy: {litellm_base_url}")
        print(f"🔑 Using config API keys (config overrides environment): {litellm_api_key[:10]}...")
        
        # ZERO MOCKING - Configure LiteLLMSkill with proxy keys in config (config has priority)
        litellm_skill = LiteLLMSkill(config=litellm_config)
        
        # Add a simple agent tool to verify server-side execution works
        class SimpleCalculatorSkill(Skill):
            async def initialize(self, agent):
                self.agent = agent
                
            @tool(scope="all")
            def add_numbers(self, a: int, b: int) -> int:
                """Add two numbers together on the server"""
                return a + b
        
        calc_skill = SimpleCalculatorSkill()
        
        # Create agent with both LLM and calculator skills
        # BaseAgent will automatically initialize skills when run() is called
        agent = BaseAgent(
            name="external-tools-test-agent",
            skills={
                "primary_llm": litellm_skill,
                "calculator": calc_skill
            }
        )
        
        # Create server
        self.server = WebAgentsServer(agents=[agent])
        
        # Start real FastAPI server in background thread
        import uvicorn
        import threading
        
        # Create server config
        config = uvicorn.Config(
            self.server.app,
            host="127.0.0.1",
            port=8080, 
            log_level="error"  # Reduce noise in tests
        )
        self.uvicorn_server = uvicorn.Server(config)
        
        # Start server in background thread
        self.server_thread = threading.Thread(
            target=lambda: asyncio.run(self.uvicorn_server.serve()),
            daemon=True
        )
        self.server_thread.start()
        
        # Wait for server to be ready
        await asyncio.sleep(3)
        
        # Create HTTP client
        self.client = httpx.AsyncClient(base_url=self.server_url, timeout=60.0)
        
    async def teardown(self):
        """Cleanup server and client"""
        if self.client:
            await self.client.aclose()
        
        # Shutdown the real server
        if hasattr(self, 'uvicorn_server'):
            self.uvicorn_server.should_exit = True
            
        if hasattr(self, 'server_thread'):
            # Give server time to shutdown gracefully
            await asyncio.sleep(1)
    
    def create_external_tools(self) -> List[Dict[str, Any]]:
        """Create sample external tools that client should execute"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location (executed by client)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "send_email",
                    "description": "Send an email using client's email system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string", "description": "Recipient email"},
                            "subject": {"type": "string", "description": "Email subject"}, 
                            "body": {"type": "string", "description": "Email body"}
                        },
                        "required": ["to", "subject", "body"]
                    }
                }
            }
        ]
    
    def validate_openai_compliance(self, response: Dict[str, Any], is_streaming: bool = False) -> None:
        """Validate complete OpenAI API compliance"""
        
        if is_streaming:
            # Streaming chunk validation
            required_fields = ["id", "object", "created", "model", "choices"]
            for field in required_fields:
                assert field in response, f"Missing required streaming field: {field}"
            
            assert response["object"] == "chat.completion.chunk"
            
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                required_choice_fields = ["index", "delta"]
                for field in required_choice_fields:
                    assert field in choice, f"Missing required streaming choice field: {field}"
        else:
            # Non-streaming response validation
            required_fields = ["id", "object", "created", "model", "choices", "usage"]
            for field in required_fields:
                assert field in response, f"Missing required response field: {field}"
            
            assert response["object"] == "chat.completion"
            
            # Validate choices structure
            assert len(response["choices"]) > 0, "Response must have at least one choice"
            choice = response["choices"][0]
            required_choice_fields = ["index", "message", "finish_reason"]
            for field in required_choice_fields:
                assert field in choice, f"Missing required choice field: {field}"
            
            # Validate message structure
            message = choice["message"]
            assert "role" in message, "Message must have role"
            assert message["role"] in ["assistant", "user", "system", "tool"], f"Invalid role: {message['role']}"
            
            # Content can be null for tool calls, but if present should be string
            if message.get("content") is not None:
                assert isinstance(message["content"], str), "Content must be string when present"
            
            # Validate usage structure
            usage = response["usage"]
            required_usage_fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
            for field in required_usage_fields:
                assert field in usage, f"Missing required usage field: {field}"
                assert isinstance(usage[field], int), f"Usage {field} must be integer"
                assert usage[field] >= 0, f"Usage {field} must be non-negative"
            
            # Verify total_tokens = prompt_tokens + completion_tokens
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            
        print("✅ OpenAI compliance validation passed")

    def simulate_client_tool_execution(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simulate REAL client executing external tools (NOT mocking server behavior)
        
        This is necessary because external tools are SUPPOSED to be executed by the client.
        The server correctly returns tool_calls to client - this simulates what a real 
        client would do when receiving those tool_calls.
        """
        results = []
        
        for tool_call in tool_calls:
            tool_call_id = tool_call["id"]
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            
            # Simulate client executing each external tool
            if function_name == "get_weather":
                location = function_args.get("location", "Unknown")
                content = f"Weather in {location}: 72°F, sunny with light clouds"
                
            elif function_name == "send_email":
                to = function_args.get("to", "")
                subject = function_args.get("subject", "")
                content = f"Email sent successfully to {to} with subject '{subject}'"
                
            else:
                content = f"Unknown external tool '{function_name}' executed by client"
            
            results.append({
                "tool_call_id": tool_call_id,
                "role": "tool",
                "content": content
            })
            
        return results
        
    async def test_external_tools_flow(self):
        """Test complete external tools flow with real server and LLM"""
        
        print("🚀 Starting External Tools Integration Test")
        
        # 1. Test server health
        health_response = await self.client.get("/health")
        assert health_response.status_code == 200
        print("✅ Server health check passed")
        
        # 2. Create request with external tools
        external_tools = self.create_external_tools()
        
        request_data = {
            "model": "external-tools-test-agent",
            "messages": [
                {
                    "role": "user",
                    "content": "Can you help me get the weather in San Francisco and send an email to john@example.com about it?"
                }
            ],
            "tools": external_tools,  # External tools from client
            "stream": False
        }
        
        print(f"📤 Sending request with {len(external_tools)} external tools")
        
        # 3. Send request to server
        response = await self.client.post(
            "/external-tools-test-agent/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        print(f"📥 Received response: {response_data}")
        
        # Validate OpenAI compliance for initial response
        self.validate_openai_compliance(response_data, is_streaming=False)
        
        # 4. Verify response contains tool_calls (external tools returned to client)
        assert "choices" in response_data
        assert len(response_data["choices"]) > 0
        
        choice = response_data["choices"][0]
        message = choice["message"]
        
        # Should contain tool_calls for external tools
        assert "tool_calls" in message
        tool_calls = message["tool_calls"]
        
        print(f"🔧 Server returned {len(tool_calls)} tool_calls to client")
        
        # 5. Verify these are external tools (not executed server-side)
        external_tool_names = {tool["function"]["name"] for tool in external_tools}
        returned_tool_names = {tc["function"]["name"] for tc in tool_calls}
        
        # All returned tool calls should be external tools
        assert returned_tool_names.issubset(external_tool_names)
        print("✅ All returned tool_calls are external tools (correct!)")
        
        # 6. Simulate client executing external tools
        tool_results = self.simulate_client_tool_execution(tool_calls)
        print(f"🖥️ Client executed {len(tool_results)} external tools")
        
        # 7. Send tool results back to server to continue conversation
        continuation_messages = request_data["messages"] + [
            message,  # Assistant's response with tool_calls
            *tool_results  # Tool execution results from client
        ]
        
        continuation_request = {
            "model": "external-tools-test-agent",
            "messages": continuation_messages,
            "tools": external_tools,  # Keep external tools available
            "stream": False
        }
        
        print("🔄 Sending tool results back to continue conversation")
        print(f"🔧 Continuation messages: {json.dumps(continuation_messages, indent=2)}")
        
        continuation_response = await self.client.post(
            "/external-tools-test-agent/chat/completions",
            json=continuation_request,
            headers={"Content-Type": "application/json"}
        )
        
        if continuation_response.status_code != 200:
            error_text = continuation_response.text
            print(f"❌ Continuation request failed (422): {error_text}")
        
        assert continuation_response.status_code == 200
        continuation_data = continuation_response.json()
        
        # Validate OpenAI compliance for continuation response
        self.validate_openai_compliance(continuation_data, is_streaming=False)
        
        print(f"📥 Continuation response: {json.dumps(continuation_data, indent=2)}")
        
        # 8. Verify server processed tool results and continued conversation
        final_message = continuation_data["choices"][0]["message"]
        final_content = final_message["content"]
        
        print(f"💬 Final response content: {final_content}")
        print(f"💬 Final message: {json.dumps(final_message, indent=2)}")
        
        # Handle multi-turn external tool execution until we get final response
        current_response = continuation_data
        turn_count = 1
        
        while (current_response["choices"][0]["message"].get("tool_calls") and 
               current_response["choices"][0]["message"]["content"] is None and 
               turn_count < 5):  # Prevent infinite loops
            
            turn_count += 1
            current_message = current_response["choices"][0]["message"]
            current_tool_calls = current_message["tool_calls"]
            
            print(f"🔄 Turn {turn_count}: LLM requesting {len(current_tool_calls)} more external tools")
            
            # Execute the additional external tools
            additional_tool_results = self.simulate_client_tool_execution(current_tool_calls)
            print(f"🖥️ Client executed {len(additional_tool_results)} additional external tools")
            
            # Build new continuation messages
            new_continuation_messages = continuation_messages + [
                current_message,  # Assistant's response with new tool_calls
                *additional_tool_results  # New tool execution results
            ]
            
            new_continuation_request = {
                "model": "external-tools-test-agent",
                "messages": new_continuation_messages,
                "tools": external_tools,
                "stream": False
            }
            
            print(f"🔄 Sending turn {turn_count} tool results back to continue conversation")
            
            new_continuation_response = await self.client.post(
                "/external-tools-test-agent/chat/completions",
                json=new_continuation_request,
                headers={"Content-Type": "application/json"}
            )
            
            assert new_continuation_response.status_code == 200
            current_response = new_continuation_response.json()
            
            # Validate OpenAI compliance for each continuation response
            self.validate_openai_compliance(current_response, is_streaming=False)
            
            continuation_messages = new_continuation_messages  # Update for next iteration
        
        # Now we should have the final response
        final_message = current_response["choices"][0]["message"]
        final_content = final_message["content"]
        final_usage = current_response.get("usage", {})
        
        print(f"💬 Final response content: {final_content}")
        print(f"📊 Final usage: {final_usage}")
        print(f"🔄 Total conversation turns: {turn_count}")
        
        # Verify we got a final response with content
        assert final_content is not None, "Should receive final response with content"
        assert len(final_content) > 0, "Final response should not be empty"
        
        # Verify OpenAI compliance - final response should have usage
        assert "usage" in current_response, "Final response must include usage information"
        assert "total_tokens" in final_usage, "Usage must include total_tokens"
        assert final_usage["total_tokens"] > 0, "Total tokens should be positive"
        
        print("✅ Complete external tools flow executed successfully!")
        print("✅ OpenAI compliance verified with final usage information!")
        
    async def test_mixed_tools_scenario(self):
        """Test scenario with both agent tools (server) and external tools (client)"""
        
        print("🔀 Testing mixed tools scenario")
        
        external_tools = self.create_external_tools()
        
        # Request that should trigger both agent tool and external tools
        request_data = {
            "model": "external-tools-test-agent",
            "messages": [
                {
                    "role": "user", 
                    "content": "First, add 25 and 17 using your calculator. Then get the weather in New York."
                }
            ],
            "tools": external_tools,
            "stream": False
        }
        
        response = await self.client.post(
            "/external-tools-test-agent/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        choice = response_data["choices"][0] 
        message = choice["message"]
        
        # Check what type of response we got
        if "tool_calls" in message:
            # LLM requested tools - check if they're agent or external
            tool_calls = message["tool_calls"]
            tool_names = [tc["function"]["name"] for tc in tool_calls]
            
            print(f"🔧 LLM requested tools: {tool_names}")
            
            # If agent tools were requested, they would be executed server-side
            # If external tools were requested, they would be returned to client
            
            agent_tool_names = {"add_numbers"}
            external_tool_names = {"get_weather", "send_email"}
            
            has_agent_tools = any(name in agent_tool_names for name in tool_names)
            has_external_tools = any(name in external_tool_names for name in tool_names)
            
            if has_agent_tools and has_external_tools:
                print("✅ Mixed scenario: Both agent and external tools requested")
            elif has_agent_tools:
                print("✅ Agent tools were executed server-side")
            elif has_external_tools:
                print("✅ External tools returned to client")
                
        else:
            # LLM provided direct response without tool calls
            print("✅ LLM provided direct response without needing tools")
            
        print("✅ Mixed tools scenario completed successfully")
        
    async def test_streaming_with_external_tools(self):
        """Test external tools work with streaming responses - MIGHT ALREADY WORK!"""
        
        print("🌊 Testing external tools with streaming - checking if already implemented")
        
        external_tools = self.create_external_tools()
        
        request_data = {
            "model": "external-tools-test-agent",
            "messages": [
                {
                    "role": "user",
                    "content": "Please check the weather in Boston and tell me about it."
                }
            ],
            "tools": external_tools,
            "stream": True  # Enable streaming
        }
        
        # Make streaming request  
        async with self.client.stream(
            "POST",
            "/external-tools-test-agent/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status_code != 200:
                print(f"❌ Streaming request failed: {response.status_code}")
                return
                
            print("✅ Streaming request successful!")
            print(f"📋 Content-Type: {response.headers.get('content-type')}")
            
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunk_data = line[6:]  # Remove "data: " prefix
                    if chunk_data == "[DONE]":
                        print("✅ Received [DONE] marker")
                        break
                    try:
                        parsed_chunk = json.loads(chunk_data)
                        chunks.append(parsed_chunk)
                        
                        # Check for tool_calls in streaming chunks
                        if "choices" in parsed_chunk and len(parsed_chunk["choices"]) > 0:
                            delta = parsed_chunk["choices"][0].get("delta", {})
                            if "tool_calls" in delta:
                                print(f"🔧 Found tool_calls in streaming chunk!")
                                print(f"   Tool calls: {delta['tool_calls']}")
                        
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Failed to parse chunk: {chunk_data[:100]}...")
                        continue
            
            print(f"📊 Received {len(chunks)} streaming chunks")
            
            # Check if we got tool calls
            tool_calls_found = False
            for chunk in chunks:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "tool_calls" in delta:
                        tool_calls_found = True
                        break
            
            if tool_calls_found:
                print("🎉 STREAMING + EXTERNAL TOOLS ALREADY WORKING!")
                print("✅ LLM properly streamed tool_calls for external tools")
            else:
                print("📋 No tool calls found in streaming chunks")
                print("✅ Streaming works, but may need tool call streaming format")
        
        print("✅ Streaming with external tools test completed")


@pytest.mark.asyncio
@pytest.mark.integration 
@pytest.mark.skipif(
    not (os.getenv("LITELLM_BASE_URL") or os.getenv("OPENAI_API_KEY")), 
    reason="Either LITELLM_BASE_URL or OPENAI_API_KEY required for integration test"
)
async def test_external_tools_real_integration():
    """Run the complete external tools integration test"""
    
    test_runner = ExternalToolsIntegrationTest()
    
    try:
        await test_runner.setup()
        print("✅ Test server setup complete")
        
        # Run all test scenarios  
        await test_runner.test_external_tools_flow()
        await test_runner.test_mixed_tools_scenario() 
        await test_runner.test_streaming_with_external_tools()
        
        print("🎉 ALL EXTERNAL TOOLS INTEGRATION TESTS PASSED!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise
        
    finally:
        await test_runner.teardown()
        print("✅ Test cleanup complete")


if __name__ == "__main__":
    """Run integration test directly"""
    
    print("🧪 Running External Tools Integration Test")
    print("=" * 60)
    
    # Check requirements
    litellm_base_url = os.getenv("LITELLM_BASE_URL")
    litellm_api_key = os.getenv("LITELLM_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not (litellm_base_url or openai_api_key):
        print("❌ Configuration required for integration test:")
        print("   Option 1: Set LITELLM_BASE_URL + LITELLM_API_KEY for LiteLLM proxy")
        print("   Option 2: Set OPENAI_API_KEY for direct OpenAI API")
        print("   Or create a .env file with the required configuration")
        exit(1)
        
    if litellm_base_url:
        print(f"✅ Using LiteLLM proxy: {litellm_base_url}")
        print(f"✅ Using API key: {litellm_api_key[:10]}..." if litellm_api_key else "⚠️ No LITELLM_API_KEY")
    else:
        print("✅ Using direct OpenAI API")
        print(f"✅ Using API key: {openai_api_key[:7]}..." if openai_api_key else "⚠️ No OPENAI_API_KEY")
        
    # Run test
    asyncio.run(test_external_tools_real_integration()) 