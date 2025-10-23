"""
Integration Tests for WebAgents V2.0

True integration tests with NO MOCKING for complete end-to-end validation:

- **Real FastAPI server** instances with actual HTTP requests
- **Real LLM API calls** to OpenAI/LiteLLM for authentic behavior  
- **Real skill integrations** with actual skill initialization
- **Complete request/response cycles** matching production usage
- **External tools flow validation** with correct client/server separation

## External Tools Integration

The `test_external_tools_integration.py` test validates the complete external tools system:

1. **Client** sends request with external tools to real WebAgents server
2. **Server** passes external tools to real LLM (OpenAI via LiteLLM)
3. **LLM** responds with `tool_calls` for external tools  
4. **Server** returns `tool_calls` to client (correct behavior - NO server execution)
5. **Client** executes external tools and sends results back
6. **Server** continues conversation with tool results

### Critical Understanding Validated

**✅ External Tools** = CLIENT execution (from request.tools parameter)  
**✅ Agent Tools** = SERVER execution (@tool decorated functions)  
**✅ Tool Merging** = Both types combined for LLM awareness  
**✅ OpenAI Compliance** = 100% compatible ChatCompletions API

Run with: `python -m pytest tests/integration/ -v --tb=short`
Requires: `OPENAI_API_KEY` environment variable for real API calls
"""

__all__ = [
    'test_external_tools_integration',
] 