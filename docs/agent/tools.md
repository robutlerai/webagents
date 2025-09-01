# Agent Tools

Tools extend agent capabilities with executable functions. There are two types: **internal tools** and **external tools**. Internal tools are Python functions the agent can call directly; external tools follow OpenAI's tool-calling protocol and are executed by the client.

## Tool Types

### Internal Tools

Internal tools are executed within the agent's process. They can be:

1. **Skill Tools** - Defined in skills using `@tool` decorator
2. **Standalone Tools** - Decorated functions passed to agent

### External Tools

External tools are defined in the request and executed on the client side. The agent will emit OpenAI tool calls; your client is responsible for executing them and returning results in a follow-up message. This keeps server responsibilities minimal while remaining compatible with OpenAI tooling.

!!! info "HTTP Endpoints"

    For creating custom HTTP API endpoints, see **[Agent Endpoints](endpoints.md)** which covers the `@http` decorator and REST API creation.

## Internal Tools

### Standalone Tools

```python
from webagents.agents.tools.decorators import tool
from webagents.agents import BaseAgent

# Define standalone tool functions
@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expressions"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except:
        return "Invalid expression"

@tool(scope="owner")
def admin_function(action: str) -> str:
    """Owner-only administrative function"""
    return f"Admin action: {action}"

# Pass to agent
agent = BaseAgent(
    name="my-agent",
    model="openai/gpt-4o",
    tools=[calculate, admin_function]  # Internal tools
)
```



### Capabilities Auto-Registration

Use the `capabilities` parameter to automatically register decorated functions:

```python
from webagents.agents.tools.decorators import tool, hook, handoff

@tool(scope="owner")
def my_tool(message: str) -> str:
    return f"Tool: {message}"

@tool
def another_tool(data: str) -> str:
    return f"Processed: {data}"

@hook("on_request", priority=10)
def my_hook(context):
    return context

@handoff(handoff_type="agent")
def my_handoff(target: str):
    return HandoffResult(result=f"Handoff to {target}")

# Auto-register all decorated functions
agent = BaseAgent(
    name="capable-agent",
    model="openai/gpt-4o",
    capabilities=[my_tool, another_tool, my_hook, my_handoff]
)
```

The agent will automatically categorize and register each function based on its decorator type. Tools will be registered as callable functions for the LLM.

### Skill Tools

```python
from webagents.agents.skills import Skill
from webagents.agents.tools.decorators import tool

class CalculatorSkill(Skill):
    @tool
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        return a + b
    
    @tool(scope="owner")
    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers (owner only)"""
        return x * y
```


### Tool Parameters

```python
@tool(
    name="custom_name",      # Override function name
    description="Custom",    # Override docstring
    scope="all",            # Access control: all/owner/admin
    # For priced tools, prefer the PaymentSkill's @pricing decorator; this field is descriptive only
)
def my_tool(param: str) -> str:
    """Tool implementation"""
    return f"Result: {param}"
```

## OpenAI Schema Generation

Tools automatically generate OpenAI-compatible schemas:

```python
@tool
def search_web(query: str, max_results: int = 10) -> List[str]:
    """Search the web for information
    
    Args:
        query: Search query string
        max_results: Maximum results to return
    
    Returns:
        List of search results
    """
    return ["result1", "result2"]

# Generates schema:
{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
}
```

## External Tools

External tools are defined in the request's `tools` parameter and executed on the requester's side. They follow the standard OpenAI tool definition format.

### Standard OpenAI Tool Definition Format

External tools use the standard OpenAI format:

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "function_name",
        "description": "Function description",
        "parameters": {
          "type": "object",
          "properties": {
            "param_name": {
              "type": "string",
              "description": "Parameter description"
            }
          },
          "required": ["param_name"]
        }
      }
    }
  ]
}
```

### Using External Tools

```python
# Define external tools in the request
external_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "description": "Temperature unit (celsius or fahrenheit)",
                        "enum": ["celsius", "fahrenheit"]
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
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body content"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]

# Pass tools in the request
messages = [{"role": "user", "content": "What's the weather in Paris?"}]
response = await agent.run(messages=messages, tools=external_tools)
```

### Handling Tool Calls

When the agent makes tool calls, you receive them in the response and execute them client-side:

```python
# Agent response with tool calls
response = await agent.run(messages=messages, tools=external_tools)
assistant_message = response.choices[0].message

if assistant_message.tool_calls:
    # Execute each tool call
    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Execute the tool based on name
        if function_name == "get_weather":
            result = get_weather_external(arguments["location"])
        elif function_name == "send_email":
            result = send_email_external(
                arguments["to"], 
                arguments["subject"], 
                arguments["body"]
            )
        
        # Add tool result to conversation
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [tool_call]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
        
        # Get final response
        final_response = await agent.run(messages=messages, tools=external_tools)
        return final_response.choices[0].message.content

def get_weather_external(location: str) -> str:
    """Your implementation of the external weather tool"""
    # Your weather API call here
    return f"Sunny in {location}, 22°C"

def send_email_external(to: str, subject: str, body: str) -> str:
    """Your implementation of the external email tool"""
    # Your email sending logic here
    return f"Email sent to {to}"
```

## Tool Execution

### Automatic Tool Calling

```python
# Agent automatically calls tools when needed
user_msg = "What's the weather in Paris?"
response = await agent.run([
    {"role": "user", "content": user_msg}
])
# Agent calls get_weather("Paris") automatically
```

### Manual Tool Results

```python
# Include tool results in conversation
messages = [
    {"role": "user", "content": "Calculate 42 * 17"},
    {"role": "assistant", "content": "I'll calculate that for you.", 
     "tool_calls": [{
         "id": "call_123",
         "type": "function",
         "function": {"name": "multiply", "arguments": '{"x": 42, "y": 17}'}
     }]},
    {"role": "tool", "tool_call_id": "call_123", "content": "714"}
]
response = await agent.run(messages)
```

## Advanced Tool Features

### Dynamic Tool Registration

```python
class AdaptiveSkill(Skill):
    @hook("on_connection")
    async def register_dynamic_tools(self, context):
        """Register tools based on context"""
        
        if context.peer_user_id == "admin":
            # Register admin tools
            self.register_tool(self.admin_tool, scope="admin")
        
        if "math" in str(context.messages):
            # Register math tools
            self.register_tool(self.advanced_calc)
        
        return context
    
    def admin_tool(self, action: str) -> str:
        """Admin-only tool"""
        return f"Admin action: {action}"
```

### Tool Middleware

```python
class ToolMonitor(Skill):
    @hook("before_toolcall", priority=1)
    async def validate_tool(self, context):
        """Validate before execution"""
        tool_name = context["tool_call"]["function"]["name"]
        
        # Rate limiting
        if self.is_rate_limited(tool_name):
            raise RateLimitError(f"Tool {tool_name} rate limited")
        
        # Parameter validation
        args = json.loads(context["tool_call"]["function"]["arguments"])
        self.validate_args(tool_name, args)
        
        return context
    
    @hook("after_toolcall", priority=90)
    async def log_result(self, context):
        """Log tool execution"""
        await self.log_tool_usage(
            tool=context["tool_call"]["function"]["name"],
            result=context["tool_result"],
            duration=context.get("tool_duration")
        )
        return context
```

### Tool Pricing

```python
from webagents.agents.tools.decorators import tool, pricing

class PaidToolsSkill(Skill):
    @tool
    @pricing(cost=0.10, currency="USD")
    def expensive_api_call(self, query: str) -> str:
        """Call expensive external API"""
        # Automatically tracks usage for billing
        return self.call_paid_api(query)
    
    @tool
    @pricing(cost=0.01, per="request")
    def database_query(self, sql: str) -> List[Dict]:
        """Execute database query"""
        return self.execute_sql(sql)
```

## Tool Patterns

### Validation Pattern

```python
@tool
def update_record(self, record_id: str, data: Dict) -> Dict:
    """Update record with validation"""
    # Validate inputs
    if not self.validate_record_id(record_id):
        return {"error": "Invalid record ID"}
    
    if not self.validate_data(data):
        return {"error": "Invalid data format"}
    
    # Perform update
    try:
        result = self.db.update(record_id, data)
        return {"success": True, "record": result}
    except Exception as e:
        return {"error": str(e)}
```

### Async Pattern

```python
@tool
async def fetch_data(self, urls: List[str]) -> List[Dict]:
    """Fetch data from multiple URLs concurrently"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        tasks = [self.fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    
    return results
```

### Caching Pattern

```python
class CachedToolsSkill(Skill):
    def __init__(self, config=None):
        super().__init__(config)
        self.cache = {}
    
    @tool
    def expensive_calculation(self, input: str) -> str:
        """Cached expensive calculation"""
        if input in self.cache:
            return self.cache[input]
        
        result = self.perform_calculation(input)
        self.cache[input] = result
        return result
```

## Best Practices

1. **Clear Descriptions** - Help LLM understand when to use tools
2. **Type Hints** - Enable automatic schema generation
3. **Error Handling** - Return errors as data, not exceptions
4. **Scope Control** - Use appropriate access levels
5. **Performance** - Consider caching and async execution 