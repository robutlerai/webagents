---
title: Agent Tools
description: Internal and external tools — how to define them, how the agent invokes them, and how clients participate in OpenAI-style tool calling.
---

# Agent Tools

Tools extend agent capabilities with executable functions. There are two types: **internal tools** and **external tools**. Internal tools live inside the agent process; external tools follow OpenAI's tool-calling protocol and are executed by the client.

## Tool Types

### Internal Tools

Internal tools are executed within the agent's process. They can be:

1. **Skill tools** — defined in skills using the `@tool` decorator.
2. **Standalone tools** — decorated functions passed directly to the agent.

### External Tools

External tools are defined in the request and executed on the client side. The agent emits OpenAI tool calls; your client is responsible for executing them and returning results in a follow-up message. This keeps server responsibilities minimal while remaining compatible with OpenAI tooling.

> For creating custom HTTP API endpoints, see [Agent Endpoints](./endpoints.md), which covers the `@http` decorator and REST API creation.

## Internal Tools

### Standalone Tools

```typescript tab="TypeScript"
import { BaseAgent, tool, Skill } from 'webagents';

class CalculatorTools extends Skill {
  readonly name = 'calculator-tools';

  @tool({ description: 'Calculate mathematical expressions' })
  async calculate(params: { expression: string }): Promise<string> {
    try {
      const result = Function(`"use strict"; return (${params.expression})`)();
      return String(result);
    } catch {
      return 'Invalid expression';
    }
  }

  @tool({ description: 'Owner-only administrative function', scopes: ['owner'] })
  async adminFunction(params: { action: string }): Promise<string> {
    return `Admin action: ${params.action}`;
  }
}

const agent = new BaseAgent({
  name: 'my-agent',
  model: 'openai/gpt-4o',
  skills: [new CalculatorTools()],
});
```

```python tab="Python"
from webagents import BaseAgent, tool

@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expressions"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return "Invalid expression"

@tool(scope="owner")
def admin_function(action: str) -> str:
    """Owner-only administrative function"""
    return f"Admin action: {action}"

agent = BaseAgent(
    name="my-agent",
    model="openai/gpt-4o",
    tools=[calculate, admin_function],  # Internal tools
)
```

### Skill Tools

```typescript tab="TypeScript"
import { Skill, tool } from 'webagents';

class CalculatorSkill extends Skill {
  readonly name = 'calculator';

  @tool({ description: 'Add two numbers' })
  async add(params: { a: number; b: number }): Promise<number> {
    return params.a + params.b;
  }

  @tool({ description: 'Multiply two numbers (owner only)', scopes: ['owner'] })
  async multiply(params: { x: number; y: number }): Promise<number> {
    return params.x * params.y;
  }
}
```

```python tab="Python"
from webagents import Skill, tool

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

```typescript tab="TypeScript"
@tool({
  name: 'custom_name',          // Override method name
  description: 'Custom',        // Description for the LLM
  scopes: ['all'],              // Access control: 'all' | 'owner' | 'admin' | …
  provides: 'chart',            // Capability this tool provides (for discovery)
  parameters: { /* JSON Schema */ },
})
async myTool(params: { value: string }): Promise<string> {
  return `Result: ${params.value}`;
}
```

```python tab="Python"
@tool(
    name="custom_name",      # Override function name
    description="Custom",    # Override docstring
    scope="all",             # Access control: all/owner/admin
    provides="chart",        # Capability this tool provides (for discovery)
)
def my_tool(param: str) -> str:
    """Tool implementation"""
    return f"Result: {param}"
```

#### The `provides` field

The `provides` field declares what capability a tool provides. This is used for:

- **Agent capability discovery** — Clients can query what an agent can do.
- **UAMP capabilities** — Exposed in `Capabilities.provides` for agent-to-agent communication.

```typescript tab="TypeScript"
@tool({ provides: 'web_search', description: 'Search the web for information' })
async searchWeb(params: { query: string }): Promise<string> { /* ... */ return ''; }

@tool({ provides: 'chart', description: 'Render data as a chart widget' })
async renderChart(params: { data: string }): Promise<string> { /* ... */ return ''; }

@tool({ provides: 'tts', description: 'Convert text to speech audio' })
async textToSpeech(params: { text: string }): Promise<Uint8Array> { /* ... */ return new Uint8Array(); }
```

```python tab="Python"
@tool(provides="web_search")
async def search_web(query: str) -> str:
    """Search the web for information."""
    ...

@tool(provides="chart")
async def render_chart(data: str) -> str:
    """Render data as a chart widget."""
    ...

@tool(provides="tts")
async def text_to_speech(text: str) -> bytes:
    """Convert text to speech audio."""
    ...
```

The agent aggregates all `provides` values from tools, handoffs, and endpoints into its capabilities.

## OpenAI Schema Generation

Tools generate OpenAI-compatible schemas automatically. In Python the schema is derived from type hints and the docstring; in TypeScript pass an explicit `parameters` object (JSON Schema) when richer descriptions are needed.

```typescript tab="TypeScript"
@tool({
  description: 'Search the web for information',
  parameters: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'Search query string' },
      max_results: { type: 'integer', description: 'Maximum results', default: 10 },
    },
    required: ['query'],
  },
})
async searchWeb(params: { query: string; max_results?: number }): Promise<string[]> {
  return ['result1', 'result2'];
}
```

```python tab="Python"
from typing import List

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
```

Both produce the same schema:

```json
{
  "type": "function",
  "function": {
    "name": "search_web",
    "description": "Search the web for information",
    "parameters": {
      "type": "object",
      "properties": {
        "query": { "type": "string", "description": "Search query string" },
        "max_results": { "type": "integer", "description": "Maximum results to return", "default": 10 }
      },
      "required": ["query"]
    }
  }
}
```

## External Tools

External tools are defined in the request's `tools` parameter and executed on the requester's side. They follow the standard OpenAI tool definition format.

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
            "param_name": { "type": "string", "description": "Parameter description" }
          },
          "required": ["param_name"]
        }
      }
    }
  ]
}
```

### Using External Tools

```typescript tab="TypeScript"
const externalTools = [
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get current weather for a location',
      parameters: {
        type: 'object',
        properties: {
          location: { type: 'string', description: 'The city and state, e.g. San Francisco, CA' },
          unit: { type: 'string', description: 'Temperature unit', enum: ['celsius', 'fahrenheit'] },
        },
        required: ['location'],
      },
    },
  },
] as const;

const messages = [{ role: 'user' as const, content: "What's the weather in Paris?" }];
const response = await agent.run(messages, { tools: externalTools as any });
```

```python tab="Python"
external_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "description": "Temperature unit", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
]

messages = [{"role": "user", "content": "What's the weather in Paris?"}]
response = await agent.run(messages=messages, tools=external_tools)
```

### Handling Tool Calls

When the agent emits tool calls, you execute them client-side and feed the results back:

```typescript tab="TypeScript"
const response = await agent.run(messages, { tools: externalTools as any });
const message = response;

if (message.tool_calls?.length) {
  for (const call of message.tool_calls) {
    const args = JSON.parse(call.function.arguments);
    let result = '';

    if (call.function.name === 'get_weather') {
      result = await getWeatherExternal(args.location);
    }

    messages.push({ role: 'assistant', content: message.content, tool_calls: [call] } as any);
    messages.push({ role: 'tool', tool_call_id: call.id, content: result } as any);
  }

  const final = await agent.run(messages, { tools: externalTools as any });
  console.log(final.content);
}

async function getWeatherExternal(location: string): Promise<string> {
  return `Sunny in ${location}, 22°C`;
}
```

```python tab="Python"
import json

response = await agent.run(messages=messages, tools=external_tools)
assistant_message = response.choices[0].message

if assistant_message.tool_calls:
    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name == "get_weather":
            result = get_weather_external(arguments["location"])

        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [tool_call],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    final_response = await agent.run(messages=messages, tools=external_tools)

def get_weather_external(location: str) -> str:
    return f"Sunny in {location}, 22°C"
```

## Tool Execution

### Automatic Tool Calling

```typescript tab="TypeScript"
const response = await agent.run([
  { role: 'user', content: "What's the weather in Paris?" },
]);
```

```python tab="Python"
response = await agent.run([
    {"role": "user", "content": "What's the weather in Paris?"}
])
```

### Manual Tool Results

```typescript tab="TypeScript"
const messages = [
  { role: 'user' as const, content: 'Calculate 42 * 17' },
  {
    role: 'assistant' as const,
    content: "I'll calculate that for you.",
    tool_calls: [
      {
        id: 'call_123',
        type: 'function',
        function: { name: 'multiply', arguments: '{"x": 42, "y": 17}' },
      },
    ],
  },
  { role: 'tool' as const, tool_call_id: 'call_123', content: '714' },
];
const response = await agent.run(messages as any);
```

```python tab="Python"
messages = [
    {"role": "user", "content": "Calculate 42 * 17"},
    {
        "role": "assistant",
        "content": "I'll calculate that for you.",
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"x": 42, "y": 17}'},
        }],
    },
    {"role": "tool", "tool_call_id": "call_123", "content": "714"},
]
response = await agent.run(messages)
```

## Advanced Tool Features

### Dynamic Tool Registration

```typescript tab="TypeScript"
import { Skill, hook } from 'webagents';

class AdaptiveSkill extends Skill {
  readonly name = 'adaptive';

  @hook({ lifecycle: 'on_connection' })
  async registerDynamicTools(data, ctx) {
    if (ctx.auth?.userId === 'admin') {
      // TS does not yet support runtime self-registration of decorated tools;
      // expose the tool unconditionally and gate it via @tool({ scopes: ['admin'] }).
    }
    return data;
  }
}
```

```python tab="Python"
class AdaptiveSkill(Skill):
    @hook("on_connection")
    async def register_dynamic_tools(self, context):
        """Register tools based on context"""

        if context.peer_user_id == "admin":
            self.register_tool(self.admin_tool, scope="admin")

        if "math" in str(context.messages):
            self.register_tool(self.advanced_calc)

        return context

    def admin_tool(self, action: str) -> str:
        """Admin-only tool"""
        return f"Admin action: {action}"
```

### Tool Middleware

```typescript tab="TypeScript"
import { Skill, hook } from 'webagents';

class ToolMonitor extends Skill {
  readonly name = 'tool-monitor';

  @hook({ lifecycle: 'before_toolcall', priority: 1 })
  async validateTool(data, ctx) {
    const toolName = data.tool_call?.function?.name;
    if (this.isRateLimited(toolName)) {
      throw new Error(`Tool ${toolName} rate limited`);
    }
    const args = JSON.parse(data.tool_call?.function?.arguments ?? '{}');
    this.validateArgs(toolName, args);
    return data;
  }

  @hook({ lifecycle: 'after_toolcall', priority: 90 })
  async logResult(data, ctx) {
    await this.logToolUsage({
      tool: data.tool_call?.function?.name,
      result: data.tool_result,
      duration: data.tool_duration,
    });
    return data;
  }

  private isRateLimited(_: string) { return false; }
  private validateArgs(_: string, __: unknown) {}
  private async logToolUsage(_: object) {}
}
```

```python tab="Python"
import json

class ToolMonitor(Skill):
    @hook("before_toolcall", priority=1)
    async def validate_tool(self, context):
        """Validate before execution"""
        tool_name = context["tool_call"]["function"]["name"]

        if self.is_rate_limited(tool_name):
            raise RateLimitError(f"Tool {tool_name} rate limited")

        args = json.loads(context["tool_call"]["function"]["arguments"])
        self.validate_args(tool_name, args)
        return context

    @hook("after_toolcall", priority=90)
    async def log_result(self, context):
        """Log tool execution"""
        await self.log_tool_usage(
            tool=context["tool_call"]["function"]["name"],
            result=context["tool_result"],
            duration=context.get("tool_duration"),
        )
        return context
```

### Tool Pricing

```typescript tab="TypeScript"
import { Skill, tool, pricing } from 'webagents';

class PaidToolsSkill extends Skill {
  readonly name = 'paid-tools';

  @pricing({ creditsPerCall: 0.10 })
  @tool({ description: 'Call expensive external API' })
  async expensiveApiCall(params: { query: string }): Promise<string> {
    return await this.callPaidApi(params.query);
  }

  @pricing({ creditsPerCall: 0.01 })
  @tool({ description: 'Execute database query' })
  async databaseQuery(params: { sql: string }): Promise<unknown[]> {
    return await this.executeSql(params.sql);
  }

  private async callPaidApi(_: string) { return ''; }
  private async executeSql(_: string): Promise<unknown[]> { return []; }
}
```

```python tab="Python"
from typing import List, Dict
from webagents import tool
from webagents.agents.skills.robutler.payments import pricing

class PaidToolsSkill(Skill):
    @tool
    @pricing(credits_per_call=0.10)
    def expensive_api_call(self, query: str) -> str:
        """Call expensive external API"""
        return self.call_paid_api(query)

    @tool
    @pricing(credits_per_call=0.01)
    def database_query(self, sql: str) -> List[Dict]:
        """Execute database query"""
        return self.execute_sql(sql)
```

## Tool Patterns

### Validation Pattern

```typescript tab="TypeScript"
@tool({ description: 'Update record with validation' })
async updateRecord(params: { recordId: string; data: Record<string, unknown> }) {
  if (!this.validateRecordId(params.recordId)) {
    return { error: 'Invalid record ID' };
  }
  if (!this.validateData(params.data)) {
    return { error: 'Invalid data format' };
  }
  try {
    const result = await this.db.update(params.recordId, params.data);
    return { success: true, record: result };
  } catch (e) {
    return { error: (e as Error).message };
  }
}
```

```python tab="Python"
@tool
def update_record(self, record_id: str, data: Dict) -> Dict:
    """Update record with validation"""
    if not self.validate_record_id(record_id):
        return {"error": "Invalid record ID"}

    if not self.validate_data(data):
        return {"error": "Invalid data format"}

    try:
        result = self.db.update(record_id, data)
        return {"success": True, "record": result}
    except Exception as e:
        return {"error": str(e)}
```

### Async Pattern

```typescript tab="TypeScript"
@tool({ description: 'Fetch data from multiple URLs concurrently' })
async fetchData(params: { urls: string[] }): Promise<unknown[]> {
  return await Promise.all(params.urls.map((u) => this.fetchUrl(u)));
}
```

```python tab="Python"
import asyncio
import aiohttp

@tool
async def fetch_data(self, urls: List[str]) -> List[Dict]:
    """Fetch data from multiple URLs concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [self.fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results
```

### Caching Pattern

```typescript tab="TypeScript"
class CachedToolsSkill extends Skill {
  readonly name = 'cached-tools';
  private cache = new Map<string, string>();

  @tool({ description: 'Cached expensive calculation' })
  async expensiveCalculation(params: { input: string }): Promise<string> {
    const cached = this.cache.get(params.input);
    if (cached) return cached;
    const result = await this.performCalculation(params.input);
    this.cache.set(params.input, result);
    return result;
  }

  private async performCalculation(_: string) { return ''; }
}
```

```python tab="Python"
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

1. **Clear descriptions** — help the LLM understand when to use each tool.
2. **Type hints / schemas** — enable accurate schema generation.
3. **Error handling** — return errors as structured data, not exceptions.
4. **Scope control** — use `scope` / `scopes` to gate tool visibility per caller.
5. **Performance** — consider caching and concurrent execution.
