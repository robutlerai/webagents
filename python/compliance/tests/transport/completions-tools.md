---
name: completions-tools
version: 1.0
transport: completions
tags: [core, tools]
---

# Tool Calling

Tests tool/function calling via the `/chat/completions` endpoint.

## Setup

Create an agent with the following configuration:
- Name: `tool-agent`
- Instructions: "Use the provided tools to help users. Always use tools when appropriate."
- Tools: [get_weather, search, calculate]

### Tool Definitions

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather for a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "City name"},
          "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "calculate",
      "description": "Perform mathematical calculations",
      "parameters": {
        "type": "object",
        "properties": {
          "expression": {"type": "string", "description": "Math expression"}
        },
        "required": ["expression"]
      }
    }
  }
]
```

## Test Cases

### 1. Tool Call Response

**Request:**
POST `/chat/completions`
```json
{
  "model": "tool-agent",
  "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
  "tools": [/* get_weather tool */]
}
```

**Assertions:**
- Response status is 200
- Response indicates tool call is needed
- Tool call is for `get_weather` function
- Arguments include location
- Location relates to Tokyo
- finish_reason is "tool_calls"

**Strict:**
```yaml
status: 200
body:
  choices[0].message.tool_calls: exists
  choices[0].message.tool_calls[0].type: function
  choices[0].message.tool_calls[0].function.name: get_weather
  choices[0].message.tool_calls[0].function.arguments: contains("Tokyo")
  choices[0].finish_reason: tool_calls
```

### 2. Tool Result Handling

**Request:**
POST `/chat/completions`
```json
{
  "model": "tool-agent",
  "messages": [
    {"role": "user", "content": "What's the weather in NYC?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"NYC\"}"
        }
      }
    ]},
    {"role": "tool", "tool_call_id": "call_123", "content": "{\"temperature\": 72, \"condition\": \"sunny\"}"}
  ],
  "tools": [/* get_weather tool */]
}
```

**Assertions:**
- Response status is 200
- Agent incorporated tool result in response
- Response mentions temperature or weather condition
- finish_reason is "stop"

**Strict:**
```yaml
status: 200
body:
  choices[0].message.role: assistant
  choices[0].message.content: type(string)
  choices[0].finish_reason: stop
```

### 3. Multiple Tool Calls

**Request:**
POST `/chat/completions`
```json
{
  "model": "tool-agent",
  "messages": [{"role": "user", "content": "Compare weather in NYC and LA"}],
  "tools": [/* get_weather tool */]
}
```

**Assertions:**
- Response may request multiple tool calls
- Each tool call has unique ID
- Tool calls are for different locations

**Strict:**
```yaml
status: 200
body:
  choices[0].finish_reason: tool_calls
```

### 4. Parallel Tool Calls

**Request:**
POST `/chat/completions`
```json
{
  "model": "tool-agent",
  "messages": [{"role": "user", "content": "What's the weather in NYC and calculate 2+2"}],
  "tools": [/* get_weather, calculate tools */],
  "parallel_tool_calls": true
}
```

**Assertions:**
- Response may include multiple tool calls
- Tool calls can be for different functions
- Each has unique ID

### 5. Tool Choice: None

**Request:**
POST `/chat/completions`
```json
{
  "model": "tool-agent",
  "messages": [{"role": "user", "content": "What's the weather in NYC?"}],
  "tools": [/* get_weather tool */],
  "tool_choice": "none"
}
```

**Assertions:**
- Response status is 200
- No tool calls in response
- Agent responds with text only
- finish_reason is "stop"

**Strict:**
```yaml
status: 200
body:
  choices[0].message.tool_calls: not_exists
  choices[0].finish_reason: stop
```

### 6. Tool Choice: Specific Function

**Request:**
POST `/chat/completions`
```json
{
  "model": "tool-agent",
  "messages": [{"role": "user", "content": "Hello, how are you?"}],
  "tools": [/* get_weather tool */],
  "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
}
```

**Assertions:**
- Response forces use of get_weather tool
- Tool call is present even for unrelated query
- Tool call is for get_weather

**Strict:**
```yaml
status: 200
body:
  choices[0].message.tool_calls: exists
  choices[0].message.tool_calls[0].function.name: get_weather
```

### 7. Invalid Tool Name in Result

**Request:**
POST `/chat/completions`
```json
{
  "model": "tool-agent",
  "messages": [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "wrong_id", "content": "{}"}
  ],
  "tools": [/* tools */]
}
```

**Assertions:**
- Response handles mismatched tool_call_id gracefully
- Either error or continues with available data

### 8. Tool Call Arguments Parsing

**Request:**
POST `/chat/completions`
```json
{
  "model": "tool-agent",
  "messages": [{"role": "user", "content": "What's 25 * 4?"}],
  "tools": [/* calculate tool */]
}
```

**Assertions:**
- Tool call arguments are valid JSON
- Arguments can be parsed
- Expression is math-related

**Strict:**
```yaml
body:
  choices[0].message.tool_calls[0].function.arguments: type(string)
```
