---
title: Writing Compliance Tests
---
# Writing Compliance Tests

This guide explains how to write effective compliance tests for WebAgents SDKs.

## Getting Started

1. Create a new `.md` file in `compliance/tests/`
2. Add YAML frontmatter with test metadata
3. Write setup, test cases, and assertions

## Choosing What to Test

### Good Candidates

- API endpoint behavior (request/response format)
- Streaming chunk format
- Error handling
- Tool call lifecycle
- Multi-agent handoffs
- Authentication flows

### Not for Compliance Tests

- Exact LLM output (too variable)
- Performance benchmarks (separate suite)
- UI behavior (use E2E tests)

## Writing Assertions

### Natural Language (Preferred)

Write assertions as you'd describe them to a colleague:

```markdown
**Assertions:**
- The response is a valid JSON object
- The assistant provided a helpful answer
- The tool was called with the location parameter
- Response completed without errors
```

**Good assertions are:**

- Clear and unambiguous
- Focused on intent, not exact values
- Testable by an AI

**Avoid:**

- Vague assertions ("response is good")
- Exact content matching ("response is 'Hello!'")
- Multiple conditions in one assertion

### Strict Assertions (For CI)

Add strict assertions for deterministic CI:

```markdown
**Strict:**
```yaml
status: 200
body:
  object: chat.completion
  choices: length(1)
  choices[0].message.role: assistant
  choices[0].finish_reason: exists
```
```

Use strict assertions for:

- Status codes
- Required fields
- Field types
- Error codes

## Multi-Agent Tests

### Setup Multiple Agents

```markdown
## Setup

### Agent: orchestrator
- Instructions: "Coordinate tasks between specialists"
- Handoffs: [researcher, writer]

### Agent: researcher
- Instructions: "Research topics and provide facts"
- Tools: [search, wikipedia]

### Agent: writer
- Instructions: "Write content based on research"
```

### Define Flow

```markdown
**Flow:**
1. User requests "Write an article about quantum computing"
2. Orchestrator assigns research task to researcher
3. Researcher gathers facts using search tool
4. Researcher returns findings to orchestrator
5. Orchestrator assigns writing task to writer
6. Writer produces article
7. Orchestrator returns final article to user
```

### Assert on Events

```markdown
**Assertions:**
- Orchestrator made at least 2 handoffs
- Researcher used the search tool
- Final response is a coherent article
- All agents completed their tasks
```

## Testing Tool Calls

```markdown
## Test Cases

### Tool Call Round-Trip

**Request:**
POST `/chat/completions`
```json
{
  "model": "weather-agent",
  "tools": [{"type": "function", "function": {"name": "get_weather", ...}}],
  "messages": [{"role": "user", "content": "What's the weather in NYC?"}]
}
```

**Assertions:**
- Response indicates a tool call is needed
- Tool call is for `get_weather` function
- Arguments include a location parameter
- Location relates to NYC

**Strict:**
```yaml
body:
  choices[0].message.tool_calls: exists
  choices[0].message.tool_calls[0].function.name: get_weather
  choices[0].finish_reason: tool_calls
```
```

## Testing Streaming

```markdown
### Streaming Response

**Request:**
POST `/chat/completions` (streaming)
```json
{
  "model": "echo-agent",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true
}
```

**Assertions:**
- Response is SSE format
- Each chunk is valid JSON
- Chunks have incrementing content
- Final chunk has finish_reason
- Final event is [DONE]

**Strict:**
```yaml
format: sse
chunks:
  - object: chat.completion.chunk
  - choices[0].delta: exists
final_chunk:
  choices[0].finish_reason: stop
```
```

## Testing Error Handling

```markdown
### Invalid Request

**Request:**
POST `/chat/completions`
```json
{
  "model": "nonexistent-agent",
  "messages": "not an array"
}
```

**Assertions:**
- Response status is 400 or 422
- Error object is present
- Error message is helpful
- Error type identifies the issue

**Strict:**
```yaml
status: [400, 422]
body:
  error: exists
  error.message: type(string)
  error.type: type(string)
```
```

## Best Practices

### 1. One Concept Per Test

```markdown
# Bad: Testing multiple things
### Test Everything
- Check status
- Check headers
- Check streaming
- Check tools
- Check auth

# Good: Focused tests
### 1. Basic Response Format
### 2. Streaming Chunks
### 3. Tool Call Format
### 4. Authentication Header
```

### 2. Clear Setup

```markdown
# Bad: Vague setup
## Setup
Create an agent.

# Good: Specific setup
## Setup
Create an agent with:
- Name: `test-echo`
- Instructions: "Echo user input prefixed with 'Echo:'"
- No tools
```

### 3. Minimal Requests

```markdown
# Bad: Kitchen sink request
```json
{
  "model": "agent",
  "messages": [...],
  "temperature": 0,
  "max_tokens": 1000,
  "top_p": 1,
  "frequency_penalty": 0,
  ...
}
```

# Good: Only what's needed
```json
{
  "model": "agent",
  "messages": [{"role": "user", "content": "Hello"}]
}
```
```

### 4. Meaningful Assertions

```markdown
# Bad: Too vague
- Response is valid

# Bad: Too specific  
- Response is exactly "Hello! I'm here to help you today."

# Good: Intent-focused
- Response is a friendly greeting
- Response offers to help
```

## Template

```markdown
---
name: your-test-name
version: 1.0
transport: completions
tags: [category]
---

# Test Title

Brief description of what this test validates.

## Setup

Describe the agent configuration needed.

## Test Cases

### 1. Happy Path

**Request:**
[HTTP method and path]
[Request body]

**Assertions:**
- Natural language assertion 1
- Natural language assertion 2

**Strict:**
```yaml
status: 200
body:
  field: value
```

### 2. Edge Case

...
```
