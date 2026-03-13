---
title: Test Format Reference
---
# Test Format Reference

Compliance tests are written in structured Markdown with YAML frontmatter.

## Basic Structure

```markdown
---
name: test-name
version: 1.0
transport: completions
tags: [core, required]
---

# Test Title

Brief description of what this test validates.

## Setup

Agent and environment configuration.

## Test Cases

### 1. Test Case Name

**Request:**
HTTP request details

**Assertions:**
- Natural language assertion 1
- Natural language assertion 2

**Strict:**
```yaml
# Optional deterministic assertions
status: 200
body:
  key: value
```
```

## Frontmatter

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique test identifier |
| `version` | string | Yes | Test spec version |
| `transport` | string | Yes | Transport being tested (completions, realtime, a2a) |
| `tags` | array | No | Categorization tags |
| `type` | string | No | `single-agent` (default) or `multi-agent` |
| `timeout` | number | No | Test timeout in seconds |

## Setup Section

### Single Agent

```markdown
## Setup

Create an agent with the following configuration:
- Name: `echo-agent`
- Instructions: "Repeat back exactly what the user says, prefixed with 'Echo: '"
- Tools: [get_weather, search]
```

### Multi-Agent

```markdown
## Setup

### Agent: router
- Name: `router`
- Instructions: "Route requests to appropriate specialists"
- Handoffs: [weather-agent, search-agent]

### Agent: weather-agent
- Name: `weather-agent`
- Instructions: "Provide weather information"
```

### Environment Variables

```markdown
## Setup

### Environment
- `BASE_URL`: SDK server URL (default: http://localhost:8765)
- `API_KEY`: Test API key
```

## Test Cases

### Request Formats

**HTTP Request:**
```markdown
**Request:**
POST `/chat/completions`
```json
{
  "model": "test-agent",
  "messages": [{"role": "user", "content": "Hello"}]
}
```
```

**With Headers:**
```markdown
**Request:**
POST `/chat/completions`
Headers:
- `Authorization: Bearer test-token`
- `X-Custom-Header: value`

Body:
```json
{...}
```
```

**Streaming Request:**
```markdown
**Request:**
POST `/chat/completions` (streaming)
```json
{"stream": true, ...}
```
```

### Assertions

#### Natural Language

Human-readable assertions for agentic validation:

```markdown
**Assertions:**
- Response status is 200
- The assistant responded with a greeting
- Response contains weather information for the requested location
- Tool call was made to `get_weather`
- finish_reason is "stop"
```

#### Strict (Deterministic)

Optional YAML block for exact matching:

```markdown
**Strict:**
```yaml
status: 200
body:
  choices[0].message.role: assistant
  choices[0].message.content: /^Echo:/
  choices[0].finish_reason: stop
headers:
  content-type: application/json
```
```

##### Strict Assertion Operators

| Operator | Example | Description |
|----------|---------|-------------|
| `equals` (default) | `status: 200` | Exact match |
| `regex` | `content: /^Hello/` | Regex match (prefix with `/`) |
| `contains` | `content: contains("weather")` | Substring match |
| `exists` | `choices: exists` | Field is present |
| `not_null` | `choices[0]: not_null` | Field is not null |
| `length` | `choices: length(1)` | Array/string length |
| `type` | `content: type(string)` | Type check |

##### JSONPath

Use JSONPath for nested values:

```yaml
body:
  choices[0].message.content: "Hello"
  usage.total_tokens: type(number)
  choices[*].finish_reason: "stop"  # All choices
```

### Flow (Multi-Agent)

For multi-agent tests, describe the expected flow:

```markdown
**Flow:**
1. User sends "What's the weather in NYC?" to `router`
2. Router recognizes weather intent
3. Router hands off to `weather-agent`
4. Weather-agent responds with weather information

**Assertions:**
- Handoff event was triggered
- Final response came from weather-agent
- Response mentions temperature or weather
```

## Advanced Features

### Conditional Tests

```markdown
### 2. Tool Call (skip if no tools)

**Condition:** Agent has tools configured

**Request:**
...
```

### Expected Failures

```markdown
### 3. Error Handling

**Request:**
POST `/chat/completions` with invalid JSON

**Assertions:**
- Response status is 400
- Error message explains the issue

**Expected:** failure
```

### Test Dependencies

```markdown
---
name: auth-flow
depends_on: [session-create]
---
```

### Data Generation

```markdown
**Request:**
POST `/chat/completions`
```json
{
  "model": "{{agent_name}}",
  "messages": [{"role": "user", "content": "{{random_greeting}}"}]
}
```

**Variables:**
- `agent_name`: From setup
- `random_greeting`: One of ["Hello", "Hi", "Hey"]
```

## Complete Example

```markdown
---
name: completions-basic
version: 1.0
transport: completions
tags: [core, required]
---

# Basic Chat Completion

Tests that the `/chat/completions` endpoint handles simple requests correctly.

## Setup

Create an agent with the following configuration:
- Name: `echo-agent`
- Instructions: "Repeat back exactly what the user says, prefixed with 'Echo: '"

## Test Cases

### 1. Simple Message

**Request:**
POST `/chat/completions`
```json
{
  "model": "echo-agent",
  "messages": [{"role": "user", "content": "Hello"}]
}
```

**Assertions:**
- Response status is 200
- Response contains `choices` array with at least one item
- The assistant message starts with "Echo:"
- `finish_reason` is "stop"

**Strict:**
```yaml
status: 200
body:
  choices[0].message.role: assistant
  choices[0].message.content: /^Echo:/
  choices[0].finish_reason: stop
```

### 2. Empty Message Handling

**Request:**
POST `/chat/completions`
```json
{
  "model": "echo-agent",
  "messages": []
}
```

**Assertions:**
- Response status is 400
- Error message explains the issue

**Strict:**
```yaml
status: 400
body:
  error.type: invalid_request_error
```
```
