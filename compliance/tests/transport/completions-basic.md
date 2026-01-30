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
- Response is valid JSON
- Response contains `choices` array with at least one item
- The assistant message has role "assistant"
- The response contains content that echoes the input
- `finish_reason` is "stop" or equivalent completion indicator

**Strict:**
```yaml
status: 200
body:
  object: chat.completion
  choices: length(1)
  choices[0].message.role: assistant
  choices[0].message.content: type(string)
  choices[0].finish_reason: stop
```

### 2. System Message

**Request:**
POST `/chat/completions`
```json
{
  "model": "echo-agent",
  "messages": [
    {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
    {"role": "user", "content": "Hello"}
  ]
}
```

**Assertions:**
- Response status is 200
- System message influenced the response style
- Response has pirate-like language

**Strict:**
```yaml
status: 200
body:
  choices[0].message.role: assistant
  choices[0].message.content: type(string)
```

### 3. Multi-turn Conversation

**Request:**
POST `/chat/completions`
```json
{
  "model": "echo-agent",
  "messages": [
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What is my name?"}
  ]
}
```

**Assertions:**
- Response status is 200
- Response remembers the name from conversation history
- Response mentions "Alice"

**Strict:**
```yaml
status: 200
body:
  choices[0].message.content: contains("Alice")
```

### 4. Empty Messages Array

**Request:**
POST `/chat/completions`
```json
{
  "model": "echo-agent",
  "messages": []
}
```

**Assertions:**
- Response status is 400 (Bad Request)
- Error object is present
- Error message explains the issue

**Expected:** failure

**Strict:**
```yaml
status: 400
body:
  error: exists
  error.message: type(string)
```

### 5. Missing Model

**Request:**
POST `/chat/completions`
```json
{
  "messages": [{"role": "user", "content": "Hello"}]
}
```

**Assertions:**
- Response status is 400 (Bad Request)
- Error indicates missing model field

**Expected:** failure

**Strict:**
```yaml
status: 400
body:
  error: exists
```

### 6. Invalid JSON

**Request:**
POST `/chat/completions`
Content-Type: application/json

```
{invalid json}
```

**Assertions:**
- Response status is 400 (Bad Request)
- Error message indicates JSON parsing failure

**Expected:** failure

**Strict:**
```yaml
status: 400
```
