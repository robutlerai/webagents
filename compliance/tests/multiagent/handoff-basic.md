---
name: multiagent-handoff-basic
version: 1.0
transport: completions
type: multi-agent
tags: [multiagent, handoff]
---

# Multi-Agent Handoff

Tests agent-to-agent handoff functionality.

## Setup

### Agent: router
- Name: `router`
- Instructions: "You are a routing agent. Route weather questions to weather-agent, math questions to calculator-agent."
- Handoffs: [weather-agent, calculator-agent]

### Agent: weather-agent
- Name: `weather-agent`
- Instructions: "You are a weather specialist. Provide weather information for requested locations."
- Tools: [get_weather]

### Agent: calculator-agent
- Name: `calculator-agent`
- Instructions: "You are a calculator. Solve math problems."
- Tools: [calculate]

## Test Cases

### 1. Successful Weather Handoff

**Input:**
User sends "What's the weather in Paris?" to `router`

**Flow:**
1. Router receives user message
2. Router identifies weather intent
3. Router hands off to weather-agent
4. Weather-agent processes the request
5. Weather-agent responds with weather information

**Assertions:**
- Handoff event occurred from router to weather-agent
- Weather-agent received the original query
- Final response is about weather
- Response mentions Paris or weather conditions

**Strict:**
```yaml
events:
  - type: handoff
    from: router
    to: weather-agent
response:
  status: 200
  agent: weather-agent
```

### 2. Successful Calculator Handoff

**Input:**
User sends "What is 42 * 17?" to `router`

**Flow:**
1. Router receives math question
2. Router identifies math intent
3. Router hands off to calculator-agent
4. Calculator-agent solves the problem
5. Calculator-agent responds with result

**Assertions:**
- Handoff to calculator-agent occurred
- Response contains the correct answer (714)
- Response came from calculator-agent

**Strict:**
```yaml
events:
  - type: handoff
    from: router
    to: calculator-agent
response:
  content: contains("714")
```

### 3. Direct Response (No Handoff)

**Input:**
User sends "Hello, how are you?" to `router`

**Flow:**
1. Router receives greeting
2. Router determines no specialist needed
3. Router responds directly

**Assertions:**
- No handoff occurred
- Router responded with a greeting
- Response is friendly and appropriate

**Strict:**
```yaml
events:
  handoff: not_exists
response:
  agent: router
```

### 4. Handoff with Context Preservation

**Input:**
Conversation:
1. User to router: "My favorite city is Tokyo"
2. Router: "That's a great city!"
3. User to router: "What's the weather there?"

**Flow:**
1. Router maintains context about Tokyo
2. Router hands off to weather-agent with context
3. Weather-agent understands "there" refers to Tokyo

**Assertions:**
- Weather-agent received conversation context
- Response is about Tokyo's weather
- "There" was correctly resolved to Tokyo

**Strict:**
```yaml
events:
  - type: handoff
    to: weather-agent
    context: contains("Tokyo")
response:
  content: contains("Tokyo")
```

### 5. Failed Handoff Recovery

**Input:**
User sends "What's the weather?" to `router`

**Setup:**
- weather-agent is unavailable/offline

**Flow:**
1. Router attempts handoff to weather-agent
2. Handoff fails (timeout/connection error)
3. Router handles error gracefully
4. Router informs user of the issue

**Assertions:**
- Handoff was attempted
- Error was caught and handled
- User received a meaningful error message
- No crash or unhandled exception

**Strict:**
```yaml
events:
  - type: handoff
    status: failed
response:
  status: [200, 503]
  content: type(string)
```

### 6. Chained Handoffs

**Setup:**
Add a third agent:
- Name: `report-agent`
- Instructions: "Summarize weather data into reports"

Router is updated:
- Handoffs: [weather-agent, calculator-agent, report-agent]

Weather-agent is updated:
- Handoffs: [report-agent]

**Input:**
User sends "Get the weather in NYC and create a report" to `router`

**Flow:**
1. Router hands to weather-agent
2. Weather-agent gets weather data
3. Weather-agent hands to report-agent
4. Report-agent creates summary
5. Response returns to user

**Assertions:**
- Two handoffs occurred
- Final response is a formatted report
- Report contains weather data

**Strict:**
```yaml
events:
  - type: handoff
    from: router
    to: weather-agent
  - type: handoff
    from: weather-agent
    to: report-agent
```

### 7. Handoff Metadata

**Input:**
User sends "Weather in London please" to `router`

**Assertions:**
- Handoff event contains metadata
- Metadata includes original user message
- Metadata includes timestamp
- Metadata includes session ID

**Strict:**
```yaml
events:
  - type: handoff
    metadata:
      message: contains("London")
      timestamp: type(number)
      session_id: type(string)
```
