# Multi-Agent Testing

Testing interactions between multiple agents is crucial for complex agentic systems.

## Overview

Multi-agent tests verify:

- Agent handoffs and routing
- A2A (agent-to-agent) authentication
- Orchestration patterns
- State passing between agents

## Test Format

### Setup Multiple Agents

```markdown
---
name: multi-agent-handoff
version: 1.0
type: multi-agent
tags: [handoff, a2a]
---

# Multi-Agent Handoff

## Setup

### Agent: router
- Name: `router`
- Instructions: "Route requests to specialists based on intent"
- Handoffs: [weather-agent, search-agent, calculator-agent]

### Agent: weather-agent
- Name: `weather-agent`
- Instructions: "Provide weather information for locations"
- Tools: [get_current_weather, get_forecast]

### Agent: search-agent
- Name: `search-agent`
- Instructions: "Search the web for information"
- Tools: [web_search]
```

### Define Test Flow

```markdown
## Test Cases

### 1. Weather Request Routing

**Input:**
User sends "What's the weather in Tokyo?" to `router`

**Flow:**
1. Router receives user message
2. Router identifies weather intent
3. Router hands off to weather-agent
4. Weather-agent processes request
5. Weather-agent calls get_current_weather tool
6. Weather-agent responds with weather info

**Assertions:**
- Handoff event occurred from router to weather-agent
- Weather-agent received the original query
- Weather-agent used the weather tool
- Final response contains temperature information
- Response mentions Tokyo
```

## Handoff Patterns

### Direct Handoff

One agent transfers completely to another:

```markdown
**Flow:**
1. Router receives "Calculate 2+2"
2. Router hands off to calculator-agent
3. Calculator-agent returns result to user

**Assertions:**
- Single handoff occurred
- Calculator-agent handled the math
```

### Supervised Handoff

Router maintains control:

```markdown
**Flow:**
1. Router receives "Research quantum computing and write a summary"
2. Router assigns research to researcher-agent
3. Researcher returns findings to router
4. Router assigns writing to writer-agent
5. Writer returns summary to router
6. Router returns final result to user

**Assertions:**
- Router made 2 handoffs (to researcher, to writer)
- Router received intermediate results
- Final response came from router
```

### Parallel Handoff

Multiple agents work simultaneously:

```markdown
**Flow:**
1. Orchestrator receives "Compare weather in NYC and LA"
2. Orchestrator spawns two parallel requests to weather-agent
3. Both results return to orchestrator
4. Orchestrator synthesizes comparison

**Assertions:**
- Two parallel requests to weather-agent
- Both completed successfully
- Final response compares both cities
```

## A2A Authentication

### Testing Auth Flow

```markdown
### 2. Authenticated Handoff

**Setup:**
- router has AuthSkill configured
- weather-agent requires authentication

**Flow:**
1. Router requests token from its AuthSkill
2. Router calls weather-agent with bearer token
3. Weather-agent validates token via JWKS
4. Weather-agent processes request

**Assertions:**
- Authorization header was present
- Token was valid JWT
- weather-agent accepted authentication
- Request succeeded

**Strict:**
```yaml
handoff_request:
  headers:
    authorization: /^Bearer eyJ/
weather_agent_response:
  status: 200
```
```

### Testing Auth Failure

```markdown
### 3. Unauthorized Handoff

**Setup:**
- weather-agent requires authentication
- router does NOT have AuthSkill

**Flow:**
1. Router attempts handoff to weather-agent
2. weather-agent rejects unauthenticated request

**Assertions:**
- Handoff was attempted
- weather-agent returned 401 Unauthorized
- Error message mentions authentication
```

## State Passing

### Context Preservation

```markdown
### 4. State Across Handoffs

**Flow:**
1. User says "My name is Alice"
2. Router acknowledges and stores name
3. User says "What's the weather in NYC?"
4. Router hands off to weather-agent with context
5. Weather-agent responds addressing user by name

**Assertions:**
- Weather-agent received user name in context
- Response includes "Alice" or addresses user personally
```

### Conversation History

```markdown
### 5. History Forwarding

**Flow:**
1. User: "I'm planning a trip to Japan"
2. Router: "When are you planning to go?"
3. User: "Next month"
4. User: "What's the weather like there?"
5. Router hands off to weather-agent

**Assertions:**
- Weather-agent received full conversation history
- Weather-agent understood the trip context
- Response relates to Japan weather next month
```

## Event Assertions

### Handoff Events

```markdown
**Assertions:**
- At least one `handoff` event was emitted
- Handoff target was `weather-agent`
- Handoff included the original user message

**Strict:**
```yaml
events:
  - type: handoff
    target: weather-agent
    message: contains("weather")
```
```

### Tool Call Events

```markdown
**Assertions:**
- weather-agent called `get_current_weather` tool
- Tool was called with location parameter
- Tool result was incorporated in response

**Strict:**
```yaml
events:
  - type: tool.call
    agent: weather-agent
    name: get_current_weather
    arguments:
      location: exists
```
```

## Complex Scenarios

### Error Recovery

```markdown
### 6. Handoff Target Unavailable

**Setup:**
- search-agent is not running

**Flow:**
1. User: "Search for quantum computing papers"
2. Router attempts handoff to search-agent
3. Handoff fails (connection refused)
4. Router handles error gracefully

**Assertions:**
- Handoff was attempted
- Error was caught
- Router informed user of the issue
- No crash or unhandled exception
```

### Circular Handoff Prevention

```markdown
### 7. No Circular Handoffs

**Setup:**
- router can hand off to assistant
- assistant can hand off to router

**Flow:**
1. User: "Help me with a complex task"
2. Router hands to assistant
3. Assistant should NOT hand back to router indefinitely

**Assertions:**
- Maximum 2 handoffs total
- No infinite loop
- Eventually returns a response
```

## Best Practices

### 1. Test One Pattern at a Time

```markdown
# Bad: Testing everything
- Handoff
- Auth  
- State
- Tools
- Error handling

# Good: Focused tests
### 1. Basic Handoff
### 2. Authenticated Handoff  
### 3. State Preservation
```

### 2. Clear Agent Roles

```markdown
# Bad: Vague agents
### Agent: helper
- Instructions: "Help users"

# Good: Specific roles
### Agent: weather-specialist
- Instructions: "Provide detailed weather forecasts"
- Tools: [get_weather, get_forecast, get_alerts]
```

### 3. Verifiable Flows

```markdown
# Bad: Unverifiable
**Assertions:**
- Agents communicated

# Good: Specific events
**Assertions:**
- Handoff event from router to weather-agent
- weather-agent received message containing "weather"
- Tool call to get_weather with location "NYC"
```

### 4. Isolate External Dependencies

```markdown
## Setup

### Mock Services
- weather-agent uses mock weather API
- search-agent uses mock search results

This ensures consistent test results regardless of external service state.
```
