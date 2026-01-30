# Agentic Testing

WebAgents uses an innovative **agentic testing** approach where an AI agent reads human-readable test specifications and validates SDK implementations.

## Philosophy

Traditional software testing relies on deterministic assertions—exact string matches, status codes, JSON schemas. But AI agent behavior is inherently variable. An agent might respond to "Hello" with "Hi there!", "Hello!", or "Hey! How can I help?"—all valid responses.

**Agentic testing** addresses this by:

1. **Natural language assertions** - "The response should be a friendly greeting"
2. **Intent-based validation** - Testing *what* should happen, not *exactly how*
3. **LLM-powered reasoning** - An AI agent validates whether responses meet the intent
4. **Strict fallback** - Optional deterministic assertions for CI reproducibility

## Test Hierarchy

| Layer | Location | LLM | Cache | Trigger |
|-------|----------|-----|-------|---------|
| **Unit** | SDK repos | Mocked | N/A | Every PR |
| **Compliance** | Main repo (submodule) | temp=0 | Read+Write | Every PR |
| **Integration** | Main repo | temp>0 | Disabled | Nightly |

### Unit Tests (Deterministic)

Fast, mocked tests in each SDK repo:

- Type validation
- Message serialization
- Hook ordering
- Tool registration
- Context management

**No LLM calls** - everything is mocked.

### Compliance Tests (Cached LLM)

Tests that verify SDK implementations conform to the [UAMP protocol](https://uamp.dev):

- API contract validation (endpoints, formats)
- Streaming behavior
- Tool calling lifecycle
- Multi-agent handoffs

Uses **temperature=0** with response caching for reproducibility.

### Integration Tests (Live LLM)

End-to-end tests with real LLM calls:

- Full conversation flows
- A2A authentication
- Portal connectivity
- Performance benchmarks

Runs **nightly** with **temperature>0** to catch edge cases.

## Getting Started

### Writing Tests

See [Writing Tests](writing-tests.md) for a complete guide.

### Test Format

Tests are written in structured Markdown. See [Test Format Reference](test-format.md).

### Running Tests

See [Running Tests](running-tests.md) for CLI and CI usage.

## Key Concepts

### Agentic Runner

The test runner is itself a WebAgents agent with tools to:

- Make HTTP requests to SDKs under test
- Parse markdown test specifications
- Validate responses against natural language assertions
- Execute strict (deterministic) validations
- Report results

### Response Caching

For CI reproducibility, LLM responses are cached:

- **CI/PR**: temp=0, cache read+write → deterministic
- **Nightly**: temp=0.3, cache write only → regression detection
- **Pre-release**: temp=0.7, no cache → full variability

### Multi-Agent Testing

Tests can define multiple agents and verify their interactions:

```markdown
## Setup

### Agent: router
- Instructions: "Route weather questions to weather-agent"
- Handoffs: [weather-agent]

### Agent: weather-agent
- Instructions: "Provide weather information"

## Test Cases

### Successful Handoff
**Flow:**
1. User sends "What's the weather?" to router
2. Router hands off to weather-agent
3. Weather-agent responds
```

## Documentation

- [Writing Tests](writing-tests.md) - How to write test specifications
- [Test Format Reference](test-format.md) - Markdown format documentation
- [Running Tests](running-tests.md) - CLI and CI usage
- [Response Caching](caching.md) - Caching strategy for reproducibility
- [Multi-Agent Testing](multi-agent.md) - Testing agent interactions
