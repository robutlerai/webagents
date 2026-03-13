---
title: Running Compliance Tests
---
# Running Compliance Tests

This guide covers how to run compliance tests locally and in CI.

## Prerequisites

1. SDK server running (Python or TypeScript)
2. Test runner installed
3. LLM API key (for agentic validation)

## Quick Start

```bash
# From SDK repo (with compliance submodule)
cd compliance

# Run all tests
python -m compliance.runner

# Run specific test
python -m compliance.runner tests/transport/completions-basic.md

# Run by tag
python -m compliance.runner --tag core
```

## CLI Options

```bash
python -m compliance.runner [options] [test_files...]

Options:
  --base-url URL       SDK server URL (default: http://localhost:8765)
  --tag TAG            Run tests with specific tag
  --skip-tag TAG       Skip tests with specific tag
  --strict-only        Only run strict assertions (no LLM)
  --cache-mode MODE    Cache mode: read_write, write_only, disabled
  --temperature TEMP   LLM temperature (default: 0)
  --timeout SECONDS    Test timeout (default: 60)
  --parallel N         Run N tests in parallel
  --output FORMAT      Output format: text, json, junit
  --verbose            Show detailed output
```

## Running Locally

### 1. Start SDK Server

**Python SDK:**
```bash
cd webagents-python
webagents serve --port 8765
```

**TypeScript SDK:**
```bash
cd webagents-ts
npm run serve -- --port 8765
```

### 2. Run Tests

```bash
# All compliance tests
python -m compliance.runner

# Specific transport
python -m compliance.runner tests/transport/

# Single test
python -m compliance.runner tests/transport/completions-basic.md
```

### 3. View Results

```
Compliance Test Results
=======================

✓ completions-basic: 3/3 passed
  ✓ 1. Simple Message
  ✓ 2. Empty Message Handling  
  ✓ 3. Streaming Response

✓ completions-tools: 2/2 passed
  ✓ 1. Tool Call Request
  ✓ 2. Tool Result Handling

✗ multiagent-handoff: 1/2 passed
  ✓ 1. Successful Handoff
  ✗ 2. Auth Required Handoff
    Assertion failed: "Authorization header present"
    Response did not include Authorization header

Summary: 6/7 tests passed (85.7%)
```

## Cache Management

### Cache Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `read_write` | Read from cache, write new responses | CI/PR (default) |
| `write_only` | Always call LLM, write to cache | Nightly |
| `disabled` | No caching | Pre-release |

### Commands

```bash
# Use cache (default)
python -m compliance.runner --cache-mode read_write

# Refresh cache
python -m compliance.runner --cache-mode write_only

# No cache
python -m compliance.runner --cache-mode disabled
```

### Cache Location

```
compliance/
├── .cache/
│   ├── openai/
│   │   └── {hash}.json
│   └── manifest.json
```

### Clearing Cache

```bash
# Clear all cache
rm -rf compliance/.cache/

# Clear specific provider
rm -rf compliance/.cache/openai/
```

## CI Integration

### GitHub Actions

```yaml
name: Compliance Tests

on: [push, pull_request]

jobs:
  compliance:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: true  # For compliance submodule
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install -e .
          pip install -r compliance/requirements.txt
      
      - name: Start server
        run: |
          webagents serve --port 8765 &
          sleep 5
      
      - name: Run compliance tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python -m compliance.runner \
            --base-url http://localhost:8765 \
            --cache-mode read_write \
            --output junit \
            > test-results.xml
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: compliance-results
          path: test-results.xml
```

### Cache in CI

```yaml
      - name: Cache compliance responses
        uses: actions/cache@v4
        with:
          path: compliance/.cache
          key: compliance-cache-${{ hashFiles('compliance/tests/**/*.md') }}
          restore-keys: |
            compliance-cache-
```

## Strict-Only Mode

For CI without LLM API access:

```bash
python -m compliance.runner --strict-only
```

This runs only the `**Strict:**` assertions, skipping natural language validation.

## Output Formats

### Text (Default)

Human-readable console output.

### JSON

```bash
python -m compliance.runner --output json
```

```json
{
  "summary": {
    "total": 7,
    "passed": 6,
    "failed": 1,
    "skipped": 0
  },
  "tests": [
    {
      "name": "completions-basic",
      "cases": [
        {"name": "Simple Message", "status": "passed"},
        {"name": "Empty Handling", "status": "passed"}
      ]
    }
  ]
}
```

### JUnit XML

```bash
python -m compliance.runner --output junit > results.xml
```

Compatible with CI systems like Jenkins, GitHub Actions.

## Debugging Failed Tests

### Verbose Mode

```bash
python -m compliance.runner --verbose tests/transport/completions-basic.md
```

Shows:
- Full request/response bodies
- Assertion evaluation details
- LLM reasoning (if applicable)

### Single Test Case

```bash
python -m compliance.runner \
  tests/transport/completions-basic.md \
  --test-case "Simple Message"
```

### Debug Logging

```bash
DEBUG=compliance python -m compliance.runner
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for agentic runner |
| `COMPLIANCE_BASE_URL` | Default SDK server URL |
| `COMPLIANCE_CACHE_DIR` | Cache directory path |
| `COMPLIANCE_TIMEOUT` | Default test timeout |
