# Development Setup

This guide covers setting up a development environment for working on Robutler.

## Prerequisites

- **Python**: 3.8 or higher
- **Git**: Latest version
- **OpenAI API Key**: For agent functionality

## Environment Setup

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/robutlerai/robutler.git
cd robutler-proxy

# Or clone your fork
git clone https://github.com/YOUR_USERNAME/robutler.git
cd robutler-proxy
```

### 2. Python Environment

#### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
# Install development dependencies
pip install -e ".[dev]"
```

### 4. Environment Variables

Create a `.env` file in the project root:

```bash
# Required for agent functionality
OPENAI_API_KEY=your-openai-api-key

# Optional Robutler API configuration
WEBAGENTS_API_KEY=rok_your-robutler-api-key
ROBUTLER_API_URL=https://robutler.ai

# Development settings
ROBUTLER_DEBUG=true
```

## Development Tools

### Code Formatting and Linting

#### Black (Code Formatting)

```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .
```

#### isort (Import Sorting)

```bash
# Sort imports
isort .

# Check import sorting
isort --check-only .
```

#### flake8 (Linting)

```bash
# Run linting
flake8 robutler/
```

### Testing

#### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=robutler

# Run specific test file
pytest tests/test_agent.py

# Run tests with verbose output
pytest -v
```

### Documentation

#### Building Documentation

```bash
# Serve documentation locally
cd docs
mkdocs serve

# Build documentation
mkdocs build
```

## IDE Configuration

### VS Code

Recommended extensions:
- Python
- Black Formatter
- isort
- Flake8

VS Code settings (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

## Running the Development Server

### Basic Agent Server

```python
# Create a simple test agent
from webagents.agent import RobutlerAgent
from webagents.server import RobutlerServer

agent = RobutlerAgent(
    name="test-agent",
    instructions="You are a helpful test assistant.",
    credits_per_token=5
)

app = RobutlerServer(agents=[agent])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

## Common Development Tasks

### Adding a New Tool

1. Create the tool function:

```python
from agents import function_tool
from webagents.server import pricing

@function_tool
@pricing(credits_per_call=1000)
def my_new_tool(input_text: str) -> str:
    """Description of what the tool does."""
    # Implementation here
    return f"Processed: {input_text}"
```

2. Add to agent:

```python
agent = RobutlerAgent(
    name="test-agent",
    instructions="You have access to custom tools.",
    tools=[my_new_tool],
    credits_per_token=5
)
```

3. Test the tool:

```python
# Test in development
messages = [{"role": "user", "content": "Use the new tool"}]
response = await agent.run(messages=messages)
print(response)
```

### Adding New API Endpoints

```python
from webagents.server import RobutlerServer

app = RobutlerServer()

@app.agent("/custom-endpoint")
@app.pricing(credits_per_token=10)
async def custom_agent(request):
    """Custom agent endpoint."""
    messages = request.messages
    # Process messages
    return "Custom response"
```

### Testing Changes

```bash
# Run tests for specific modules
pytest tests/test_agent.py -v

# Run integration tests
pytest tests/test_integration.py

# Check code formatting
black --check .
isort --check-only .
flake8 robutler/
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export ROBUTLER_DEBUG=true
```

### Common Debug Tasks

```bash
# Test agent endpoint
curl -X POST http://localhost:8000/test-agent/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test-agent", "messages": [{"role": "user", "content": "Hello"}]}'

# Check available tools
curl http://localhost:8000/test-agent
```

This covers the essential development setup needed to contribute to Robutler. 