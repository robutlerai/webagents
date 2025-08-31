# Contributing to WebAgents

Thank you for your interest in contributing to WebAgents! This guide will help you get started with contributing to the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A GitHub account

### Development Environment Setup

1. **Fork the repository**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/webagents.git
   cd webagents
   ```

2. **Set up the development environment**
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Add your API keys
   OPENAI_API_KEY=your-openai-api-key
   ROBUTLER_API_KEY=your-robutler-api-key
   ```

4. **Verify the setup**
   ```bash
   # Run tests to ensure everything is working
   pytest
   
   # Run linting
   flake8 webagents/
   black --check webagents/
   ```

## Development Workflow

### 1. Create a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Write clean, readable code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run the full test suite
pytest

# Run tests with coverage
pytest --cov=webagents

# Run specific tests
pytest tests/test_agent.py

# Run linting
flake8 webagents/
black --check webagents/
```

### 4. Commit Your Changes

We use conventional commits for clear commit messages:

```bash
git add .
git commit -m "feat: add new agent configuration option"
# or
git commit -m "fix: resolve payment processing error"
# or
git commit -m "docs: update API documentation"
```

Commit types:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### 5. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference to any related issues
- Screenshots or examples if applicable

## Code Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for all public functions
- Use docstrings for all public classes and functions
- Prefer f-strings for string formatting

Example:

```python
from typing import Optional, List, Dict, Any

class ExampleClass:
    """Example class demonstrating code style.
    
    This class shows the preferred code style for WebAgents
    including type hints, docstrings, and formatting.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the example class.
        
        Args:
            name: The name of the instance
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
    
    def process_items(self, items: List[str]) -> Dict[str, int]:
        """Process a list of items and return counts.
        
        Args:
            items: List of items to process
            
        Returns:
            Dictionary mapping items to their counts
            
        Raises:
            ValueError: If items list is empty
        """
        if not items:
            raise ValueError("Items list cannot be empty")
        
        return {item: items.count(item) for item in set(items)}
```

### Documentation Style

- Use Google-style docstrings
- Include type information in docstrings
- Provide examples for complex functions
- Keep documentation up to date with code changes

## Testing Guidelines

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Use fixtures for common test data

Example test:

```python
import pytest
from webagents.agent import BaseAgent

class TestBaseAgent:
    """Test cases for BaseAgent class."""
    
    def test_agent_creation_with_valid_config(self):
        """Test that agent can be created with valid configuration."""
        # Arrange
        name = "test-agent"
        instructions = "You are a helpful assistant."
        
        # Act
        agent = BaseAgent(
            name=name,
            instructions=instructions,
            credits_per_token=10
        )
        
        # Assert
        assert agent.name == name
        assert agent.instructions == instructions
        assert agent.credits_per_token == 10
    
    def test_agent_with_tools(self):
        """Test that agent can be created with tools."""
        from agents import function_tool
        
        @function_tool
        def test_tool() -> str:
            return "test result"
        
        agent = BaseAgent(
            name="tool-agent",
            instructions="You have tools.",
            tools=[test_tool]
        )
        
        assert len(agent.tools) == 1
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agent.py

# Run with coverage
pytest --cov=webagents

# Run tests matching a pattern
pytest -k "test_agent"

# Run tests with verbose output
pytest -v
```

## Contributing Areas

### Areas Where We Need Help

1. **Agent Tools**: New tools that extend agent capabilities
2. **Documentation**: Improving guides and API documentation
3. **Testing**: Adding test coverage for existing functionality
4. **Bug Fixes**: Resolving reported issues
5. **Performance**: Optimizing agent response times
6. **Examples**: Creating example applications and use cases

### Feature Requests

Before implementing new features:

1. **Check existing issues**: See if the feature is already requested
2. **Create an issue**: Discuss the feature with maintainers first
3. **Get approval**: Wait for maintainer approval before starting work
4. **Follow guidelines**: Use this contributing guide for implementation

## Getting Help

- **Issues**: Check [GitHub Issues](https://github.com/robutlerai/webagents/issues) for existing problems
- **Discussions**: Use [GitHub Discussions](https://github.com/robutlerai/webagents/discussions) for questions
- **Discord**: Join our community Discord for real-time help

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

Thank you for contributing to WebAgents! 