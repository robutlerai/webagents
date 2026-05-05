---
title: Contributing to WebAgents
description: Workflow, code style, and testing guidelines for contributing to the WebAgents Python and TypeScript SDKs.
---

# Contributing to WebAgents

Thank you for your interest in contributing to WebAgents! This guide will help you get started.

## Getting Started

### Prerequisites

- Node.js 20+ and pnpm 9.x (TypeScript SDK)
- Python 3.10+ (Python SDK)
- Git and a GitHub account

### Development Environment Setup

1. **Fork & clone**

   ```bash
   git clone https://github.com/YOUR_USERNAME/webagents.git
   cd webagents
   ```

2. **Install the SDK you're working on**

   ```bash tab="TypeScript"
   corepack enable
   cd webagents/typescript
   pnpm install
   ```

   ```bash tab="Python"
   cd webagents/python
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Set up environment variables** (`.env` in project root):

   ```bash
   OPENAI_API_KEY=your-openai-api-key
   ROBUTLER_API_KEY=rok_your-robutler-api-key
   ```

4. **Verify the setup**

   ```bash tab="TypeScript"
   pnpm test
   pnpm run lint
   pnpm run typecheck
   ```

   ```bash tab="Python"
   pytest
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

```bash tab="TypeScript"
pnpm test
pnpm test -- --coverage
pnpm test src/agents/__tests__/base-agent.test.ts
pnpm run lint
pnpm run typecheck
```

```bash tab="Python"
pytest
pytest --cov=webagents
pytest tests/test_agent.py
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

```typescript tab="TypeScript"
// 2-space indent, ES modules, no `any` in public APIs.
// Public symbols carry TSDoc comments. Use Prettier + ESLint configs from the repo.
export interface ExampleConfig {
  /** Display name. */
  name: string;
  /** Optional runtime overrides. */
  options?: Record<string, unknown>;
}

export class ExampleClass {
  constructor(private readonly config: ExampleConfig) {}

  /**
   * Count occurrences of each item.
   * @throws if `items` is empty.
   */
  processItems(items: string[]): Record<string, number> {
    if (items.length === 0) {
      throw new Error('Items list cannot be empty');
    }
    return items.reduce<Record<string, number>>((acc, item) => {
      acc[item] = (acc[item] ?? 0) + 1;
      return acc;
    }, {});
  }
}
```

```python tab="Python"
# PEP 8, line length 88 (Black default), Google-style docstrings,
# type hints on every public symbol.
from typing import Optional, List, Dict, Any


class ExampleClass:
    """Example class demonstrating code style."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the example class.

        Args:
            name: The name of the instance.
            config: Optional configuration dictionary.
        """
        self.name = name
        self.config = config or {}

    def process_items(self, items: List[str]) -> Dict[str, int]:
        """Count occurrences of each item.

        Raises:
            ValueError: If items list is empty.
        """
        if not items:
            raise ValueError("Items list cannot be empty")
        return {item: items.count(item) for item in set(items)}
```

### Documentation Style

- TypeScript: TSDoc comments on public exports.
- Python: Google-style docstrings with type hints.
- Provide examples for non-trivial behavior.
- Keep documentation up to date with code changes.

## Testing Guidelines

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Use fixtures for common test data

Example test:

```typescript tab="TypeScript"
import { describe, it, expect } from 'vitest';
import { BaseAgent, Skill, tool } from 'webagents';

class TestSkill extends Skill {
  readonly name = 'test';

  @tool({ description: 'Returns a fixed result' })
  async testTool(): Promise<string> {
    return 'test result';
  }
}

describe('BaseAgent', () => {
  it('creates with valid config', () => {
    const agent = new BaseAgent({
      name: 'test-agent',
      instructions: 'You are a helpful assistant.',
      model: 'openai/gpt-4o-mini',
    });

    expect(agent.name).toBe('test-agent');
    expect(agent.instructions).toBe('You are a helpful assistant.');
  });

  it('registers skill tools', () => {
    const agent = new BaseAgent({
      name: 'tool-agent',
      instructions: 'You have tools.',
      skills: [new TestSkill()],
    });

    expect(agent.skills.length).toBe(1);
  });
});
```

```python tab="Python"
import pytest
from webagents import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool


class TestSkill(Skill):
    @tool(description="Returns a fixed result")
    async def test_tool(self) -> str:
        return "test result"


class TestBaseAgent:
    """Test cases for BaseAgent class."""

    def test_agent_creation_with_valid_config(self):
        """Test that agent can be created with valid configuration."""
        agent = BaseAgent(
            name="test-agent",
            instructions="You are a helpful assistant.",
            model="openai/gpt-4o-mini",
        )

        assert agent.name == "test-agent"
        assert agent.instructions == "You are a helpful assistant."

    def test_agent_with_skill(self):
        """Test that agent can be created with a skill."""
        agent = BaseAgent(
            name="tool-agent",
            instructions="You have tools.",
            skills={"test": TestSkill()},
        )

        assert "test" in agent.skills
```

### Running Tests

```bash tab="TypeScript"
pnpm test
pnpm test src/agents/__tests__/base-agent.test.ts
pnpm test -- --coverage
pnpm test -- -t "agent"
```

```bash tab="Python"
pytest
pytest tests/test_agent.py
pytest --cov=webagents
pytest -k "test_agent"
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