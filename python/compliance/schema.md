# Compliance Test Schema

This document defines the schema for compliance test specifications.

## File Format

Compliance tests are Markdown files with YAML frontmatter.

## Frontmatter Schema

```yaml
# Required fields
name: string          # Unique test identifier (kebab-case)
version: string       # Spec version (semver)
transport: string     # Transport being tested: completions, realtime, a2a

# Optional fields
type: string          # single-agent (default) | multi-agent
tags: [string]        # Categorization tags
timeout: number       # Timeout in seconds (default: 60)
depends_on: [string]  # Test dependencies (run after these)
skip_reason: string   # If present, test is skipped
```

## Body Structure

### Sections

1. `# Title` - Test title (H1)
2. Description - Brief explanation (paragraph after H1)
3. `## Setup` - Agent/environment configuration
4. `## Test Cases` - Individual test cases (H3)

### Setup Section

#### Single Agent
```markdown
## Setup

Create an agent with:
- Name: `agent-name`
- Instructions: "Agent system prompt"
- Tools: [tool1, tool2]
- Skills: [skill1, skill2]
```

#### Multi-Agent
```markdown
## Setup

### Agent: agent-id
- Name: `display-name`
- Instructions: "System prompt"
- Handoffs: [other-agent-ids]
- Tools: [tools]

### Agent: another-agent
- Name: `another-name`
- Instructions: "Another prompt"
```

#### Environment
```markdown
### Environment
- `ENV_VAR`: Description
- `BASE_URL`: SDK server URL
```

### Test Case Structure

```markdown
### N. Test Case Name

**Request:**
[HTTP method] `[path]`
[Headers (optional)]
```json
{
  "request": "body"
}
```

**Flow:** (for multi-agent)
1. Step one
2. Step two

**Assertions:**
- Natural language assertion 1
- Natural language assertion 2

**Strict:**
```yaml
# Deterministic assertions
status: 200
body:
  field: value
```

**Expected:** success | failure
```

## Assertion Types

### Natural Language

Any human-readable assertion:

```markdown
- Response status is 200
- The assistant provided a helpful greeting
- Tool call was made to get_weather
- Response mentions the requested location
```

### Strict Assertions

YAML block with JSONPath expressions:

```yaml
status: number              # HTTP status code
headers:
  header-name: value        # Header assertions
body:
  path.to.field: value      # Exact match
  path.to.field: /regex/    # Regex match
  path.to.field: exists     # Field exists
  path.to.field: not_null   # Not null
  array[0]: value           # Array access
  array[*]: value           # All elements
```

### Operators

| Syntax | Description |
|--------|-------------|
| `value` | Exact match |
| `/regex/` | Regular expression |
| `exists` | Field is present |
| `not_null` | Not null/undefined |
| `type(string)` | Type check |
| `length(N)` | Array/string length |
| `contains("str")` | Substring match |
| `[min, max]` | Value in range |

## Multi-Agent Specific

### Flow Section

```markdown
**Flow:**
1. User sends [message] to `agent-id`
2. Agent performs [action]
3. Agent hands off to `other-agent`
4. Other agent [action]
5. Response returned to user
```

### Event Assertions

```yaml
events:
  - type: handoff
    from: router
    to: specialist
  - type: tool.call
    agent: specialist
    name: tool_name
    arguments:
      param: exists
```

## Validation

Tests are validated against this schema before execution:

1. Frontmatter has required fields
2. Name is unique and kebab-case
3. At least one test case exists
4. Assertions are parseable
5. Strict YAML is valid

## Examples

See `compliance/tests/` for complete examples.
