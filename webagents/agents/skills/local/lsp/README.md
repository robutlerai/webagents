# LSP Skill

Language Server Protocol skill providing code intelligence via [Microsoft multilspy](https://github.com/microsoft/multilspy).

## Features

- **Go to Definition** - Find where a symbol is defined
- **Find References** - Find all usages of a symbol
- **Code Completions** - Get contextual completions at cursor
- **Hover Information** - Get type info and documentation
- **Document Symbols** - List all symbols in a file

## Supported Languages

| Language | Extensions |
|----------|------------|
| Python | `.py`, `.pyw`, `.pyi` |
| TypeScript | `.ts`, `.tsx` |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` |
| Java | `.java` |
| Rust | `.rs` |
| Go | `.go` |
| C# | `.cs` |
| Dart | `.dart` |
| Ruby | `.rb` |
| Kotlin | `.kt`, `.kts` |

## Installation

Add to your dependencies:

```
multilspy>=0.0.15
```

## Configuration

```yaml
skills:
  lsp:
    project_root: "."  # Path to project root (default: current directory)
```

## Usage

### In Agent Configuration

```python
from webagents.agents.skills.local.lsp import LSPSkill

# Create skill with config
lsp_skill = LSPSkill({"project_root": "/path/to/project"})

# Add to agent
agent.add_skill(lsp_skill)
```

### Available Tools

#### `goto_definition`

Find the definition of a symbol at a given position.

```python
result = await skill.goto_definition(
    file_path="src/main.py",
    line=10,        # 1-indexed
    column=5,       # 1-indexed
    language=None   # auto-detected
)
# Returns: {"found": True, "file": "src/utils.py", "line": 5, "column": 1}
```

#### `find_references`

Find all references to a symbol.

```python
result = await skill.find_references(
    file_path="src/main.py",
    line=10,
    column=5,
    include_declaration=True,
    language=None
)
# Returns: {"count": 3, "references": [...]}
```

#### `get_completions`

Get code completions at cursor position.

```python
result = await skill.get_completions(
    file_path="src/main.py",
    line=10,
    column=5,
    language=None
)
# Returns: {"count": 15, "completions": [...]}
```

#### `get_hover`

Get hover information (type info, docs) for a symbol.

```python
result = await skill.get_hover(
    file_path="src/main.py",
    line=10,
    column=5,
    language=None
)
# Returns: {"found": True, "content": "def greet(name: str) -> str"}
```

#### `get_document_symbols`

Get all symbols defined in a file.

```python
result = await skill.get_document_symbols(
    file_path="src/main.py",
    language=None
)
# Returns: {"count": 5, "symbols": [{"name": "greet", "kind": "Function", "line": 2}, ...]}
```

### Slash Commands

#### `/lsp`

Show LSP skill status including active servers and supported languages.

```
/lsp

**LSP Status**
Project: /path/to/project
Active servers: python, typescript
Supported: python, typescript, javascript, java, rust, go, csharp, dart, ruby, kotlin
```

## API Notes

### Line/Column Indexing

- **API uses 1-indexed** positions (line 1 is the first line)
- Internally converts to 0-indexed for LSP protocol
- Results are converted back to 1-indexed for display

### Language Detection

Language is automatically detected from file extension. You can override by passing the `language` parameter.

### Lazy Initialization

Language servers are started on first use and cached. Call `cleanup()` to shut down all servers when done.

### Error Handling

- Unsupported file extensions raise `ValueError`
- Missing definitions/references return `{"found": False}` with message
- Server errors are logged and may raise exceptions

## Architecture

```
webagents/agents/skills/local/lsp/
├── __init__.py      # Export LSPSkill
├── skill.py         # LSPSkill class wrapping multilspy
├── languages.py     # Language detection and configuration
└── README.md        # This documentation
```

## Dependencies

- `multilspy>=0.0.15` - Microsoft's multi-language LSP client

## See Also

- [multilspy GitHub](https://github.com/microsoft/multilspy)
- [Language Server Protocol Specification](https://microsoft.github.io/language-server-protocol/)
