---
title: LSP Skill
---
# LSP Skill

Language Server Protocol skill providing code intelligence via [Microsoft multilspy](https://github.com/microsoft/multilspy).

## Overview

The LSP skill provides code intelligence capabilities:

- **Go to Definition** - Find where a symbol is defined
- **Find References** - Find all usages of a symbol
- **Code Completions** - Get contextual completions at cursor
- **Hover Information** - Get type info and documentation
- **Document Symbols** - List all symbols in a file

## Supported Languages

| Language | Extensions | Notes |
|----------|------------|-------|
| Python | `.py`, `.pyw`, `.pyi` | Pyright/Pylance |
| TypeScript | `.ts`, `.tsx` | tsserver |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` | tsserver |
| Java | `.java` | Eclipse JDT |
| Rust | `.rs` | rust-analyzer |
| Go | `.go` | gopls |
| C# | `.cs` | OmniSharp |
| Dart | `.dart` | Dart Analysis Server |
| Ruby | `.rb` | Solargraph |
| Kotlin | `.kt`, `.kts` | kotlin-language-server |

Language is automatically detected from file extension.

## Configuration

```yaml
skills:
  lsp:
    project_root: "."  # Path to project root (default: current directory)
```

### Python API

```python
from webagents.agents.skills.local.lsp import LSPSkill

# Create skill with config
lsp_skill = LSPSkill({"project_root": "/path/to/project"})

# Add to agent
agent = BaseAgent(
    name="code-agent",
    skills={"lsp": lsp_skill},
)
```

## Tools

### `goto_definition`

Find the definition of a symbol at a given position.

```python
result = await skill.goto_definition(
    file_path="src/main.py",
    line=10,        # 1-indexed
    column=5,       # 1-indexed
    language=None   # auto-detected from extension
)
# Returns: {"found": True, "file": "src/utils.py", "line": 5, "column": 1}
```

### `find_references`

Find all references to a symbol.

```python
result = await skill.find_references(
    file_path="src/main.py",
    line=10,
    column=5,
    include_declaration=True,
    language=None
)
# Returns: {"count": 3, "references": [{"file": "...", "line": ..., "column": ...}]}
```

### `get_completions`

Get code completions at cursor position.

```python
result = await skill.get_completions(
    file_path="src/main.py",
    line=10,
    column=5,
    language=None
)
# Returns: {
#   "count": 15, 
#   "completions": [
#     {"label": "append", "kind": 2, "detail": "(item) -> None", ...}
#   ]
# }
```

### `get_hover`

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

### `get_document_symbols`

Get all symbols defined in a file.

```python
result = await skill.get_document_symbols(
    file_path="src/main.py",
    language=None
)
# Returns: {
#   "count": 5, 
#   "symbols": [
#     {"name": "greet", "kind": "Function", "line": 2, "parent": None},
#     {"name": "name", "kind": "Variable", "line": 3, "parent": "greet"}
#   ]
# }
```

## Slash Commands

| Command | Description |
|---------|-------------|
| `/lsp` | Show LSP status including active servers and supported languages |

Example output:
```
**LSP Status**
Project: /Users/me/myproject
Active servers: python, typescript
Supported: python, typescript, javascript, java, rust, go, csharp, dart, ruby, kotlin
```

## Usage Examples

### Agent Chat

```
User: What function is called at line 45 of main.py?

Agent: [uses get_hover tool]
The function at line 45 is `process_data(items: List[str]) -> Dict[str, int]`
which processes a list of strings and returns a frequency dictionary.

User: Where is that function defined?

Agent: [uses goto_definition tool]
`process_data` is defined at src/utils.py:23
```

### Programmatic Use

```python
# Initialize with project
lsp = LSPSkill({"project_root": "/path/to/project"})
await lsp.initialize(agent)

# Find all usages of a function
refs = await lsp.find_references("src/api.py", 50, 10)
print(f"Found {refs['count']} references")

for ref in refs['references']:
    print(f"  {ref['file']}:{ref['line']}")

# Don't forget to cleanup when done
await lsp.cleanup()
```

## API Notes

### Line/Column Indexing

- **API uses 1-indexed** positions (line 1 is the first line)
- Internally converts to 0-indexed for LSP protocol
- Results are converted back to 1-indexed for display

### Lazy Initialization

Language servers are started on first use and cached. This means:
- No startup delay when skill loads
- First request for a language may take longer
- Subsequent requests are fast

Call `cleanup()` to shut down all servers when done.

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Unsupported extension | Raises `ValueError` with list of supported extensions |
| Definition not found | Returns `{"found": False, "message": "..."}` |
| No completions | Returns `{"count": 0, "completions": []}` |
| Server errors | Logged; may raise exceptions |

## Symbol Kinds

The `get_document_symbols` tool returns a `kind` field with these values:

| Kind | Description |
|------|-------------|
| File | File |
| Module | Module |
| Namespace | Namespace |
| Package | Package |
| Class | Class |
| Method | Method |
| Property | Property |
| Field | Field |
| Constructor | Constructor |
| Enum | Enum |
| Interface | Interface |
| Function | Function |
| Variable | Variable |
| Constant | Constant |
| String | String literal |
| Number | Number literal |
| Boolean | Boolean literal |
| Array | Array |
| Object | Object literal |
| Struct | Struct |
| Event | Event |
| Operator | Operator |
| TypeParameter | Type parameter |

## Dependencies

```
multilspy>=0.0.15
```

Install with:
```bash
pip install multilspy
```

## Architecture

```
webagents/agents/skills/local/lsp/
├── __init__.py      # Export LSPSkill
├── skill.py         # LSPSkill class wrapping multilspy
├── languages.py     # Language detection and configuration
└── README.md        # Developer documentation
```

## See Also

- [multilspy GitHub](https://github.com/microsoft/multilspy)
- [Language Server Protocol Specification](https://microsoft.github.io/language-server-protocol/)
