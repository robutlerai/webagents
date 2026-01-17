# Filesystem Skill

The Filesystem Skill provides agents with the ability to interact with the local filesystem in a sandboxed environment.

## Features

- **Sandboxing**: Restricts access to a specific base directory (usually the agent's folder) and whitelisted paths.
- **Security**: Prevents access to sensitive system directories (blacklist).
- **Comprehensive Tools**: Reading, writing, listing, searching, and replacing content.

## Configuration

In your `AGENT.md`:

```yaml
skills:
  - filesystem

# Optional configuration
filesystem:
  base_dir: "." # Defaults to agent's directory
  whitelist:
    - "/path/to/allowed/dir"
  blacklist:
    - "/path/to/blocked/dir"
```

## Tools

### `list_directory`
Lists files and directories.
- `path`: Directory path.
- `ignore`: Optional glob patterns to ignore.
- `respect_git_ignore`: Boolean to respect `.gitignore`.

### `read_file`
Reads file content.
- `path`: File path.
- `offset`: Start line (optional).
- `limit`: Max lines (optional).
- Handles text and binary files (returning base64 for media).

### `write_file`
Writes content to a file.
- `file_path`: File path.
- `content`: Content string.

### `glob`
Finds files matching a pattern.
- `pattern`: Glob pattern (e.g., `**/*.py`).
- `path`: Base path for search.

### `search_file_content`
Searches text within files using Regex.
- `pattern`: Regex pattern.
- `path`: Directory to search.
- `include`: Glob pattern to filter files.

### `replace`
Replaces text in a file.
- `file_path`: File path.
- `old_string`: Exact string to match.
- `new_string`: Replacement string.
- `expected_replacements`: Safety check for number of matches.
