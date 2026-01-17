In your `AGENT.md` YAML header:

```yaml
skills:
  - sandbox
  - mcp

mcp:
  sqlite:
    command: uvx
    args:
      - mcp-server-sqlite
      - --db-path
      - ./data.db  # You can use relative paths!
    # No "sandbox: true" needed - it's automatic when 'sandbox' skill is present!
```

### Explanation

1.  **Skills**: Including `sandbox` activates the Docker container.
2.  **Relative Paths**: The sandbox automatically sets the working directory to `/workspace` (which maps to your agent's folder).
    *   `./data.db` correctly resolves to `/workspace/data.db` inside the container.
    *   This makes your configuration portable between local and sandboxed modes.
3.  **Dependencies**: Ensure your Docker image has the necessary tools (like `uv` for `uvx`).
    *   The default `python:3.11-slim` image does not include `uv`.
    *   You may need to configure a custom image or install it.
