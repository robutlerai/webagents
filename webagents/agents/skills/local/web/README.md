# Web Skill

The Web Skill provides tools for interacting with the internet.

## Features

- **Fetching**: Retrieves and summarizes web content.
- **Extraction**: Uses `trafilatura` to extract main content and remove clutter.
- **Safety**: Asks for user confirmation in interactive modes.

## Configuration

In your `AGENT.md`:

```yaml
skills:
  - web
```

## Tools

### `web_fetch`
Fetches and processes content from URLs.
- `prompt`: Natural language prompt containing URLs and instructions (e.g., "Summarize https://example.com").
