# Web Skill

The Web Skill provides tools for interacting with the internet.

## Features

- **Fetching**: Retrieves and summarizes web content.
- **Extraction**: Uses `trafilatura` (the `scraping` extra) to extract the main content when installed; otherwise the page's text without scripts, styles and markup.
- **Safety**: Fetches only public internet addresses. Every address a name resolves to is checked, the connection goes to the checked address, and each redirect is checked again before it is followed. Loopback, private-network, link-local and cloud metadata addresses are refused.

## Configuration

In your `AGENT.md`:

```yaml
skills:
  - web
```

To reach a private address on purpose (a docs server on your own machine), name it, with the REST tool's syntax:

```yaml
skills:
  - web:
      allow_private: ["127.0.0.1:8080", "10.0.0.0/8"]
```

Link-local and cloud metadata addresses stay refused whatever the list says.

## Tools

### `web_fetch`
Fetches and processes content from URLs.
- `prompt`: Natural language prompt containing URLs and instructions (e.g., "Summarize https://example.com").
