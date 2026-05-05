---
title: Widgets
description: Interactive HTML / ChatKit widgets rendered in the chat — Python-first today, TypeScript on the roadmap.
---

# Widgets

Widgets are interactive components that can be rendered in the chat interface, providing rich user experiences beyond text and images.

> **TypeScript: Coming soon.** The `@widget` decorator and `WidgetTemplateRenderer` ship in the Python SDK only. In the TypeScript SDK, you can return widget-formatted HTML strings from a regular `@tool`, and chat clients that recognize the `<widget>` envelope will render them. Track parity in the [Python ↔ TypeScript Parity Matrix](../internal/python-typescript-parity.md).

## Overview

The WebAgents widget system supports two distinct widget types:

| Type | Description | Creation | Rendering | Use Cases |
|------|-------------|----------|-----------|-----------|
| **WebAgents Widgets** | Custom HTML/JavaScript components | Code with `@widget` decorator (Python) or tool returning `<widget kind="webagents">` HTML (both) | Sandboxed iframes | Interactive UIs, media players, forms, visualizations |
| **OpenAI ChatKit Widgets** | Widgets from OpenAI's tools | [OpenAI Widget Builder](https://widgets.chatkit.studio/) / [Agent Builder](https://platform.openai.com/agent-builder) | Direct React rendering | Structured layouts, data displays, simple interactions |

## Widget Decorator

The `@widget` decorator (Python) accepts:

- **name** *(optional)* — override widget name (defaults to function name)
- **description** *(optional)* — widget description for LLM awareness (defaults to docstring)
- **template** *(optional)* — path to Jinja2 template file (WebAgents widgets only)
- **scope** *(optional)* — access control: `"all"`, `"owner"`, `"admin"`, or list of scopes
- **auto_escape** *(optional, default `True`)* — automatically HTML-escape string arguments

## WebAgents Widgets

WebAgents widgets are custom HTML/JavaScript components rendered in sandboxed iframes. They provide maximum flexibility for interactive user interfaces.

### Basic Example

```typescript tab="TypeScript"
// @widget is Python-only today. The TypeScript equivalent is a regular @tool
// that returns a string wrapped in <widget kind="webagents"> tags. Chat clients
// that support widget rendering will pick it up automatically.

import { Skill, tool } from 'webagents';

class MusicPlayerSkill extends Skill {
  readonly name = 'music-player';

  @tool({
    description: 'Display an interactive music player for a given track',
    scopes: ['all'],
  })
  async playMusic(params: { song_url: string; title: string; artist?: string }): Promise<string> {
    const escape = (s: string) => s.replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!
    ));
    const title = escape(params.title);
    const artist = escape(params.artist ?? 'Unknown Artist');
    const songUrl = escape(params.song_url);

    const html = `<!DOCTYPE html>
<html>
<head><script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-gray-900 p-4">
  <div class="max-w-md mx-auto bg-gray-800 rounded-lg p-6">
    <h2 class="text-white text-xl font-bold">${title}</h2>
    <p class="text-gray-400">${artist}</p>
    <audio controls class="w-full mt-4"><source src="${songUrl}" type="audio/mpeg"></audio>
    <button onclick="window.parent.postMessage({type:'widget_message',content:'Play next song'},'*')"
            class="bg-blue-600 text-white px-4 py-2 rounded mt-4 w-full">Next Song</button>
  </div>
</body>
</html>`;

    return `<widget kind="webagents" id="music_player">${html}</widget>`;
  }
}
```

```python tab="Python"
from webagents import Skill, widget
from webagents.server.context.context_vars import Context

class MusicPlayerSkill(Skill):
    @widget(
        name="play_music",
        description="Display an interactive music player for a given track",
        scope="all",
        # auto_escape=True by default — arguments are escaped automatically.
    )
    async def play_music(
        self,
        song_url: str,
        title: str,
        artist: str = "Unknown Artist",
        context: Context = None,
    ) -> str:
        """Create an interactive music player widget.

        All string arguments are automatically HTML-escaped for security.
        """

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 p-4">
    <div class="max-w-md mx-auto bg-gray-800 rounded-lg p-6">
        <h2 class="text-white text-xl font-bold">{title}</h2>
        <p class="text-gray-400">{artist}</p>
        <audio controls class="w-full mt-4">
            <source src="{song_url}" type="audio/mpeg">
        </audio>
        <button
            onclick="sendMessage('Play next song')"
            class="bg-blue-600 text-white px-4 py-2 rounded mt-4 w-full">
            Next Song
        </button>
    </div>
    <script>
        function sendMessage(text) {{
            window.parent.postMessage({{
                type: 'widget_message',
                content: text
            }}, '*');
        }}
    </script>
</body>
</html>"""

        return f'<widget kind="webagents" id="music_player">{html_content}</widget>'
```

### Widget Format

WebAgents widgets must follow this format:

```text
<widget kind="webagents" id="<widget_id>">{html_content}</widget>
```

**Required attributes:**

- `kind="webagents"` — identifies this as a WebAgents widget.
- `id` — unique identifier for the widget.

**Optional attributes:**

- `data` — JSON metadata for state restoration (see [Advanced Usage](#advanced-widget-data-attribute)).

### Template Rendering

For complex widgets, use Jinja2 templates (Python) or template literals (TypeScript):

```typescript tab="TypeScript"
import { Skill, tool } from 'webagents';

class ComplexWidgetSkill extends Skill {
  readonly name = 'complex-widget';

  @tool({ description: 'Render a complex widget' })
  async complexWidget(params: { data: Record<string, string> }): Promise<string> {
    const html = renderTemplate('complex.html', params.data);
    return `<widget kind="webagents" id="complex">${html}</widget>`;
  }
}

function renderTemplate(_name: string, _data: Record<string, string>): string {
  return '<div>complex widget</div>';
}
```

```python tab="Python"
from webagents import WidgetTemplateRenderer

renderer = WidgetTemplateRenderer(template_dir="widgets")

@widget(name="complex_widget", template="complex.html")
async def complex_widget(self, data: dict) -> str:
    html = renderer.render("complex.html", data)
    return f'<widget kind="webagents" id="complex">{html}</widget>'
```

For simple widgets, return inline HTML:

```typescript tab="TypeScript"
@tool({ description: 'Render a simple widget' })
async simpleWidget(params: { text: string }): Promise<string> {
  const escape = (s: string) => s.replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!
  ));
  const html = `<div class='p-4'>${escape(params.text)}</div>`;
  return `<widget kind="webagents" id="simple">${html}</widget>`;
}
```

```python tab="Python"
@widget
async def simple_widget(self, text: str) -> str:
    html = f"<div class='p-4'>{text}</div>"  # Auto-escaped
    return f'<widget kind="webagents" id="simple">{html}</widget>'
```

### Styling

WebAgents widgets should use Tailwind CSS. Include the CDN in your HTML:

```html
<script src="https://cdn.tailwindcss.com"></script>
```

Or use the helper method (Python):

```python
from webagents import WidgetTemplateRenderer

html = WidgetTemplateRenderer.inject_tailwind_cdn(my_html)
```

### Advanced: Widget `data` attribute

The optional `data` attribute carries structured metadata for state restoration, analytics, error recovery, or dynamic configuration.

**Backend — adding data:**

```typescript tab="TypeScript"
@tool({ description: 'Stateful music player' })
async statefulPlayer(params: { song_url: string; last_position?: number }): Promise<string> {
  const widgetData = {
    song_url: params.song_url,
    last_position: params.last_position ?? 0,
  };

  const html = `<audio id="audio" src="${params.song_url}"></audio>
<script>
  window.addEventListener('message', (e) => {
    if (e.data.type === 'widget_init') {
      const audio = document.getElementById('audio');
      audio.currentTime = e.data.data.last_position || 0;
      audio.play();
    }
  });
</script>`;

  const escapeAttr = (s: string) => s.replace(/[&<>"]/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]!
  ));
  const escapedData = escapeAttr(JSON.stringify(widgetData));
  return `<widget kind="webagents" id="player" data="${escapedData}">${html}</widget>`;
}
```

```python tab="Python"
import html
import json
import time

@widget(name="music_player")
async def play_music(self, song_url: str, title: str, artist: str) -> str:
    html_content = f"""..."""  # Your widget HTML

    widget_data = {
        "song_url": song_url,
        "title": title,
        "artist": artist,
        "timestamp": time.time(),
    }

    escaped_data = html.escape(json.dumps(widget_data), quote=True)

    return f'<widget kind="webagents" id="music_player" data="{escaped_data}">{html_content}</widget>'
```

**Frontend — accessing data:**

```javascript
window.addEventListener('message', (event) => {
  if (event.data.type === 'widget_init') {
    const widgetData = event.data.data;
    // Use widgetData for state restoration, analytics, etc.
  }
});
```

### Security

#### Sandboxing

Widgets render in sandboxed iframes:

```html
<iframe sandbox="allow-scripts allow-same-origin" />
```

**Security features:**

- Isolated execution — no access to the parent window.
- No cookies/storage access.
- Blob URLs — content served from memory.
- Script execution allowed for interactivity.
- Same-origin policy for styling and APIs.

#### XSS Prevention

Widgets are **secure by default** with automatic HTML escaping (Python `auto_escape=True`).

In TypeScript, escape user-provided strings yourself before interpolating into HTML — there is no `auto_escape` runtime hook today.

#### Communication

Widgets communicate with the chat interface via the `postMessage` API:

```javascript
window.parent.postMessage({
  type: 'widget_message',
  content: 'User message text',
}, '*');
```

1. Widget sends a `widget_message` event.
2. Frontend validates the message structure.
3. Content is appended as a user message.
4. Agent processes the new message.

### Browser Detection

WebAgents widgets are only available to browser clients. The system detects browsers via User-Agent headers (Mozilla, Chrome, Safari, Firefox, Edge). OpenAI ChatKit widgets may work in more contexts since they don't require iframe support.

## OpenAI ChatKit Widgets

ChatKit widgets are created using OpenAI's [Widget Builder](https://widgets.chatkit.studio/) and [Agent Builder](https://platform.openai.com/agent-builder). WebAgents provides full rendering support for widgets created in these tools.

```typescript tab="TypeScript"
import { Skill, tool } from 'webagents';

class OpenAIWidgetSkill extends Skill {
  readonly name = 'openai-widgets';

  @tool({ description: "Render a widget created in OpenAI's Widget Builder" })
  async renderOpenAIWidget(params: { widget_json: string }): Promise<string> {
    return `<widget kind="openai">${params.widget_json}</widget>`;
  }
}
```

```python tab="Python"
import json
from webagents import widget

@widget(name="openai_widget")
async def render_openai_widget(self, widget_json: str, context: Context = None) -> str:
    """Render a widget created in OpenAI's Widget Builder."""
    return f'<widget kind="openai">{widget_json}</widget>'
```

### Generating Widget JSON

```typescript tab="TypeScript"
@tool({ description: 'Display an informational card (OpenAI ChatKit compatible)' })
async infoCard(params: { title: string; description: string }): Promise<string> {
  const widgetStructure = {
    $kind: 'card',
    content: [
      { $kind: 'text', content: params.title, size: 'lg', weight: 'bold' },
      { $kind: 'text', content: params.description },
    ],
  };
  return `<widget kind="openai">${JSON.stringify(widgetStructure)}</widget>`;
}
```

```python tab="Python"
import json
from webagents import widget

@widget(name="info_card")
async def info_card(self, title: str, description: str) -> str:
    """Display an informational card (OpenAI ChatKit compatible)"""
    widget_structure = {
        "$kind": "card",
        "content": [
            {"$kind": "text", "content": title, "size": "lg", "weight": "bold"},
            {"$kind": "text", "content": description},
        ],
    }
    return f'<widget kind="openai">{json.dumps(widget_structure)}</widget>'
```

### Widget Format

OpenAI ChatKit widgets use this format:

```text
<widget kind="openai">{widget_json}</widget>
```

**Sources:**

- JSON from [OpenAI's Widget Builder](https://widgets.chatkit.studio/).
- Widget definitions from [OpenAI's Agent Builder](https://platform.openai.com/agent-builder).
- Programmatically generated JSON (must follow the OpenAI ChatKit format).

### Supported Components

WebAgents renders all components from [OpenAI's Widget Builder](https://widgets.chatkit.studio/):

| Category | Component | Status | Description |
|----------|-----------|--------|-------------|
| **Layout** | `Card` | Supported | Container with optional styling |
| | `Box` | Supported | Flexible container |
| | `Row` | Supported | Horizontal layout |
| | `Col` | Supported | Vertical layout |
| | `Spacer` | Supported | Flexible space |
| | `Divider` | Supported | Visual separator |
| **Typography** | `Text` | Supported | Text content with formatting |
| | `Caption` | Supported | Small text captions |
| | `Title` | Supported | Large heading text |
| | `Label` | Supported | Label text |
| | `Markdown` | Supported | Markdown content |
| **Content** | `Image` | Supported | Display images |
| | `Icon` | Supported | Display icons (emoji/unicode) |
| | `Chart` | Supported | Bar and line chart visualizations |
| | `Badge` | Supported | Status badges with variants |
| **Controls** | `Button` | Supported | Interactive buttons |
| | `DatePicker` | Supported | Date selection input |
| | `Select` | Supported | Dropdown selection |
| | `Checkbox` | Supported | Checkbox input |
| | `RadioGroup` | Supported | Radio button group |
| | `Form` | Supported | Form container |
| **Other** | `Transition` | Supported | Animated transitions |

## Choosing Between Widget Types

| Choose | When You Need |
|--------|---------------|
| **WebAgents Widgets** | Custom HTML/JS interactivity, complex layouts, full control over rendering |
| **OpenAI ChatKit Widgets** | Visual widget creation with [OpenAI's tools](https://widgets.chatkit.studio/), standard UI patterns, compatibility with OpenAI agents |
| **Tools** | Data fetching/processing, no UI, text-based output |

## Troubleshooting

### Widget Not Appearing

- Verify the User-Agent is from a browser.
- Check that `<widget>` tags are properly formatted.
- Ensure the `kind` attribute is correct.

### `postMessage` Not Working

- Verify `type: 'widget_message'` is set.
- Check that the iframe sandbox allows `allow-scripts`.
- Ensure the target is `'*'` for blob URLs.

### Styling Issues

- Confirm the Tailwind CDN is included.
- Check for conflicting styles.
- Test with `colorScheme: 'normal'` on the iframe.

## API Reference (Python)

```python
@widget(
    name: Optional[str] = None,
    description: Optional[str] = None,
    template: Optional[str] = None,
    scope: Union[str, List[str]] = "all",
    auto_escape: bool = True,
)
```

`WidgetTemplateRenderer` — Jinja2 template renderer for WebAgents HTML widgets.

```python
class WidgetTemplateRenderer:
    def __init__(self, template_dir: Optional[str] = None): ...
    def render(self, template_name: str, context: Dict[str, Any]) -> str: ...
    def render_inline(self, html_string: str, context: Dict[str, Any]) -> str: ...

    @staticmethod
    def escape_data(data: Any) -> str: ...

    @staticmethod
    def inject_tailwind_cdn(html_content: str) -> str: ...
```

## Related Documentation

- [Tools](./tools.md) — Simple tool-based interactions
- [Handoffs](./handoffs.md) — Agent delegation
- [Skills](../skills/overview.md) — Skill development
