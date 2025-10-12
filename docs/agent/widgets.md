# Widgets

Widgets are interactive HTML components that can be rendered in the chat interface, providing rich user experiences beyond text and images. The WebAgents widget system supports both custom skill widgets (sandboxed iframes) and OpenAI ChatKit widgets.

## Overview

The widget system consists of:

- **Backend**: `@widget` decorator for defining widgets in skills
- **Frontend**: Sandboxed iframe rendering for security
- **Communication**: postMessage API for widget-to-chat interaction
- **Two Widget Types**:
  - **WebAgents Widgets** (`kind="webagents"`): Custom HTML/JS widgets in sandboxed iframes
  - **OpenAI Widgets** (`kind="openai"`): OpenAI ChatKit component library widgets

## Creating a Widget

### Basic Widget Example

```python
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import widget
from webagents.server.context.context_vars import Context

class MusicPlayerSkill(Skill):
    @widget(
        name="play_music",
        description="Display an interactive music player for a given track",
        scope="all"
        # auto_escape=True by default - arguments are automatically escaped!
    )
    async def play_music(
        self,
        song_url: str,
        title: str,
        artist: str = "Unknown Artist",
        context: Context = None
    ) -> str:
        """Create an interactive music player widget
        
        All string arguments are automatically HTML-escaped for security.
        No need for manual html.escape() calls!
        """
        
        # Create HTML content - arguments are already escaped!
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
        
        # Return wrapped in <widget> tags (no data attribute needed)
        return f'<widget kind="webagents" id="music_player">{html_content}</widget>'
```

## Widget Decorator Parameters

The `@widget` decorator accepts the following parameters:

- **name** (optional): Override for widget name (defaults to function name)
- **description** (optional): Widget description for LLM awareness (defaults to docstring)
- **template** (optional): Path to Jinja2 template file
- **scope** (optional): Access control - `"all"`, `"owner"`, `"admin"`, or list of scopes
- **auto_escape** (optional, default=True): Automatically HTML-escape all string arguments for security

## Widget Function Signature

Widget functions should:

1. Return a string containing HTML wrapped in `<widget>` tags
2. Use `kind="webagents"` attribute to specify WebAgents widgets (default if not provided)
3. Include an `id` attribute for identification
4. Optionally include a `data` attribute with JSON metadata (HTML-escaped) - see below
5. Accept `context: Context = None` parameter for automatic context injection

## Widget Data Attribute (Advanced)

The optional `data` attribute allows you to pass structured metadata to your widgets. This is useful for:

- **State Restoration**: Resume playback position, volume, or other settings after re-render
- **Analytics & Logging**: Track widget interactions with full context
- **Error Recovery**: Provide fallback URLs or alternate configurations
- **Dynamic Configuration**: Adapt widget behavior based on metadata

### Backend: Adding Data

```python
import html
import json

@widget(name="music_player")
async def play_music(self, song_url: str, title: str, artist: str) -> str:
    html_content = f"""..."""  # Your widget HTML
    
    # Optional: Add metadata for state restoration/analytics
    widget_data = {
        'song_url': song_url,
        'title': title,
        'artist': artist,
        'timestamp': time.time()
    }
    
    # Escape JSON for HTML attribute (must use html.escape with quote=True)
    escaped_data = html.escape(json.dumps(widget_data), quote=True)
    
    return f'<widget kind="webagents" id="music_player" data="{escaped_data}">{html_content}</widget>'
```

### Frontend: Accessing Data

The widget iframe receives data in two ways:

**1. Via postMessage (Recommended)**

```javascript
// Inside widget HTML
window.addEventListener('message', (event) => {
    if (event.data.type === 'widget_init') {
        const widgetData = event.data.data;
        
        // Restore state
        audio.src = widgetData.song_url;
        audio.currentTime = widgetData.lastPosition || 0;
        
        // Track analytics
        console.log('Widget initialized:', widgetData);
    }
});
```

**2. Via iframe data attribute (Fallback)**

```javascript
// Inside widget HTML - read from parent iframe element
const iframe = window.frameElement;
if (iframe) {
    const widgetData = JSON.parse(iframe.dataset.widgetProps || '{}');
    console.log('Widget data:', widgetData);
}
```

### Example: State Restoration

```python
@widget(name="stateful_player")
async def stateful_player(self, song_url: str, last_position: float = 0.0) -> str:
    widget_data = {
        'song_url': song_url,
        'last_position': last_position  # Resume from where user left off
    }
    
    html_content = f"""
    <audio id="audio" src="{song_url}"></audio>
    <script>
        window.addEventListener('message', (e) => {{
            if (e.data.type === 'widget_init') {{
                const audio = document.getElementById('audio');
                audio.currentTime = e.data.data.last_position || 0;
                audio.play();
            }}
        }});
    </script>
    """
    
    escaped_data = html.escape(json.dumps(widget_data), quote=True)
    return f'<widget id="player" data="{escaped_data}">{html_content}</widget>'
```

**Note**: Simple widgets don't need a `data` attribute. Only use it when you need state persistence, analytics, or dynamic configuration.

## Browser Detection

Widgets are only included in the LLM context for requests from browser clients. The system automatically detects browsers by checking the User-Agent header for common markers:

- Mozilla
- Chrome
- Safari
- Firefox
- Edge

This ensures widgets aren't offered to API clients or non-browser interfaces where they cannot be rendered.

## Security

### Sandboxing

WebAgents widgets render in sandboxed iframes with the following restrictions:

```typescript
<iframe sandbox="allow-scripts allow-same-origin" />
```

This provides:

- **Isolated execution**: Widget JavaScript cannot access parent window
- **No cookies/storage access**: Widgets cannot read chat history or user data
- **Blob URLs**: Content served from memory, no network access
- **Script execution**: JavaScript enabled for interactivity
- **Same-origin**: Allows styling and modern browser APIs

### XSS Prevention

WebAgents widgets are **secure by default** with automatic HTML escaping:

- **Auto-escaping enabled**: All string arguments are automatically HTML-escaped (`auto_escape=True` by default)
- **No boilerplate needed**: Just use variables directly in your HTML - they're already safe
- **Escape hatch available**: Set `auto_escape=False` if you need to pass pre-rendered HTML (use with caution)
- **Blob URL rendering**: Widget content is rendered from blob URLs, not inline HTML

Example (secure by default):
```python
@widget(name="safe_widget")  # auto_escape=True
async def safe_widget(self, user_input: str) -> str:
    # user_input is automatically escaped!
    return f'<widget><div>{user_input}</div></widget>'
```

For pre-rendered HTML (advanced):
```python
@widget(name="unsafe_widget", auto_escape=False)
async def unsafe_widget(self, trusted_html: str) -> str:
    # trusted_html is NOT escaped - ensure it's safe!
    return f'<widget><div>{trusted_html}</div></widget>'
```

### Communication

Widgets communicate with the chat interface **only** through the postMessage API:

```javascript
// From widget → chat
window.parent.postMessage({
    type: 'widget_message',
    content: 'User message text'
}, '*');
```

The frontend:
1. Listens for `widget_message` events
2. Validates the message structure
3. Appends the content as a user message
4. Triggers the agent to process the new message

This one-way communication ensures widgets cannot read chat history or inject arbitrary code.

## Template Rendering

### Using Jinja2 Templates

For complex widgets, use Jinja2 templates:

```python
from webagents.agents.widgets import ChatKitRenderer

renderer = ChatKitRenderer(template_dir="widgets")

@widget(name="complex_widget", template="complex.html")
async def complex_widget(self, data: dict) -> str:
    html = renderer.render("complex.html", data)
    escaped_data = renderer.escape_data(json.dumps(data))
    return f'<widget kind="webagents" id="complex" data="{escaped_data}">{html}</widget>'
```

### Inline HTML

For simple widgets, use inline HTML with template strings:

```python
@widget
async def simple_widget(self, text: str) -> str:
    html = f"<div class='p-4'>{html.escape(text)}</div>"
    return f'<widget kind="webagents" id="simple">{html}</widget>'
```

## Styling with Tailwind CSS

Widgets should use Tailwind CSS for styling. Include the CDN in your HTML:

```html
<script src="https://cdn.tailwindcss.com"></script>
```

The `ChatKitRenderer.inject_tailwind_cdn()` helper automatically injects Tailwind if not present:

```python
from webagents.agents.widgets import ChatKitRenderer

html = ChatKitRenderer.inject_tailwind_cdn(my_html)
```

## Widget vs Tool

Use widgets when:
- You need rich interactive UI (forms, players, visualizations)
- User needs to interact with results (buttons, controls)
- Visual presentation is important (charts, cards, galleries)

Use tools when:
- Simple data fetching or processing
- No user interaction needed
- Text-based output is sufficient

## OpenAI ChatKit Widgets

The system also supports OpenAI ChatKit widgets, which use a different rendering engine:

```xml
<widget kind="openai">
{
  "$kind": "card",
  "content": [
    { "$kind": "text", "content": "Hello World" }
  ]
}
</widget>
```

These use `kind="openai"` and are rendered using the existing `WidgetRenderer` component (not sandboxed iframes).

## Best Practices

1. **Keep widgets focused**: Each widget should do one thing well
2. **Validate inputs**: Always escape user-provided data
3. **Test in sandbox**: Ensure your widget works with sandbox restrictions
4. **Handle errors gracefully**: Provide fallback UI for failed operations
5. **Optimize size**: Keep HTML/CSS/JS minimal for fast loading
6. **Use semantic HTML**: Ensure accessibility with proper markup
7. **Provide feedback**: Use loading states and success/error messages
8. **Mobile-friendly**: Use responsive Tailwind classes

## Debugging

### Backend Debugging

Enable debug logging to see widget registration:

```python
self.logger.debug(f"🎨 Widget registered: {widget_name}")
```

### Frontend Debugging

Check browser console for widget messages:

```javascript
console.log('[SkillWidget] Message from widget:', message);
```

### Common Issues

**Widget not appearing**:
- Check if User-Agent is from a browser
- Verify `<widget>` tags are properly formatted
- Ensure `kind` attribute is set correctly

**postMessage not working**:
- Verify `type: 'widget_message'` is set
- Check sandbox allows `allow-scripts`
- Ensure target is `'*'` for blob URLs

**Styling issues**:
- Confirm Tailwind CDN is included
- Check for conflicting styles
- Test with `colorScheme: 'normal'` on iframe

## Example Use Cases

### Music Player
Play audio tracks with controls for next/previous songs.

### Data Visualization
Render charts and graphs with interactive tooltips and legends.

### Forms
Collect structured input from users (surveys, feedback, configuration).

### Games
Simple interactive games or puzzles within the chat.

### Media Galleries
Display images, videos, or documents with navigation controls.

### Custom Controls
Specialized UI controls (color pickers, sliders, date pickers).

## API Reference

### @widget Decorator

```python
@widget(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    template: Optional[str] = None,
    scope: Union[str, List[str]] = "all"
)
```

### ChatKitRenderer Class

```python
class ChatKitRenderer:
    def __init__(self, template_dir: Optional[str] = None)
    
    def render(self, template_name: str, context: Dict[str, Any]) -> str
    
    def render_inline(self, html_string: str, context: Dict[str, Any]) -> str
    
    @staticmethod
    def escape_data(data: Any) -> str
    
    @staticmethod
    def inject_tailwind_cdn(html_content: str) -> str
```

### BaseAgent Methods

```python
def get_all_widgets(self) -> List[Dict[str, Any]]
def register_widget(self, widget_func: Callable, source: str = "manual", scope: Union[str, List[str]] = None)
def _is_browser_request(self, context=None) -> bool
```

## See Also

- [Tools Documentation](tools.md) - For simple tool-based interactions
- [Handoffs Documentation](handoffs.md) - For agent delegation
- [Skills Documentation](../skills/overview.md) - For skill development

