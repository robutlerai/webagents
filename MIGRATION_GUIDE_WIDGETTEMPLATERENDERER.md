# Migration Guide: ChatKitRenderer → WidgetTemplateRenderer

## Overview

The backend class `ChatKitRenderer` has been renamed to `WidgetTemplateRenderer` to avoid confusion with OpenAI ChatKit widgets. The old name is still available as a deprecated alias.

## What Changed

### Before (Deprecated)
```python
from webagents.agents.widgets import ChatKitRenderer

renderer = ChatKitRenderer(template_dir="widgets")
html = renderer.render("template.html", context)
```

### After (Recommended)
```python
from webagents.agents.widgets import WidgetTemplateRenderer

renderer = WidgetTemplateRenderer(template_dir="widgets")
html = renderer.render("template.html", context)
```

## Why the Change?

The class was misnamed - it's a **Jinja2 template renderer** for WebAgents HTML widgets, not for OpenAI ChatKit widgets. The frontend has a separate `ChatKitRenderer` function that actually renders ChatKit widgets, causing confusion.

## Backward Compatibility

✅ **Your existing code will continue to work** - the old `ChatKitRenderer` name is still available as a deprecated alias.

However, you will see a deprecation warning:
```
DeprecationWarning: ChatKitRenderer is deprecated and will be removed in a future version.
Use WidgetTemplateRenderer instead. Note: This class renders WebAgents HTML widgets, not OpenAI ChatKit widgets.
```

## Migration Steps

### 1. Update Imports

**Search and replace** across your codebase:

```python
# Old
from webagents.agents.widgets import ChatKitRenderer

# New
from webagents.agents.widgets import WidgetTemplateRenderer
```

### 2. Update Class Instantiation

```python
# Old
renderer = ChatKitRenderer(template_dir="widgets")

# New
renderer = WidgetTemplateRenderer(template_dir="widgets")
```

### 3. Update Static Method Calls

```python
# Old
html = ChatKitRenderer.inject_tailwind_cdn(html_content)

# New
html = WidgetTemplateRenderer.inject_tailwind_cdn(html_content)
```

## No Changes Needed For

- The `@widget` decorator - still works the same
- Widget function signatures
- HTML content and templates
- Frontend components
- Widget data attributes

## Timeline

- **Now**: Both names work (old name shows deprecation warning)
- **Future release**: Old name will be removed

## API Compatibility

All methods remain identical:

```python
class WidgetTemplateRenderer:
    def __init__(self, template_dir: Optional[str] = None)
    def render(self, template_name: str, context: Dict[str, Any]) -> str
    def render_inline(self, html_string: str, context: Dict[str, Any]) -> str
    
    @staticmethod
    def escape_data(data: Any) -> str
    
    @staticmethod
    def inject_tailwind_cdn(html_content: str) -> str
```

## Questions?

See:
- `docs/agent/widgets.md` - Updated widget documentation
- `WIDGET_SYSTEM_SUMMARY.md` - Complete architecture overview

