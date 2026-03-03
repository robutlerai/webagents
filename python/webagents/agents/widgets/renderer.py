"""Widget template renderer for HTML templates with Jinja2 support"""

import os
import warnings
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
import html


class WidgetTemplateRenderer:
    """Renders WebAgents widget HTML from Jinja2 templates or inline HTML strings
    
    This renderer is specifically for WebAgents HTML widgets (kind="webagents"),
    NOT for OpenAI ChatKit widgets (kind="openai").
    
    Supports both file-based templates and inline HTML with variable substitution.
    Ensures proper HTML escaping for data attributes to prevent XSS.
    
    Example:
        # File-based template
        renderer = WidgetTemplateRenderer(template_dir="widgets")
        html = renderer.render("music_player.html", {"title": "Song Name"})
        
        # Inline HTML
        renderer = WidgetTemplateRenderer()
        html = renderer.render_inline("<div>{{ title }}</div>", {"title": "Song Name"})
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """Initialize widget renderer
        
        Args:
            template_dir: Directory containing Jinja2 template files (optional)
        """
        self.template_dir = template_dir or "widgets"
        
        # Initialize Jinja2 environment if template directory exists
        if os.path.exists(self.template_dir):
            self.env = Environment(
                loader=FileSystemLoader(self.template_dir),
                autoescape=True  # Auto-escape HTML for security
            )
        else:
            self.env = None
    
    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render widget HTML from a template file
        
        Args:
            template_name: Name of template file (e.g., "music_player.html")
            context: Dictionary of variables to pass to template
        
        Returns:
            Rendered HTML string
        
        Raises:
            TemplateNotFound: If template file doesn't exist
            ValueError: If template directory not configured
        """
        if not self.env:
            raise ValueError(f"Template directory '{self.template_dir}' not found")
        
        template = self.env.get_template(template_name)
        return template.render(**context)
    
    def render_inline(self, html_string: str, context: Dict[str, Any]) -> str:
        """Render widget HTML from an inline string
        
        Args:
            html_string: HTML string with Jinja2 template syntax
            context: Dictionary of variables to pass to template
        
        Returns:
            Rendered HTML string
        """
        template = Template(html_string, autoescape=True)
        return template.render(**context)
    
    @staticmethod
    def escape_data(data: Any) -> str:
        """Escape data for safe inclusion in HTML attributes
        
        Args:
            data: Data to escape (will be converted to string)
        
        Returns:
            HTML-escaped string safe for attribute values
        """
        return html.escape(str(data), quote=True)
    
    @staticmethod
    def inject_tailwind_cdn(html_content: str) -> str:
        """Inject Tailwind CSS CDN if not already present
        
        Args:
            html_content: HTML content to check and modify
        
        Returns:
            HTML content with Tailwind CDN injected in <head>
        """
        # Check if Tailwind is already present
        if 'tailwindcss.com' in html_content:
            return html_content
        
        # Check if there's a <head> tag
        if '<head>' in html_content:
            # Inject after <head>
            return html_content.replace(
                '<head>',
                '<head>\n    <script src="https://cdn.tailwindcss.com"></script>',
                1  # Only replace first occurrence
            )
        elif '<html>' in html_content:
            # Inject after <html>
            return html_content.replace(
                '<html>',
                '<html>\n<head>\n    <script src="https://cdn.tailwindcss.com"></script>\n</head>',
                1
            )
        else:
            # No HTML structure, wrap in basic HTML with Tailwind
            return f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
{html_content}
</body>
</html>"""


# Backward compatibility alias with deprecation warning
class ChatKitRenderer(WidgetTemplateRenderer):
    """Deprecated: Use WidgetTemplateRenderer instead.
    
    This class was misnamed - it renders WebAgents HTML widgets using Jinja2,
    NOT OpenAI ChatKit widgets. Use WidgetTemplateRenderer for clarity.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ChatKitRenderer is deprecated and will be removed in a future version. "
            "Use WidgetTemplateRenderer instead. Note: This class renders WebAgents HTML widgets, "
            "not OpenAI ChatKit widgets.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

