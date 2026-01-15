"""
Agent Templates

Bundled templates for quick agent creation.
"""

from pathlib import Path
from typing import List, Dict, Optional


# Bundled templates directory
TEMPLATES_DIR = Path(__file__).parent


def list_templates() -> List[Dict]:
    """List available bundled templates.
    
    Returns:
        List of template info dicts
    """
    templates = []
    
    for template_dir in TEMPLATES_DIR.iterdir():
        if template_dir.is_dir() and not template_dir.name.startswith("_"):
            template_file = template_dir / "TEMPLATE.md"
            if template_file.exists():
                # Parse template metadata
                content = template_file.read_text()
                info = _parse_template_info(content)
                info["name"] = template_dir.name
                info["path"] = str(template_file)
                templates.append(info)
    
    return templates


def get_template(name: str) -> Optional[str]:
    """Get template content by name.
    
    Args:
        name: Template name
        
    Returns:
        Template content or None
    """
    template_file = TEMPLATES_DIR / name / "TEMPLATE.md"
    if template_file.exists():
        return template_file.read_text()
    return None


def _parse_template_info(content: str) -> Dict:
    """Parse template metadata from content.
    
    Args:
        content: Template file content
        
    Returns:
        Template info dict
    """
    import re
    import yaml
    
    info = {"description": ""}
    
    # Extract YAML frontmatter
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if match:
        try:
            data = yaml.safe_load(match.group(1))
            info["description"] = data.get("description", "")
        except Exception:
            pass
    
    return info
