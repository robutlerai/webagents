"""
Plugin Loader

Load plugins from configuration.
"""

import importlib
from typing import List, Dict, Any

from .interface import WebAgentsPlugin


def load_plugins(config: Dict[str, Any]) -> List[WebAgentsPlugin]:
    """Load plugins from configuration
    
    Args:
        config: Plugin configuration dict with "plugins" list
    
    Returns:
        List of loaded plugin instances
    
    Example config:
        {
            "plugins": [
                {
                    "module": "agents_plugin.plugin",
                    "class": "AgentsPlugin",
                    "config": {"mode": "portal"}
                }
            ]
        }
    """
    plugins = []
    
    for plugin_spec in config.get("plugins", []):
        module_path = plugin_spec["module"]
        class_name = plugin_spec.get("class", "Plugin")
        plugin_config = plugin_spec.get("config", {})
        
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            plugin = plugin_class(plugin_config)
            plugins.append(plugin)
        except Exception as e:
            # Log error but don't fail - allow server to start with available plugins
            print(f"Warning: Failed to load plugin {module_path}: {e}")
    
    return plugins
