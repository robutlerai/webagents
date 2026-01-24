"""
Extension Loader

Load extensions from configuration.
"""

import importlib
import logging
import warnings
from typing import List, Dict, Any

from .interface import WebAgentsExtension

logger = logging.getLogger("webagents.extensions")


def load_extensions(config: Dict[str, Any]) -> List[WebAgentsExtension]:
    """Load extensions from configuration
    
    Args:
        config: Extension configuration dict with "extensions" list
                (also supports deprecated "plugins" key for backwards compatibility)
    
    Returns:
        List of loaded extension instances
    
    Example config:
        {
            "extensions": [
                {
                    "module": "agents_extension.extension",
                    "class": "AgentsExtension",
                    "config": {"mode": "portal"}
                }
            ]
        }
    """
    extensions = []
    
    # Support both "extensions" (new) and "plugins" (deprecated) keys
    extension_specs = config.get("extensions", [])
    if not extension_specs and "plugins" in config:
        warnings.warn(
            "Using 'plugins' key is deprecated, use 'extensions' instead",
            DeprecationWarning,
            stacklevel=2
        )
        extension_specs = config.get("plugins", [])
    
    for ext_spec in extension_specs:
        module_path = ext_spec["module"]
        class_name = ext_spec.get("class", "Extension")
        ext_config = ext_spec.get("config", {})
        
        try:
            module = importlib.import_module(module_path)
            ext_class = getattr(module, class_name)
            extension = ext_class(ext_config)
            extensions.append(extension)
            logger.debug(f"Loaded extension: {module_path}.{class_name}")
        except Exception as e:
            # Log error but don't fail - allow server to start with available extensions
            logger.warning(f"Failed to load extension {module_path}: {e}")
    
    return extensions


# Backwards compatibility alias (deprecated)
def load_plugins(config: Dict[str, Any]) -> List[WebAgentsExtension]:
    """Deprecated: Use load_extensions instead."""
    warnings.warn(
        "load_plugins is deprecated, use load_extensions instead",
        DeprecationWarning,
        stacklevel=2
    )
    return load_extensions(config)
