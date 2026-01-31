"""
Hook Runner - Plugin Hook Execution

Executes plugin hooks defined in hooks.json configuration files.

Hook configuration format (hooks.json):
{
  "hooks": [
    {
      "event": "on_message",
      "handler": "./hooks/on_message.py",
      "priority": 50
    }
  ]
}
"""

import asyncio
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HookConfig:
    """Hook configuration.
    
    Attributes:
        event: Event name to handle (e.g., "on_message", "on_tool_call")
        handler: Path to handler file or module.function
        priority: Execution priority (lower = earlier)
        enabled: Whether hook is active
        plugin_name: Source plugin name
    """
    event: str
    handler: str
    priority: int = 50
    enabled: bool = True
    plugin_name: str = ""


class HookRunner:
    """Execute plugin hooks.
    
    Supports:
    - Loading hooks from hooks.json configuration
    - Python file handlers
    - Module.function handlers
    - Priority-based execution order
    """
    
    def __init__(self):
        """Initialize hook runner."""
        self._loaded_handlers: Dict[str, Callable] = {}
    
    def load_hooks_config(self, path: Path, plugin_name: str = "") -> List[HookConfig]:
        """Load hooks from configuration file.
        
        Args:
            path: Path to hooks.json file
            plugin_name: Source plugin name
            
        Returns:
            List of HookConfig instances
        """
        if not path.exists():
            return []
        
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            logger.error(f"Invalid hooks.json: {e}")
            return []
        
        hooks = []
        for hook_data in data.get("hooks", []):
            try:
                hooks.append(HookConfig(
                    event=hook_data["event"],
                    handler=hook_data["handler"],
                    priority=hook_data.get("priority", 50),
                    enabled=hook_data.get("enabled", True),
                    plugin_name=plugin_name,
                ))
            except KeyError as e:
                logger.warning(f"Invalid hook config, missing: {e}")
        
        return hooks
    
    async def execute_hook(
        self,
        hook: HookConfig,
        plugin_path: Path,
        event_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a single hook.
        
        Args:
            hook: Hook configuration
            plugin_path: Plugin base path for resolving handlers
            event_data: Data to pass to hook handler
            
        Returns:
            Hook execution result
        """
        if not hook.enabled:
            return {"skipped": True, "reason": "disabled"}
        
        event_data = event_data or {}
        
        try:
            handler = await self._load_handler(hook.handler, plugin_path)
            
            if asyncio.iscoroutinefunction(handler):
                result = await handler(event_data)
            else:
                result = handler(event_data)
            
            return {
                "status": "success",
                "event": hook.event,
                "result": result,
            }
        except Exception as e:
            logger.error(f"Hook execution failed ({hook.event}): {e}")
            return {
                "status": "error",
                "event": hook.event,
                "error": str(e),
            }
    
    async def execute_hooks(
        self,
        hooks: List[HookConfig],
        event: str,
        plugin_path: Path,
        event_data: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute all hooks for an event.
        
        Args:
            hooks: List of hook configurations
            event: Event name to filter by
            plugin_path: Plugin base path
            event_data: Data to pass to handlers
            
        Returns:
            List of execution results
        """
        # Filter hooks for this event
        matching_hooks = [h for h in hooks if h.event == event and h.enabled]
        
        # Sort by priority
        matching_hooks.sort(key=lambda h: h.priority)
        
        results = []
        for hook in matching_hooks:
            result = await self.execute_hook(hook, plugin_path, event_data)
            results.append(result)
            
            # Check if hook wants to stop propagation
            if result.get("stop_propagation"):
                break
        
        return results
    
    async def _load_handler(
        self,
        handler_spec: str,
        plugin_path: Path
    ) -> Callable:
        """Load a hook handler.
        
        Supports:
        - "./path/to/file.py" - Python file with run() function
        - "./path/to/file.py:function_name" - Specific function
        - "module.submodule:function" - Module import
        
        Args:
            handler_spec: Handler specification string
            plugin_path: Plugin base path
            
        Returns:
            Callable handler function
        """
        cache_key = f"{plugin_path}:{handler_spec}"
        if cache_key in self._loaded_handlers:
            return self._loaded_handlers[cache_key]
        
        handler: Callable
        
        if handler_spec.startswith("./") or handler_spec.startswith("/"):
            # File-based handler
            if ":" in handler_spec:
                file_path, func_name = handler_spec.rsplit(":", 1)
            else:
                file_path = handler_spec
                func_name = "run"  # Default function name
            
            # Resolve path
            if not file_path.startswith("/"):
                file_path = str(plugin_path / file_path.lstrip("./"))
            
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Handler file not found: {file_path}")
            
            # Load module from file
            spec = importlib.util.spec_from_file_location(
                f"plugin_hook_{file_path.stem}",
                file_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load handler: {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            
            handler = getattr(module, func_name)
        else:
            # Module-based handler
            if ":" in handler_spec:
                module_name, func_name = handler_spec.rsplit(":", 1)
            else:
                module_name = handler_spec
                func_name = "run"
            
            module = importlib.import_module(module_name)
            handler = getattr(module, func_name)
        
        self._loaded_handlers[cache_key] = handler
        return handler
    
    def clear_cache(self) -> None:
        """Clear loaded handler cache."""
        self._loaded_handlers.clear()
