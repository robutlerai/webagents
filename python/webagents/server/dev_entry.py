"""
Dev entry point for webagentsd with auto-reload support.
This file is used by uvicorn for --reload.

Supports configuration via:
- WEBAGENTS_EXTENSION_CONFIG: Path to JSON config file with extensions
- WEBAGENTS_WATCH_DIRS: Comma-separated list of directories to watch
"""
from webagents.server.core.app import create_server
from pathlib import Path
import os
import json

# Load extension config from file if specified
extension_config = None
config_path = os.environ.get("WEBAGENTS_EXTENSION_CONFIG")
if config_path and Path(config_path).exists():
    try:
        with open(config_path) as f:
            extension_config = json.load(f)
        print(f"[webagentsd] Loaded config from {config_path}")
    except Exception as e:
        print(f"[webagentsd] Warning: Failed to load config from {config_path}: {e}")

# Get watch dirs from env or default to cwd
watch_dirs_str = os.environ.get("WEBAGENTS_WATCH_DIRS")
watch_dirs = [Path(d) for d in watch_dirs_str.split(",")] if watch_dirs_str else [Path.cwd()]

# Get settings from config or use defaults
title = extension_config.get("title", "WebAgents Daemon (Dev)") if extension_config else "WebAgents Daemon (Dev)"
description = extension_config.get("description", "Local agent daemon with auto-reload") if extension_config else "Local agent daemon with auto-reload"
version = extension_config.get("version", "0.2.3") if extension_config else "0.2.3"
url_prefix = extension_config.get("url_prefix", "/agents") if extension_config else "/agents"
enable_file_watching = extension_config.get("enable_file_watching", True) if extension_config else True
enable_cron = extension_config.get("enable_cron", True) if extension_config else True
enable_monitoring = extension_config.get("enable_monitoring", False) if extension_config else False

# Create app instance
server = create_server(
    title=title,
    description=description,
    version=version,
    url_prefix=url_prefix,
    enable_file_watching=enable_file_watching,
    watch_dirs=watch_dirs,
    enable_cron=enable_cron,
    enable_monitoring=enable_monitoring,
    storage_backend="json",
    extension_config=extension_config  # Pass full config for extensions
)

app = server.app
