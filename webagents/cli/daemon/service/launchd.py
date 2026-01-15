"""
Launchd Service Generator

Generate and install launchd plist files for macOS.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List


LAUNCHD_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.robutler.webagentsd</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>webagents</string>
        <string>daemon</string>
        <string>start</string>
        <string>--port</string>
        <string>{port}</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    
    <key>StandardOutPath</key>
    <string>{log_dir}/webagentsd.log</string>
    
    <key>StandardErrorPath</key>
    <string>{log_dir}/webagentsd.error.log</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>{home}</string>
        <key>PATH</key>
        <string>{path}</string>
    </dict>
</dict>
</plist>
"""


def generate_launchd_plist(
    port: int = 8765,
    watch_dirs: Optional[List[Path]] = None,
    working_dir: Optional[Path] = None,
) -> str:
    """Generate launchd plist file content.
    
    Args:
        port: Daemon port
        watch_dirs: Directories to watch
        working_dir: Working directory
        
    Returns:
        Plist file content
    """
    home = os.environ.get("HOME", "/Users/nobody")
    working_dir = working_dir or Path.home()
    python_path = sys.executable
    path = os.environ.get("PATH", "/usr/bin:/bin")
    log_dir = Path.home() / "Library" / "Logs" / "webagentsd"
    
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    return LAUNCHD_PLIST_TEMPLATE.format(
        python_path=python_path,
        port=port,
        working_dir=working_dir,
        home=home,
        path=path,
        log_dir=log_dir,
    )


def install_launchd(
    port: int = 8765,
    watch_dirs: Optional[List[Path]] = None,
) -> Path:
    """Install launchd service.
    
    Args:
        port: Daemon port
        watch_dirs: Directories to watch
        
    Returns:
        Path to installed plist file
    """
    plist_content = generate_launchd_plist(port=port, watch_dirs=watch_dirs)
    
    # User LaunchAgents directory
    launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
    launch_agents_dir.mkdir(parents=True, exist_ok=True)
    
    plist_path = launch_agents_dir / "ai.robutler.webagentsd.plist"
    plist_path.write_text(plist_content)
    
    print(f"Installed service: {plist_path}")
    print()
    print("To load and start the service:")
    print(f"  launchctl load {plist_path}")
    print()
    print("To check status:")
    print("  launchctl list | grep webagentsd")
    print()
    print("View logs at:")
    print(f"  ~/Library/Logs/webagentsd/webagentsd.log")
    
    return plist_path


def uninstall_launchd() -> bool:
    """Uninstall launchd service.
    
    Returns:
        True if uninstalled
    """
    plist_path = Path.home() / "Library" / "LaunchAgents" / "ai.robutler.webagentsd.plist"
    
    if plist_path.exists():
        # Unload first
        os.system(f"launchctl unload {plist_path} 2>/dev/null")
        
        plist_path.unlink()
        print(f"Removed: {plist_path}")
        return True
    
    return False


def is_launchd_installed() -> bool:
    """Check if launchd service is installed."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "ai.robutler.webagentsd.plist"
    return plist_path.exists()


def is_launchd_running() -> bool:
    """Check if launchd service is running."""
    import subprocess
    result = subprocess.run(
        ["launchctl", "list"],
        capture_output=True,
        text=True
    )
    return "ai.robutler.webagentsd" in result.stdout
