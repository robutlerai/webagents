"""
Systemd Service Generator

Generate and install systemd unit files for Linux.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List


SYSTEMD_UNIT_TEMPLATE = """[Unit]
Description=WebAgents Daemon
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={working_dir}
ExecStart={python_path} -m webagents daemon start --port {port}
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Environment
Environment="HOME={home}"
Environment="PATH={path}"

[Install]
WantedBy=multi-user.target
"""


def generate_systemd_unit(
    port: int = 8765,
    watch_dirs: Optional[List[Path]] = None,
    user: Optional[str] = None,
    working_dir: Optional[Path] = None,
) -> str:
    """Generate systemd unit file content.
    
    Args:
        port: Daemon port
        watch_dirs: Directories to watch
        user: User to run as
        working_dir: Working directory
        
    Returns:
        Unit file content
    """
    user = user or os.environ.get("USER", "nobody")
    home = os.environ.get("HOME", f"/home/{user}")
    working_dir = working_dir or Path.home()
    python_path = sys.executable
    path = os.environ.get("PATH", "/usr/bin:/bin")
    
    return SYSTEMD_UNIT_TEMPLATE.format(
        user=user,
        home=home,
        working_dir=working_dir,
        python_path=python_path,
        port=port,
        path=path,
    )


def install_systemd(
    port: int = 8765,
    watch_dirs: Optional[List[Path]] = None,
    user_mode: bool = True,
) -> Path:
    """Install systemd service.
    
    Args:
        port: Daemon port
        watch_dirs: Directories to watch
        user_mode: Install as user service (not system-wide)
        
    Returns:
        Path to installed unit file
    """
    unit_content = generate_systemd_unit(port=port, watch_dirs=watch_dirs)
    
    if user_mode:
        # User service directory
        service_dir = Path.home() / ".config" / "systemd" / "user"
    else:
        # System service directory (requires root)
        service_dir = Path("/etc/systemd/system")
    
    service_dir.mkdir(parents=True, exist_ok=True)
    
    unit_path = service_dir / "webagentsd.service"
    unit_path.write_text(unit_content)
    
    print(f"Installed service: {unit_path}")
    print()
    print("To enable and start the service:")
    if user_mode:
        print("  systemctl --user daemon-reload")
        print("  systemctl --user enable webagentsd")
        print("  systemctl --user start webagentsd")
        print()
        print("To check status:")
        print("  systemctl --user status webagentsd")
    else:
        print("  sudo systemctl daemon-reload")
        print("  sudo systemctl enable webagentsd")
        print("  sudo systemctl start webagentsd")
        print()
        print("To check status:")
        print("  sudo systemctl status webagentsd")
    
    return unit_path


def uninstall_systemd(user_mode: bool = True) -> bool:
    """Uninstall systemd service.
    
    Args:
        user_mode: User service mode
        
    Returns:
        True if uninstalled
    """
    if user_mode:
        unit_path = Path.home() / ".config" / "systemd" / "user" / "webagentsd.service"
    else:
        unit_path = Path("/etc/systemd/system/webagentsd.service")
    
    if unit_path.exists():
        unit_path.unlink()
        print(f"Removed: {unit_path}")
        print()
        if user_mode:
            print("Run: systemctl --user daemon-reload")
        else:
            print("Run: sudo systemctl daemon-reload")
        return True
    
    return False
