"""
Service Installation

Generate systemd and launchd service files.
"""

from .systemd import generate_systemd_unit, install_systemd
from .launchd import generate_launchd_plist, install_launchd

__all__ = [
    "generate_systemd_unit",
    "install_systemd",
    "generate_launchd_plist",
    "install_launchd",
]
