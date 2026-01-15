"""
Local State Tests

Test .webagents/ state management.
"""

import pytest
from pathlib import Path
import tempfile

from webagents.cli.state.local import LocalState


class TestLocalState:
    """Test local state management."""
    
    def test_init_creates_directories(self):
        """Test that init creates directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = LocalState(project_dir=Path(tmpdir))
            
            assert (Path(tmpdir) / ".webagents").exists()
            assert (Path(tmpdir) / ".webagents" / "sessions").exists()
            assert (Path(tmpdir) / ".webagents" / "logs").exists()
            assert (Path(tmpdir) / ".webagents" / "cache").exists()
    
    def test_config_get_set(self):
        """Test config get and set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = LocalState(project_dir=Path(tmpdir))
            
            state.set_config("test_key", "test_value")
            value = state.get_config("test_key")
            
            assert value == "test_value"
    
    def test_config_default(self):
        """Test config default value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = LocalState(project_dir=Path(tmpdir))
            
            value = state.get_config("nonexistent", default="default")
            
            assert value == "default"
    
    def test_credentials(self):
        """Test credentials management."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = LocalState(project_dir=Path(tmpdir))
            
            state.set_credentials(token="abc123", user="test")
            creds = state.get_credentials()
            
            assert creds["token"] == "abc123"
            assert creds["user"] == "test"
    
    def test_clear_credentials(self):
        """Test clearing credentials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = LocalState(project_dir=Path(tmpdir))
            
            state.set_credentials(token="abc")
            state.clear_credentials()
            creds = state.get_credentials()
            
            assert creds == {}
    
    def test_state_update(self):
        """Test runtime state update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = LocalState(project_dir=Path(tmpdir))
            
            state.set_state(running=True, agent="test")
            current = state.get_state()
            
            assert current["running"] == True
            assert current["agent"] == "test"
            assert "updated_at" in current
