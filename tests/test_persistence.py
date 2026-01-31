"""
Tests for the persistence utilities (SessionManager, CheckpointManager).
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime


class TestSessionManager:
    """Tests for SessionManager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests"""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_session_manager_init(self, temp_dir):
        """Test SessionManager initialization"""
        from webagents.agents.skills.local.session.skill import SessionManager
        
        manager = SessionManager(temp_dir, "test-agent")
        
        assert manager.sessions_dir.exists()
        assert manager.agent_name == "test-agent"
    
    def test_save_and_load_session(self, temp_dir):
        """Test saving and loading a session"""
        from webagents.agents.skills.local.session.skill import SessionManager, Session, Message
        
        manager = SessionManager(temp_dir, "test-agent")
        
        # Create session
        session = Session(
            session_id="test-session-123",
            agent_name="test-agent",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ],
            input_tokens=10,
            output_tokens=5,
        )
        
        # Save
        filepath = manager.save(session)
        assert filepath.exists()
        
        # Load
        loaded = manager.load("test-session-123")
        assert loaded is not None
        assert loaded.session_id == "test-session-123"
        assert len(loaded.messages) == 2
        assert loaded.input_tokens == 10
    
    def test_load_latest_session(self, temp_dir):
        """Test loading the latest session"""
        from webagents.agents.skills.local.session.skill import SessionManager, Session
        
        manager = SessionManager(temp_dir, "test-agent")
        
        # Create and save multiple sessions
        for i in range(3):
            session = Session(
                session_id=f"session-{i}",
                agent_name="test-agent",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )
            manager.save(session)
        
        # Load latest
        latest = manager.load_latest()
        assert latest is not None
        assert latest.session_id == "session-2"  # Last saved
    
    def test_load_latest_with_max_messages(self, temp_dir):
        """Test loading latest session with message limit"""
        from webagents.agents.skills.local.session.skill import SessionManager, Session, Message
        
        manager = SessionManager(temp_dir, "test-agent")
        
        # Create session with many messages
        messages = [Message(role="user", content=f"Message {i}") for i in range(50)]
        session = Session(
            session_id="big-session",
            agent_name="test-agent",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            messages=messages,
        )
        manager.save(session)
        
        # Load with limit
        loaded = manager.load_latest(max_messages=10)
        assert loaded is not None
        assert len(loaded.messages) == 10
    
    def test_list_sessions(self, temp_dir):
        """Test listing sessions"""
        from webagents.agents.skills.local.session.skill import SessionManager, Session
        
        manager = SessionManager(temp_dir, "test-agent")
        
        # Create sessions
        for i in range(5):
            session = Session(
                session_id=f"session-{i}",
                agent_name="test-agent",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )
            manager.save(session)
        
        # List
        sessions = manager.list_sessions()
        assert len(sessions) == 5
    
    def test_delete_session(self, temp_dir):
        """Test deleting a session"""
        from webagents.agents.skills.local.session.skill import SessionManager, Session
        
        manager = SessionManager(temp_dir, "test-agent")
        
        # Create and save
        session = Session(
            session_id="to-delete",
            agent_name="test-agent",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        manager.save(session)
        
        # Delete
        assert manager.delete("to-delete") is True
        assert manager.load("to-delete") is None
    
    def test_a2a_session_isolation(self, temp_dir):
        """Test A2A sessions are stored separately"""
        from webagents.agents.skills.local.session.skill import SessionManager, Session
        
        manager = SessionManager(temp_dir, "test-agent")
        
        # Save regular session
        session1 = Session(
            session_id="regular-session",
            agent_name="test-agent",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        manager.save(session1)
        
        # Save A2A session
        session2 = Session(
            session_id="a2a-session",
            agent_name="test-agent",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        manager.save(session2, conversation_id="conv-123")
        
        # Verify isolation
        regular = manager.load("regular-session")
        a2a = manager.load("a2a-session", conversation_id="conv-123")
        
        assert regular is not None
        assert a2a is not None
        
        # A2A session not visible in regular list
        regular_sessions = manager.list_sessions()
        assert all(s["session_id"] != "a2a-session" for s in regular_sessions)


class TestCheckpointManager:
    """Tests for CheckpointManager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with some files"""
        temp = tempfile.mkdtemp()
        temp_path = Path(temp)
        
        # Create some test files
        (temp_path / "file1.txt").write_text("Content 1")
        (temp_path / "file2.txt").write_text("Content 2")
        (temp_path / "subdir").mkdir()
        (temp_path / "subdir" / "file3.txt").write_text("Content 3")
        
        yield temp_path
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_checkpoint_manager_init(self, temp_dir):
        """Test CheckpointManager initialization"""
        from webagents.agents.skills.local.checkpoint.skill import CheckpointManager
        
        manager = CheckpointManager(temp_dir, "test-agent")
        
        assert manager.history_dir.exists()
        assert manager.checkpoints_dir.exists()
        assert (manager.history_dir / ".git").exists()
    
    def test_create_checkpoint(self, temp_dir):
        """Test creating a checkpoint"""
        from webagents.agents.skills.local.checkpoint.skill import CheckpointManager
        
        manager = CheckpointManager(temp_dir, "test-agent")
        
        checkpoint = manager.create(description="Test checkpoint")
        
        assert checkpoint.checkpoint_id is not None
        assert checkpoint.description == "Test checkpoint"
        assert checkpoint.commit_hash is not None
        assert len(checkpoint.files_changed) > 0
    
    def test_create_checkpoint_specific_files(self, temp_dir):
        """Test creating a checkpoint with specific files"""
        from webagents.agents.skills.local.checkpoint.skill import CheckpointManager
        
        manager = CheckpointManager(temp_dir, "test-agent")
        
        checkpoint = manager.create(
            description="Partial checkpoint",
            files=[temp_dir / "file1.txt"]
        )
        
        assert len(checkpoint.files_changed) == 1
        assert "file1.txt" in checkpoint.files_changed[0]
    
    def test_list_checkpoints(self, temp_dir):
        """Test listing checkpoints"""
        from webagents.agents.skills.local.checkpoint.skill import CheckpointManager
        
        manager = CheckpointManager(temp_dir, "test-agent")
        
        # Create multiple checkpoints
        for i in range(3):
            manager.create(description=f"Checkpoint {i}")
        
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3
    
    def test_get_checkpoint(self, temp_dir):
        """Test getting checkpoint by ID"""
        from webagents.agents.skills.local.checkpoint.skill import CheckpointManager
        
        manager = CheckpointManager(temp_dir, "test-agent")
        
        created = manager.create(description="My checkpoint")
        
        retrieved = manager.get_checkpoint(created.checkpoint_id)
        assert retrieved is not None
        assert retrieved.description == "My checkpoint"
    
    def test_restore_checkpoint(self, temp_dir):
        """Test restoring a checkpoint"""
        from webagents.agents.skills.local.checkpoint.skill import CheckpointManager
        
        manager = CheckpointManager(temp_dir, "test-agent")
        
        # Create initial checkpoint
        checkpoint = manager.create(description="Initial state")
        
        # Modify files
        (temp_dir / "file1.txt").write_text("Modified content")
        
        # Restore
        success = manager.restore(checkpoint.checkpoint_id)
        assert success is True
        
        # Verify restoration
        content = (temp_dir / "file1.txt").read_text()
        assert content == "Content 1"  # Original content
    
    def test_delete_checkpoint(self, temp_dir):
        """Test deleting a checkpoint"""
        from webagents.agents.skills.local.checkpoint.skill import CheckpointManager
        
        manager = CheckpointManager(temp_dir, "test-agent")
        
        checkpoint = manager.create(description="To delete")
        
        # Delete
        assert manager.delete(checkpoint.checkpoint_id) is True
        assert manager.get_checkpoint(checkpoint.checkpoint_id) is None


class TestMessage:
    """Tests for Message dataclass"""
    
    def test_message_to_dict(self):
        """Test Message to_dict conversion"""
        from webagents.agents.skills.local.session.skill import Message
        
        msg = Message(
            role="user",
            content="Hello",
            timestamp="2024-01-15T10:00:00",
        )
        
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert d["timestamp"] == "2024-01-15T10:00:00"


class TestSession:
    """Tests for Session dataclass"""
    
    def test_session_to_dict(self):
        """Test Session to_dict conversion"""
        from webagents.agents.skills.local.session.skill import Session, Message
        
        session = Session(
            session_id="test-123",
            agent_name="test-agent",
            created_at="2024-01-15T10:00:00",
            updated_at="2024-01-15T11:00:00",
            messages=[Message(role="user", content="Hi")],
        )
        
        d = session.to_dict()
        assert d["session_id"] == "test-123"
        assert len(d["messages"]) == 1
    
    def test_session_from_dict(self):
        """Test Session from_dict construction"""
        from webagents.agents.skills.local.session.skill import Session
        
        data = {
            "session_id": "test-456",
            "agent_name": "my-agent",
            "created_at": "2024-01-15T10:00:00",
            "updated_at": "2024-01-15T11:00:00",
            "messages": [{"role": "assistant", "content": "Hello"}],
            "input_tokens": 100,
            "output_tokens": 50,
        }
        
        session = Session.from_dict(data)
        assert session.session_id == "test-456"
        assert len(session.messages) == 1
        assert session.input_tokens == 100