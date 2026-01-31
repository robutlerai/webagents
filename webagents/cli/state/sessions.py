"""
Session Management

Manage REPL sessions and checkpoints.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel


class Message(BaseModel):
    """A conversation message."""
    role: str  # user, assistant, system, tool
    content: str
    timestamp: datetime = None
    tokens: int = 0
    tool_calls: List[Dict] = []
    
    def __init__(self, **data):
        if 'timestamp' not in data or data['timestamp'] is None:
            data['timestamp'] = datetime.utcnow()
        super().__init__(**data)


class Session(BaseModel):
    """A REPL session."""
    
    # Identity
    id: str
    agent_name: str
    
    # Messages
    messages: List[Message] = []
    
    # Metadata
    created_at: datetime = None
    updated_at: datetime = None
    
    # Token stats
    input_tokens: int = 0
    output_tokens: int = 0
    
    # State
    metadata: Dict[str, Any] = {}
    
    def __init__(self, **data):
        if 'created_at' not in data or data['created_at'] is None:
            data['created_at'] = datetime.utcnow()
        if 'updated_at' not in data or data['updated_at'] is None:
            data['updated_at'] = datetime.utcnow()
        super().__init__(**data)
    
    def add_message(self, role: str, content: str, tokens: int = 0, **kwargs) -> Message:
        """Add a message to the session."""
        msg = Message(role=role, content=content, tokens=tokens, **kwargs)
        self.messages.append(msg)
        self.updated_at = datetime.utcnow()
        
        # Update token counts
        if role == "user":
            self.input_tokens += tokens
        elif role == "assistant":
            self.output_tokens += tokens
        
        return msg
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages, optionally limited."""
        if limit:
            return self.messages[-limit:]
        return self.messages
    
    def clear(self):
        """Clear all messages."""
        self.messages = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.updated_at = datetime.utcnow()


class SessionManager:
    """Manage sessions and checkpoints."""
    
    def __init__(self, sessions_dir: Optional[Path] = None):
        """Initialize session manager.
        
        Args:
            sessions_dir: Directory for session files
        """
        if sessions_dir:
            self.sessions_dir = Path(sessions_dir)
        else:
            from .local import get_state
            self.sessions_dir = get_state().get_sessions_dir()
        
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[Session] = None
    
    def create(self, agent_name: str, session_id: Optional[str] = None) -> Session:
        """Create a new session.
        
        Args:
            agent_name: Name of the agent
            session_id: Optional custom session ID
            
        Returns:
            New Session
        """
        if session_id is None:
            session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        session = Session(id=session_id, agent_name=agent_name)
        self.current_session = session
        return session
    
    def save(self, session: Optional[Session] = None, name: Optional[str] = None) -> Path:
        """Save session to file.
        
        Args:
            session: Session to save (defaults to current)
            name: Checkpoint name (defaults to session ID)
            
        Returns:
            Path to saved file
        """
        session = session or self.current_session
        if not session:
            raise ValueError("No session to save")
        
        name = name or session.id
        session_file = self.sessions_dir / f"{name}.json"
        
        session.updated_at = datetime.utcnow()
        session_file.write_text(session.model_dump_json(indent=2))
        
        # Update latest symlink
        latest_file = self.sessions_dir / "latest.json"
        latest_file.write_text(session.model_dump_json(indent=2))
        
        return session_file
    
    def load(self, name: str = "latest") -> Optional[Session]:
        """Load session from file.
        
        Args:
            name: Session/checkpoint name
            
        Returns:
            Session or None if not found
        """
        session_file = self.sessions_dir / f"{name}.json"
        
        if not session_file.exists():
            return None
        
        try:
            data = json.loads(session_file.read_text())
            session = Session(**data)
            self.current_session = session
            return session
        except Exception:
            return None
    
    def list_sessions(self) -> List[str]:
        """List available sessions/checkpoints."""
        sessions = []
        for f in self.sessions_dir.glob("*.json"):
            if f.name != "latest.json":
                sessions.append(f.stem)
        return sorted(sessions, reverse=True)
    
    def delete(self, name: str) -> bool:
        """Delete a session.
        
        Args:
            name: Session name
            
        Returns:
            True if deleted
        """
        session_file = self.sessions_dir / f"{name}.json"
        if session_file.exists():
            session_file.unlink()
            return True
        return False
    
    def get_current(self) -> Optional[Session]:
        """Get current session."""
        return self.current_session
    
    def set_current(self, session: Session):
        """Set current session."""
        self.current_session = session
