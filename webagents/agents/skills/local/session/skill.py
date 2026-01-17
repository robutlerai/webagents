"""
Session Manager Skill

Session and conversation history management using @command decorators.
Commands are exposed as CLI slash commands and HTTP endpoints.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import json
from dataclasses import dataclass, field

from ...base import Skill
from webagents.agents.tools.decorators import command


@dataclass
class Message:
    """A single message in a conversation."""
    role: str
    content: str
    timestamp: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.timestamp:
            d["timestamp"] = self.timestamp
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class Session:
    """A conversation session."""
    session_id: str
    agent_name: str
    created_at: str
    updated_at: str
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() if isinstance(m, Message) else m for m in self.messages],
            "metadata": self.metadata,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        messages = [
            Message(**m) if isinstance(m, dict) else m
            for m in data.get("messages", [])
        ]
        return cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            agent_name=data.get("agent_name", "unknown"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            messages=messages,
            metadata=data.get("metadata", {}),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
        )


class SessionManager:
    """Manages conversation sessions with JSON persistence.
    
    Sessions are stored in: <agent_path>/.webagents/sessions/
    
    Supports:
    - H2A (Human-to-Agent) sessions
    - A2A (Agent-to-Agent) sessions with conversation_id
    - Auto-resume of latest session
    """
    
    def __init__(self, agent_path: Path, agent_name: str):
        """Initialize session manager.
        
        Args:
            agent_path: Path to agent directory
            agent_name: Name of the agent
        """
        self.agent_path = Path(agent_path)
        self.agent_name = agent_name
        self.sessions_dir = self.agent_path / ".webagents" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def _session_file(self, session_id: str, conversation_id: Optional[str] = None) -> Path:
        """Get path to session file."""
        if conversation_id:
            # A2A session - store in subdirectory
            a2a_dir = self.sessions_dir / "a2a" / conversation_id
            a2a_dir.mkdir(parents=True, exist_ok=True)
            return a2a_dir / f"{session_id}.json"
        return self.sessions_dir / f"{session_id}.json"
    
    def save(self, session: Session, conversation_id: Optional[str] = None) -> Path:
        """Save session to disk.
        
        Args:
            session: Session to save
            conversation_id: Optional A2A conversation ID
            
        Returns:
            Path to saved session file
        """
        session.updated_at = datetime.now().isoformat()
        filepath = self._session_file(session.session_id, conversation_id)
        
        with open(filepath, "w") as f:
            json.dump(session.to_dict(), f, indent=2)
        
        # Update latest pointer
        self._update_latest(session.session_id, conversation_id)
        
        return filepath
    
    def load(self, session_id: str, conversation_id: Optional[str] = None) -> Optional[Session]:
        """Load session from disk.
        
        Args:
            session_id: Session ID to load
            conversation_id: Optional A2A conversation ID
            
        Returns:
            Session if found, None otherwise
        """
        filepath = self._session_file(session_id, conversation_id)
        
        if not filepath.exists():
            return None
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        return Session.from_dict(data)
    
    def load_latest(self, conversation_id: Optional[str] = None, max_messages: int = 100) -> Optional[Session]:
        """Load the most recent session.
        
        Args:
            conversation_id: Optional A2A conversation ID
            max_messages: Maximum messages to load (for context window management)
            
        Returns:
            Latest session if found, None otherwise
        """
        latest_file = self._get_latest_file(conversation_id)
        
        if not latest_file or not latest_file.exists():
            return None
        
        with open(latest_file, "r") as f:
            session_id = f.read().strip()
        
        session = self.load(session_id, conversation_id)
        
        # Trim messages if needed
        if session and len(session.messages) > max_messages:
            # Keep system prompt if present, then last N messages
            messages = session.messages
            system_msgs = [m for m in messages if m.role == "system"]
            other_msgs = [m for m in messages if m.role != "system"]
            
            # Keep system + last (max_messages - len(system)) messages
            remaining = max_messages - len(system_msgs)
            session.messages = system_msgs + other_msgs[-remaining:]
        
        return session
    
    def _get_latest_file(self, conversation_id: Optional[str] = None) -> Optional[Path]:
        """Get path to latest session pointer file."""
        if conversation_id:
            return self.sessions_dir / "a2a" / conversation_id / ".latest"
        return self.sessions_dir / ".latest"
    
    def _update_latest(self, session_id: str, conversation_id: Optional[str] = None):
        """Update the latest session pointer."""
        latest_file = self._get_latest_file(conversation_id)
        if latest_file:
            latest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(latest_file, "w") as f:
                f.write(session_id)
    
    def list_sessions(self, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all sessions.
        
        Args:
            conversation_id: Optional A2A conversation ID
            
        Returns:
            List of session metadata
        """
        sessions = []
        
        if conversation_id:
            search_dir = self.sessions_dir / "a2a" / conversation_id
        else:
            search_dir = self.sessions_dir
        
        if not search_dir.exists():
            return sessions
        
        for filepath in search_dir.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": len(data.get("messages", [])),
                })
            except Exception:
                continue
        
        # Sort by updated_at descending
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions
    
    def delete(self, session_id: str, conversation_id: Optional[str] = None) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session to delete
            conversation_id: Optional A2A conversation ID
            
        Returns:
            True if deleted, False if not found
        """
        filepath = self._session_file(session_id, conversation_id)
        
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    def clear_all(self, conversation_id: Optional[str] = None):
        """Clear all sessions.
        
        Args:
            conversation_id: Optional A2A conversation ID (clears only that conversation)
        """
        if conversation_id:
            search_dir = self.sessions_dir / "a2a" / conversation_id
        else:
            search_dir = self.sessions_dir
        
        if search_dir.exists():
            for filepath in search_dir.glob("*.json"):
                filepath.unlink()
            
            latest = search_dir / ".latest"
            if latest.exists():
                latest.unlink()


class SessionManagerSkill(Skill):
    """Session and conversation history management.
    
    Provides commands for:
    - /session/save - Save current session
    - /session/load - Load a session
    - /session/new - Start a new session
    - /session/history - Show session history
    - /session/clear - Clear all sessions
    
    Sessions are stored in: <agent_path>/.webagents/sessions/
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        config = config or {}
        self.agent_name = config.get("agent_name", "default")
        self.agent_path = Path(config.get("agent_path", Path.cwd()))
        
        self.session_manager = SessionManager(self.agent_path, self.agent_name)
        self._current_session: Optional[Session] = None
    
    def get_current_session(self) -> Session:
        """Get or create current session."""
        if not self._current_session:
            # Try to load latest
            self._current_session = self.session_manager.load_latest()
            
            if not self._current_session:
                # Create new session
                self._current_session = Session(
                    session_id=str(uuid.uuid4()),
                    agent_name=self.agent_name,
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                )
        
        return self._current_session
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to current session."""
        session = self.get_current_session()
        msg = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            **kwargs
        )
        session.messages.append(msg)
        session.updated_at = datetime.now().isoformat()
    
    def update_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Update token counts for current session."""
        session = self.get_current_session()
        session.input_tokens += input_tokens
        session.output_tokens += output_tokens
    
    @command("/session/save", description="Save current session", scope="all")
    async def save_session(self, name: str = None) -> Dict[str, Any]:
        """Save the current session.
        
        Args:
            name: Optional session name (uses session_id if not provided)
            
        Returns:
            Save confirmation with session info
        """
        session = self.get_current_session()
        
        if name:
            session.metadata["name"] = name
        
        filepath = self.session_manager.save(session)
        
        return {
            "status": "saved",
            "session_id": session.session_id,
            "name": name or session.session_id,
            "message_count": len(session.messages),
            "path": str(filepath),
        }
    
    @command("/session/load", description="Load a session by ID", scope="all")
    async def load_session(self, session_id: str = None) -> Dict[str, Any]:
        """Load a session.
        
        Args:
            session_id: Session ID to load (loads latest if not provided)
            
        Returns:
            Session info or error
        """
        if session_id:
            session = self.session_manager.load(session_id)
        else:
            session = self.session_manager.load_latest()
        
        if not session:
            return {"error": "Session not found", "session_id": session_id}
        
        self._current_session = session
        
        return {
            "status": "loaded",
            "session_id": session.session_id,
            "created_at": session.created_at,
            "message_count": len(session.messages),
            "input_tokens": session.input_tokens,
            "output_tokens": session.output_tokens,
        }
    
    @command("/session/new", alias="/new", description="Start a new session", scope="all")
    async def new_session(self) -> Dict[str, Any]:
        """Start a new session, discarding the current one.
        
        Returns:
            New session info
        """
        # Save current session if it has messages
        if self._current_session and self._current_session.messages:
            self.session_manager.save(self._current_session)
        
        # Create new session
        self._current_session = Session(
            session_id=str(uuid.uuid4()),
            agent_name=self.agent_name,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        
        return {
            "status": "created",
            "session_id": self._current_session.session_id,
            "message": "New session started",
        }
    
    @command("/session/history", description="List all sessions", scope="all")
    async def list_sessions(self, limit: int = 20) -> Dict[str, Any]:
        """List all sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session metadata
        """
        sessions = self.session_manager.list_sessions()[:limit]
        
        return {
            "sessions": sessions,
            "total": len(sessions),
            "current_session_id": self._current_session.session_id if self._current_session else None,
        }
    
    @command("/session/clear", description="Clear all sessions", scope="owner")
    async def clear_sessions(self, confirm: bool = False) -> Dict[str, Any]:
        """Clear all sessions.
        
        Args:
            confirm: Must be True to actually clear
            
        Returns:
            Confirmation or warning
        """
        if not confirm:
            return {
                "warning": "This will delete all sessions. Set confirm=true to proceed.",
                "confirm_required": True,
            }
        
        self.session_manager.clear_all()
        self._current_session = None
        
        return {
            "status": "cleared",
            "message": "All sessions have been deleted",
        }
