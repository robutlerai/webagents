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
from webagents.agents.tools.decorators import command, hook, http
from typing import Any


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
    # Messages stored in OpenAI format (raw dicts) for compatibility
    messages: List[Any] = field(default_factory=list)
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
        # Keep messages as raw dicts to preserve full OpenAI format
        # (tool_calls, tool_call_id, etc.)
        messages = data.get("messages", [])
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
            def _msg_role(m):
                return m.get('role') if isinstance(m, dict) else getattr(m, 'role', None)
            system_msgs = [m for m in messages if _msg_role(m) == "system"]
            other_msgs = [m for m in messages if _msg_role(m) != "system"]
            
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
    
    Uses hooks for automatic message logging:
    - on_connection: Initialize/load session
    - after_llm_call: Log assistant messages with tool calls
    - after_toolcall: Log tool results
    - finalize_connection: Auto-save session
    
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
        self.auto_save = config.get("auto_save", True)  # Auto-save after each message
        
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
    
    def add_message_dict(self, msg: Dict[str, Any]) -> None:
        """Add a raw message dict to current session (OpenAI format).
        
        This preserves the exact message structure from context.messages.
        """
        session = self.get_current_session()
        # Add timestamp if not present
        if 'timestamp' not in msg:
            msg = {**msg, 'timestamp': datetime.now().isoformat()}
        session.messages.append(msg)
        session.updated_at = datetime.now().isoformat()
    
    def update_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Update token counts for current session."""
        session = self.get_current_session()
        session.input_tokens += input_tokens
        session.output_tokens += output_tokens
    
    def _get_session_completions(self) -> Dict[str, List[str]]:
        """Return session IDs for autocomplete."""
        sessions = self.session_manager.list_sessions()[:20]
        return {"session_id": [s.get("session_id", "") for s in sessions if s.get("session_id")]}
    
    def _get_subcommand_completions(self) -> Dict[str, List[str]]:
        """Return subcommands for autocomplete."""
        return {"subcommand": ["save", "load", "new", "history", "clear"]}
    
    # =========================================================================
    # Hooks for automatic message logging
    # =========================================================================
    
    @hook("on_connection", priority=90)
    async def session_on_connection(self, context) -> Any:
        """Initialize session on connection start."""
        # Load or create session
        self._current_session = self.session_manager.load_latest()
        if not self._current_session:
            self._current_session = Session(
                session_id=str(uuid.uuid4()),
                agent_name=self.agent_name,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )
        
        # Log incoming user message from context
        # Context has .messages attribute directly, not in custom_data
        messages = getattr(context, 'messages', None)
        if messages and len(messages) > 0:
            last_msg = messages[-1]
            if isinstance(last_msg, dict) and last_msg.get('role') == 'user':
                self.add_message(role='user', content=last_msg.get('content', ''))
        
        return context
    
    @hook("on_message", priority=90)
    async def session_on_message(self, context) -> Any:
        """Store messages from context in OpenAI format.
        
        This stores the raw messages as they appear in context.messages:
        - assistant messages (with tool_calls array if present)
        - tool messages (with tool_call_id and content/result)
        - final assistant message
        """
        messages = getattr(context, 'messages', None) or []
        
        if not messages:
            return context
        
        # Find where we left off (after the user message we already logged)
        session = self.get_current_session()
        stored_count = len(session.messages)
        
        # Find the last user message index in context.messages
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get('role') == 'user':
                last_user_idx = i
        
        # Store all messages after the user message (assistant, tool, etc.)
        # These are in proper OpenAI format
        for msg in messages[last_user_idx + 1:]:
            if not isinstance(msg, dict):
                continue
            
            role = msg.get('role', '')
            
            # Skip system messages (prompts added by agent)
            if role == 'system':
                continue
            
            # Store the message as-is (OpenAI format)
            self.add_message_dict(msg)
        
        return context
    
    @hook("finalize_connection", priority=90)
    async def session_finalize_connection(self, context) -> Any:
        """Auto-save session at end of connection."""
        if self.auto_save and self._current_session:
            self._current_session.updated_at = datetime.now().isoformat()
            self.session_manager.save(self._current_session)
        return context
    
    # =========================================================================
    # HTTP Endpoints for frontend integration
    # =========================================================================
    
    @http("/sessions", method="get")
    async def list_sessions_http(self, limit: int = 20) -> Dict[str, Any]:
        """List all sessions for this agent.
        
        GET /agents/{name}/sessions
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session metadata
        """
        sessions = self.session_manager.list_sessions()[:limit]
        return {
            "sessions": sessions,
            "count": len(sessions),
            "agent": self.agent_name,
        }
    
    @http("/sessions/current", method="get")
    async def get_current_session_http(self) -> Dict[str, Any]:
        """Get current session with full message history.
        
        GET /agents/{name}/sessions/current
        
        This is the primary endpoint for frontends to load conversation state.
        
        Returns:
            Current session with messages array (OpenAI-compatible format)
        """
        session = self.get_current_session()
        # Handle both Message objects and raw dicts
        messages = [
            m.to_dict() if hasattr(m, 'to_dict') else m 
            for m in session.messages
        ]
        return {
            "session_id": session.session_id,
            "agent": session.agent_name,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "messages": messages,
            "metadata": session.metadata,
            "usage": {
                "input_tokens": session.input_tokens,
                "output_tokens": session.output_tokens,
            }
        }
    
    @http("/sessions/{session_id}", method="get")
    async def get_session_by_id_http(self, session_id: str) -> Dict[str, Any]:
        """Get a specific session by ID.
        
        GET /agents/{name}/sessions/{session_id}
        
        Args:
            session_id: The session ID to load
            
        Returns:
            Session with messages array
        """
        session = self.session_manager.load(session_id)
        if not session:
            return {"error": "Session not found", "session_id": session_id}
        
        # Handle both Message objects and raw dicts
        messages = [
            m.to_dict() if hasattr(m, 'to_dict') else m 
            for m in session.messages
        ]
        return {
            "session_id": session.session_id,
            "agent": session.agent_name,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "messages": messages,
            "metadata": session.metadata,
            "usage": {
                "input_tokens": session.input_tokens,
                "output_tokens": session.output_tokens,
            }
        }
    
    @http("/sessions", method="post")
    async def create_new_session_http(self) -> Dict[str, Any]:
        """Create a new empty session and set it as current.
        
        POST /agents/{name}/sessions
        
        Returns:
            New session info
        """
        self._current_session = Session(
            session_id=str(uuid.uuid4()),
            agent_name=self.agent_name,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        self.session_manager.save(self._current_session)
        
        return {
            "session_id": self._current_session.session_id,
            "agent": self.agent_name,
            "created_at": self._current_session.created_at,
            "messages": [],
        }
    
    @http("/sessions/{session_id}", method="delete")
    async def delete_session_http(self, session_id: str) -> Dict[str, Any]:
        """Delete a specific session.
        
        DELETE /agents/{name}/sessions/{session_id}
        
        Args:
            session_id: The session ID to delete
            
        Returns:
            Deletion status
        """
        success = self.session_manager.delete(session_id)
        return {
            "deleted": success,
            "session_id": session_id,
        }
    
    # =========================================================================
    # Commands
    # =========================================================================
    
    @command("/session", description="Session commands - save, load, new, history, clear", scope="all",
             completions=lambda self: self._get_subcommand_completions())
    async def session_help(self, subcommand: str = None) -> Dict[str, Any]:
        """Show session help or execute subcommand.
        
        Args:
            subcommand: Optional subcommand to execute (save, load, new, history, clear)
            
        Returns:
            Help info or subcommand result
        """
        subcommands = {
            "save": {"description": "Save current session", "usage": "/session save [name]"},
            "load": {"description": "Load a session by ID", "usage": "/session load [session_id]"},
            "new": {"description": "Start a new session", "usage": "/session new"},
            "history": {"description": "List all sessions", "usage": "/session history [limit]"},
            "clear": {"description": "Clear all sessions", "usage": "/session clear confirm=true"},
        }
        
        if not subcommand:
            # Build help display
            lines = ["[bold]/session[/bold] - Session and conversation history management", ""]
            for name, info in subcommands.items():
                lines.append(f"  [cyan]/session {name}[/cyan] - {info['description']}")
            
            return {
                "command": "/session",
                "description": "Session and conversation history management",
                "subcommands": subcommands,
                "display": "\n".join(lines),
            }
        
        # If subcommand provided, show its help
        if subcommand in subcommands:
            info = subcommands[subcommand]
            return {
                "command": f"/session {subcommand}",
                **info,
                "display": f"[cyan]{info['usage']}[/cyan]\n{info['description']}",
            }
        
        return {
            "error": f"Unknown subcommand: {subcommand}. Available: {', '.join(subcommands.keys())}",
            "display": f"[red]Error:[/red] Unknown subcommand: {subcommand}. Available: {', '.join(subcommands.keys())}",
        }
    
    @command("/session/save", description="Save current session", scope="all")
    async def save_session(self, name: str = None, messages: List[Dict] = None) -> Dict[str, Any]:
        """Save the current session.
        
        Args:
            name: Optional session name (uses session_id if not provided)
            messages: Optional list of messages to replace session (for client-side state sync)
            
        Returns:
            Save confirmation with session info
        """
        session = self.get_current_session()
        
        # If messages provided, replace session messages (client sends full state)
        if messages is not None:
            session.messages = [
                Message(**m) if isinstance(m, dict) else m
                for m in messages
            ]
            session.updated_at = datetime.now().isoformat()
        
        if name:
            session.metadata["name"] = name
        
        filepath = self.session_manager.save(session)
        
        sid = session.session_id[:8]
        display_name = name or sid
        return {
            "status": "saved",
            "session_id": session.session_id,
            "name": name or session.session_id,
            "message_count": len(session.messages),
            "path": str(filepath),
            "display": f"[green]✓ saved[/green] [{sid}] {display_name} ({len(session.messages)} msgs)",
        }
    
    @command("/session/load", description="Load a session by ID", scope="all",
             completions=lambda self: self._get_session_completions())
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
            return {
                "error": "Session not found",
                "session_id": session_id,
                "display": f"[red]Error:[/red] Session not found: {session_id}",
            }
        
        self._current_session = session
        
        sid = session.session_id[:8]
        return {
            "status": "loaded",
            "session_id": session.session_id,
            "created_at": session.created_at,
            "message_count": len(session.messages),
            "messages": [m.to_dict() if hasattr(m, 'to_dict') else m for m in session.messages],
            "input_tokens": session.input_tokens,
            "output_tokens": session.output_tokens,
            "display": f"[green]✓ loaded[/green] [{sid}] {len(session.messages)} msgs · {session.created_at[:16]}",
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
        
        # Save the new session immediately so it appears in history and is set as latest
        self.session_manager.save(self._current_session)
        
        sid = self._current_session.session_id[:8]
        return {
            "status": "created",
            "session_id": self._current_session.session_id,
            "message": "New session started",
            "display": f"[green]✓ created[/green] New session started [{sid}]",
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
        
        # Build display
        lines = ["[bold]Sessions:[/]"]
        for s in sessions[:10]:
            sid = s.get("session_id", "?")[:8]
            cnt = s.get("message_count", 0)
            updated = s.get("updated_at", "?")[:16]
            lines.append(f"  [dim]{sid}[/] {cnt} msgs · {updated}")
        
        return {
            "sessions": sessions,
            "total": len(sessions),
            "current_session_id": self._current_session.session_id if self._current_session else None,
            "display": "\n".join(lines),
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
                "display": "[yellow]⚠ Warning:[/yellow] This will delete all sessions. Set confirm=true to proceed.",
            }
        
        self.session_manager.clear_all()
        self._current_session = None
        
        return {
            "status": "cleared",
            "message": "All sessions have been deleted",
            "display": "[green]✓ cleared[/green] All sessions have been deleted",
        }
