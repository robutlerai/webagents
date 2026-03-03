"""
Session History Management

Manage conversation history and checkpoints.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel


class Message(BaseModel):
    """A conversation message."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = None
    tokens: int = 0
    
    def __init__(self, **data):
        if 'timestamp' not in data or data['timestamp'] is None:
            data['timestamp'] = datetime.utcnow()
        super().__init__(**data)


class Checkpoint(BaseModel):
    """A saved session checkpoint."""
    name: str
    messages: List[Message]
    created_at: datetime
    agent_name: str
    metadata: Dict = {}


class HistoryManager:
    """Manage conversation history and checkpoints."""
    
    def __init__(self, agent_name: str = "default"):
        self.agent_name = agent_name
        self.messages: List[Message] = []
        
        # Paths
        self.webagents_dir = Path.home() / ".webagents"
        self.sessions_dir = self.webagents_dir / "sessions"
        self.history_dir = self.webagents_dir / "history" / agent_name
        
        # Ensure directories exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
    
    def add_message(self, role: str, content: str, tokens: int = 0):
        """Add a message to history."""
        msg = Message(role=role, content=content, tokens=tokens)
        self.messages.append(msg)
        return msg
    
    def get_messages(self, limit: int = None) -> List[Message]:
        """Get conversation messages."""
        if limit:
            return self.messages[-limit:]
        return self.messages
    
    def clear(self):
        """Clear history."""
        self.messages = []
    
    def search(self, query: str) -> List[Message]:
        """Search history for messages containing query."""
        query_lower = query.lower()
        return [m for m in self.messages if query_lower in m.content.lower()]
    
    def save_checkpoint(self, name: str = None) -> Checkpoint:
        """Save current state as checkpoint."""
        if name is None:
            name = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = Checkpoint(
            name=name,
            messages=self.messages.copy(),
            created_at=datetime.utcnow(),
            agent_name=self.agent_name,
        )
        
        # Save to file
        checkpoint_file = self.sessions_dir / f"{name}.json"
        checkpoint_file.write_text(checkpoint.model_dump_json(indent=2))
        
        # Update latest symlink
        latest_file = self.sessions_dir / "latest.json"
        if latest_file.exists():
            latest_file.unlink()
        latest_file.write_text(checkpoint.model_dump_json(indent=2))
        
        return checkpoint
    
    def load_checkpoint(self, name: str = "latest") -> Optional[Checkpoint]:
        """Load a checkpoint."""
        checkpoint_file = self.sessions_dir / f"{name}.json"
        
        if not checkpoint_file.exists():
            return None
        
        data = json.loads(checkpoint_file.read_text())
        checkpoint = Checkpoint(**data)
        
        # Restore messages
        self.messages = checkpoint.messages
        self.agent_name = checkpoint.agent_name
        
        return checkpoint
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        checkpoints = []
        for f in self.sessions_dir.glob("*.json"):
            if f.name != "latest.json":
                checkpoints.append(f.stem)
        return sorted(checkpoints, reverse=True)
    
    def delete_checkpoint(self, name: str) -> bool:
        """Delete a checkpoint."""
        checkpoint_file = self.sessions_dir / f"{name}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            return True
        return False
    
    def persist_history(self):
        """Save history to daily log file."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        history_file = self.history_dir / f"{today}.jsonl"
        
        with open(history_file, "a") as f:
            for msg in self.messages:
                f.write(msg.model_dump_json() + "\n")
    
    def get_token_stats(self) -> Dict[str, int]:
        """Get token usage statistics."""
        input_tokens = sum(m.tokens for m in self.messages if m.role == "user")
        output_tokens = sum(m.tokens for m in self.messages if m.role == "assistant")
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "message_count": len(self.messages),
        }
