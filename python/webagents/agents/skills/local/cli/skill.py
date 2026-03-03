import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool

class CLISkill(Skill):
    """
    CLI Skill - Feature-packed local agent operations.
    
    Provides tools for:
    - Session management (checkpoints, history)
    - Agent configuration (mode, sandbox)
    - System operations
    """
    
    def __init__(self, session=None, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self.session = session
        # Use agent-specific directory for checkpoints if session is available
        self.checkpoints_dir = Path.home() / ".webagents" / "checkpoints"
        if session and session.agent_name:
             self.checkpoints_dir = Path.home() / ".webagents" / "agents" / session.agent_name / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    @tool(scope="all")
    async def save_checkpoint(self, name: str) -> str:
        """
        Save the current conversation state to a checkpoint.
        
        Args:
            name: Name for the checkpoint
            
        Returns:
            Status message
        """
        if not self.session:
            return "Error: No active session linked to CLI skill."
            
        try:
            checkpoint_data = {
                "timestamp": time.time(),
                "created_at": datetime.now().isoformat(),
                "messages": self.session.messages,
                "input_tokens": self.session.input_tokens,
                "output_tokens": self.session.output_tokens,
                "agent_name": self.session.agent_name,
                "model": self.session.model
            }
            
            # Save to agent-specific directory
            file_path = self.checkpoints_dir / f"{name}.json"
            file_path.write_text(json.dumps(checkpoint_data, indent=2))
            
            return f"Checkpoint '{name}' saved successfully to {file_path}."
            
        except Exception as e:
            return f"Failed to save checkpoint: {str(e)}"

    @tool(scope="all")
    async def load_checkpoint(self, name: str) -> str:
        """
        Restore conversation state from a checkpoint.
        WARNING: This will overwrite current conversation history.
        
        Args:
            name: Name of the checkpoint to load
            
        Returns:
            Status message
        """
        if not self.session:
            return "Error: No active session linked to CLI skill."
            
        try:
            # Look in agent-specific dir first
            file_path = self.checkpoints_dir / f"{name}.json"
            
            # Fallback to global dir if not found (legacy)
            if not file_path.exists():
                global_dir = Path.home() / ".webagents" / "checkpoints" / self.session.agent_name
                file_path = global_dir / f"{name}.json"
            
            if not file_path.exists():
                return f"Error: Checkpoint '{name}' not found."
                
            data = json.loads(file_path.read_text())
            
            # Restore state
            messages = data.get("messages", [])
            
            # Context management (truncate history if needed)
            # Default to 20k tokens context roughly (approx 50-100 messages depending on length)
            # Simple heuristic: keep last 100 messages
            if len(messages) > 100:
                # Keep system prompt if present
                if messages and messages[0].get("role") == "system":
                    messages = [messages[0]] + messages[-99:]
                else:
                    messages = messages[-100:]
            
            self.session.messages = messages
            self.session.input_tokens = data.get("input_tokens", 0)
            self.session.output_tokens = data.get("output_tokens", 0)
            
            return f"Checkpoint '{name}' loaded. History restored ({len(self.session.messages)} messages)."
            
        except Exception as e:
            return f"Failed to load checkpoint: {str(e)}"

    @tool(scope="all")
    async def list_checkpoints(self) -> str:
        """
        List all available checkpoints for the current agent.
        
        Returns:
            List of checkpoints with timestamps
        """
        if not self.session:
            return "Error: No active session linked to CLI skill."
            
        try:
            if not self.checkpoints_dir.exists():
                return "No checkpoints found."
                
            checkpoints = []
            for f in self.checkpoints_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    created = data.get("created_at", "Unknown")
                    msg_count = len(data.get("messages", []))
                    checkpoints.append(f"- {f.stem} ({created}, {msg_count} messages)")
                except:
                    checkpoints.append(f"- {f.stem} (corrupt)")
            
            if not checkpoints:
                return "No checkpoints found."
                
            return "Available checkpoints:\n" + "\n".join(sorted(checkpoints))
            
        except Exception as e:
            return f"Failed to list checkpoints: {str(e)}"

    @tool(scope="all")
    async def get_history(self, limit: int = 50) -> str:
        """
        Get recent conversation history.
        
        Args:
            limit: Number of messages to return
            
        Returns:
            Formatted history string
        """
        if not self.session:
            return "Error: No active session."
            
        messages = self.session.messages[-limit:]
        result = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                result.append(f"[{role.upper()}]: {content[:200]}..." if len(content) > 200 else f"[{role.upper()}]: {content}")
        
        return "\n\n".join(result)

    @tool(scope="all")
    async def clear_history(self) -> str:
        """
        Clear the current conversation history.
        
        Returns:
            Status message
        """
        if not self.session:
            return "Error: No active session."
            
        self.session.clear_history()
        return "Conversation history cleared."

    @tool(scope="all")
    async def get_status(self) -> str:
        """
        Get current session status.
        
        Returns:
            Status summary
        """
        if not self.session:
            return "Error: No active session."
            
        return (
            f"Agent: {self.session.agent_name}\n"
            f"Model: {self.session.model}\n"
            f"Messages: {len(self.session.messages)}\n"
            f"Tokens: Input={self.session.input_tokens}, Output={self.session.output_tokens}\n"
            f"Checkpoints Dir: {self.checkpoints_dir}"
        )
