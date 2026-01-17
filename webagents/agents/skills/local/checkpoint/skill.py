"""
Checkpoint Skill

Git-based file checkpointing using @command decorators.
Commands are exposed as CLI slash commands and HTTP endpoints.
All checkpoint commands require 'owner' scope by default.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import subprocess
import shutil
from datetime import datetime
import uuid
import json
from dataclasses import dataclass, field, asdict

from ...base import Skill
from webagents.agents.tools.decorators import command


@dataclass
class Checkpoint:
    """A checkpoint snapshot."""
    checkpoint_id: str
    created_at: str
    description: str
    commit_hash: Optional[str] = None
    files_changed: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Manages file checkpoints using Git.
    
    Checkpoints are stored in: <agent_path>/.webagents/history/ (Git repo)
    Metadata stored in: <agent_path>/.webagents/checkpoints/
    
    Supports:
    - Creating snapshots before file modifications
    - Restoring to previous checkpoints
    - Listing checkpoint history
    """
    
    def __init__(self, agent_path: Path, agent_name: str):
        """Initialize checkpoint manager.
        
        Args:
            agent_path: Path to agent directory
            agent_name: Name of the agent
        """
        self.agent_path = Path(agent_path).resolve()
        self.agent_name = agent_name
        self.history_dir = self.agent_path / ".webagents" / "history"
        self.checkpoints_dir = self.agent_path / ".webagents" / "checkpoints"
        
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize git repo if needed
        self._ensure_git_repo()
    
    def _ensure_git_repo(self):
        """Ensure git repo is initialized in history directory."""
        git_dir = self.history_dir / ".git"
        if not git_dir.exists():
            self._run_git(["init"], cwd=self.history_dir)
            # Configure git user identity for this repo if not set globally
            self._run_git(["config", "user.email", "agent@webagents.local"], cwd=self.history_dir)
            self._run_git(["config", "user.name", "WebAgents Bot"], cwd=self.history_dir)
            
            # Create initial .gitignore
            gitignore = self.history_dir / ".gitignore"
            gitignore.write_text("# Ignore nothing - track all files\n")
            self._run_git(["add", ".gitignore"], cwd=self.history_dir)
            self._run_git(["commit", "-m", "Initialize checkpoint repository"], cwd=self.history_dir)
    
    def _run_git(self, args: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
        """Run a git command."""
        return subprocess.run(
            ["git"] + args,
            cwd=cwd or self.history_dir,
            capture_output=True,
            text=True
        )
    
    def create(self, description: str = "", files: List[Path] = None, session_id: str = None) -> Checkpoint:
        """Create a new checkpoint.
        
        Args:
            description: Checkpoint description
            files: Specific files to snapshot (None = all tracked files in agent dir)
            session_id: Associated session ID
            
        Returns:
            Created checkpoint
        """
        checkpoint_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        
        # Copy files to history directory
        files_changed = []
        
        if files:
            # Snapshot specific files
            for filepath in files:
                src = Path(filepath)
                if src.exists() and src.is_file():
                    # Maintain relative path structure
                    try:
                        rel_path = src.relative_to(self.agent_path)
                    except ValueError:
                        rel_path = src.name
                    
                    dest = self.history_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                    files_changed.append(str(rel_path))
        else:
            # Snapshot all files in agent directory (excluding .webagents)
            for filepath in self.agent_path.rglob("*"):
                if filepath.is_file() and ".webagents" not in filepath.parts:
                    try:
                        rel_path = filepath.relative_to(self.agent_path)
                        dest = self.history_dir / rel_path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(filepath, dest)
                        files_changed.append(str(rel_path))
                    except (ValueError, PermissionError):
                        continue
        
        # Stage and commit
        self._run_git(["add", "-A"], cwd=self.history_dir)
        
        commit_msg = description or f"Checkpoint {checkpoint_id}"
        result = self._run_git(["commit", "-m", commit_msg, "--allow-empty"], cwd=self.history_dir)
        
        # Get commit hash
        hash_result = self._run_git(["rev-parse", "HEAD"], cwd=self.history_dir)
        commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else None
        
        # Create checkpoint metadata
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            created_at=datetime.now().isoformat(),
            description=description,
            commit_hash=commit_hash,
            files_changed=files_changed,
            session_id=session_id,
        )
        
        # Save metadata
        meta_file = self.checkpoints_dir / f"{checkpoint_id}.json"
        with open(meta_file, "w") as f:
            json.dump(asdict(checkpoint), f, indent=2)
        
        return checkpoint
    
    def restore(self, checkpoint_id: str) -> bool:
        """Restore files from a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to restore
            
        Returns:
            True if restored, False if checkpoint not found
        """
        # Load checkpoint metadata
        meta_file = self.checkpoints_dir / f"{checkpoint_id}.json"
        if not meta_file.exists():
            return False
        
        with open(meta_file, "r") as f:
            data = json.load(f)
        
        commit_hash = data.get("commit_hash")
        if not commit_hash:
            return False
        
        # Checkout the commit
        self._run_git(["checkout", commit_hash], cwd=self.history_dir)
        
        # Copy files back to agent directory
        for filepath in self.history_dir.rglob("*"):
            if filepath.is_file() and ".git" not in filepath.parts:
                try:
                    rel_path = filepath.relative_to(self.history_dir)
                    dest = self.agent_path / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(filepath, dest)
                except (ValueError, PermissionError):
                    continue
        
        # Return to HEAD
        self._run_git(["checkout", "-"], cwd=self.history_dir)
        
        return True
    
    def list_checkpoints(self, limit: int = 20) -> List[Checkpoint]:
        """List recent checkpoints.
        
        Args:
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoints (newest first)
        """
        checkpoints = []
        
        for meta_file in sorted(self.checkpoints_dir.glob("*.json"), reverse=True):
            if len(checkpoints) >= limit:
                break
            
            try:
                with open(meta_file, "r") as f:
                    data = json.load(f)
                checkpoints.append(Checkpoint(**data))
            except Exception:
                continue
        
        return checkpoints
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get checkpoint by ID.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Checkpoint if found, None otherwise
        """
        meta_file = self.checkpoints_dir / f"{checkpoint_id}.json"
        if not meta_file.exists():
            return None
        
        with open(meta_file, "r") as f:
            data = json.load(f)
        
        return Checkpoint(**data)
    
    def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.
        
        Note: This only removes metadata, not git history.
        
        Args:
            checkpoint_id: Checkpoint to delete
            
        Returns:
            True if deleted, False if not found
        """
        meta_file = self.checkpoints_dir / f"{checkpoint_id}.json"
        if meta_file.exists():
            meta_file.unlink()
            return True
        return False


class CheckpointSkill(Skill):
    """Git-based file checkpointing.
    
    Provides commands for:
    - /checkpoint/create (alias: /checkpoint) - Create a new checkpoint
    - /checkpoint/restore - Restore to a previous checkpoint
    - /checkpoint/list - List all checkpoints
    
    Checkpoints are stored in: <agent_path>/.webagents/history/ (Git repo)
    Metadata stored in: <agent_path>/.webagents/checkpoints/
    
    All commands require 'owner' scope by default.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        config = config or {}
        self.agent_name = config.get("agent_name", "default")
        self.agent_path = Path(config.get("agent_path", Path.cwd()))
        
        self.checkpoint_manager = CheckpointManager(self.agent_path, self.agent_name)
    
    @command("/checkpoint/create", alias="/checkpoint", description="Create a new checkpoint", scope="owner")
    async def create_checkpoint(self, description: str = "", files: str = None) -> Dict[str, Any]:
        """Create a new checkpoint snapshot.
        
        Args:
            description: Description of the checkpoint
            files: Comma-separated list of files to snapshot (all files if not specified)
            
        Returns:
            Checkpoint info
        """
        file_list = None
        if files:
            file_list = [Path(f.strip()) for f in files.split(",")]
        
        checkpoint = self.checkpoint_manager.create(
            description=description,
            files=file_list,
        )
        
        return {
            "status": "created",
            "checkpoint_id": checkpoint.checkpoint_id,
            "description": checkpoint.description,
            "commit_hash": checkpoint.commit_hash,
            "files_changed": len(checkpoint.files_changed),
            "created_at": checkpoint.created_at,
        }
    
    @command("/checkpoint/restore", description="Restore to a previous checkpoint", scope="owner")
    async def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Restore files from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            
        Returns:
            Restore confirmation or error
        """
        if not checkpoint_id:
            return {"error": "checkpoint_id is required"}
        
        # Get checkpoint info before restore
        checkpoint = self.checkpoint_manager.get_checkpoint(checkpoint_id)
        if not checkpoint:
            return {"error": f"Checkpoint not found: {checkpoint_id}"}
        
        success = self.checkpoint_manager.restore(checkpoint_id)
        
        if success:
            return {
                "status": "restored",
                "checkpoint_id": checkpoint_id,
                "description": checkpoint.description,
                "created_at": checkpoint.created_at,
                "message": f"Files restored from checkpoint {checkpoint_id}",
            }
        else:
            return {"error": f"Failed to restore checkpoint: {checkpoint_id}"}
    
    @command("/checkpoint/list", description="List all checkpoints", scope="owner")
    async def list_checkpoints(self, limit: int = 20) -> Dict[str, Any]:
        """List recent checkpoints.
        
        Args:
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoint metadata
        """
        checkpoints = self.checkpoint_manager.list_checkpoints(limit=limit)
        
        return {
            "checkpoints": [
                {
                    "checkpoint_id": cp.checkpoint_id,
                    "description": cp.description,
                    "created_at": cp.created_at,
                    "files_changed": len(cp.files_changed),
                }
                for cp in checkpoints
            ],
            "total": len(checkpoints),
        }
    
    @command("/checkpoint/info", description="Get checkpoint details", scope="owner")
    async def get_checkpoint_info(self, checkpoint_id: str) -> Dict[str, Any]:
        """Get detailed information about a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint
            
        Returns:
            Checkpoint details or error
        """
        if not checkpoint_id:
            return {"error": "checkpoint_id is required"}
        
        checkpoint = self.checkpoint_manager.get_checkpoint(checkpoint_id)
        
        if not checkpoint:
            return {"error": f"Checkpoint not found: {checkpoint_id}"}
        
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "description": checkpoint.description,
            "created_at": checkpoint.created_at,
            "commit_hash": checkpoint.commit_hash,
            "files_changed": checkpoint.files_changed,
            "session_id": checkpoint.session_id,
            "metadata": checkpoint.metadata,
        }
    
    @command("/checkpoint/delete", description="Delete a checkpoint", scope="owner")
    async def delete_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Delete a checkpoint.
        
        Note: This only removes metadata, not git history.
        
        Args:
            checkpoint_id: ID of checkpoint to delete
            
        Returns:
            Deletion confirmation or error
        """
        if not checkpoint_id:
            return {"error": "checkpoint_id is required"}
        
        success = self.checkpoint_manager.delete(checkpoint_id)
        
        if success:
            return {
                "status": "deleted",
                "checkpoint_id": checkpoint_id,
                "message": f"Checkpoint {checkpoint_id} deleted",
            }
        else:
            return {"error": f"Checkpoint not found: {checkpoint_id}"}
