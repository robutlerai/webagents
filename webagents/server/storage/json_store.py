"""
JSON Metadata Store

JSON-based local metadata storage for agents.
"""

from pathlib import Path
import json
from typing import Dict, Optional, List


class JSONMetadataStore:
    """JSON-based local metadata storage"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".webagents" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.agents_file = self.data_dir / "agents.json"
        self._load()
    
    def _load(self):
        """Load agents from JSON"""
        if self.agents_file.exists():
            self.agents = json.loads(self.agents_file.read_text())
        else:
            self.agents = {}
    
    def _save(self):
        """Save agents to JSON"""
        self.agents_file.write_text(json.dumps(self.agents, indent=2))
    
    def get_agent(self, name: str) -> Optional[Dict]:
        """Get agent metadata"""
        return self.agents.get(name)
    
    def list_agents(self) -> List[Dict]:
        """List all agents"""
        return list(self.agents.values())
    
    def register_agent(self, name: str, metadata: Dict):
        """Register agent metadata"""
        self.agents[name] = metadata
        self._save()
    
    def delete_agent(self, name: str):
        """Delete agent metadata"""
        if name in self.agents:
            del self.agents[name]
            self._save()
