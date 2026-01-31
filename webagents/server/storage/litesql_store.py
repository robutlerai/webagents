"""
LiteSQL Metadata Store

SQLite-based local metadata storage for agents.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Optional, List
import json


class LiteSQLMetadataStore:
    """SQLite-based local metadata storage"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".webagents" / "data" / "agents.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                name TEXT PRIMARY KEY,
                metadata TEXT NOT NULL,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def get_agent(self, name: str) -> Optional[Dict]:
        """Get agent metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT metadata FROM agents WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return None
    
    def list_agents(self) -> List[Dict]:
        """List all agents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT metadata FROM agents")
        rows = cursor.fetchall()
        conn.close()
        
        return [json.loads(row[0]) for row in rows]
    
    def register_agent(self, name: str, metadata: Dict, source: str = "local"):
        """Register agent metadata"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO agents (name, metadata, source, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (name, json.dumps(metadata), source))
        conn.commit()
        conn.close()
    
    def delete_agent(self, name: str):
        """Delete agent metadata"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM agents WHERE name = ?", (name,))
        conn.commit()
        conn.close()
