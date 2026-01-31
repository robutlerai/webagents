"""
Local RAG Skill

Uses ChromaDB and local sentence-transformers for embedding and retrieval.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

from ...base import Skill
from webagents.agents.tools.decorators import tool

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

class LocalRagSkill(Skill):
    """Local RAG capabilities using ChromaDB"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        if not RAG_AVAILABLE:
            print("Warning: chromadb or sentence-transformers not installed. RAG skill disabled.")
            return
            
        self.agent_name = config.get("agent_name", "unknown")
        # Base directory for agent data: ~/.webagents/agents/{agent_name}/rag
        self.base_dir = Path.home() / ".webagents" / "agents" / self.agent_name / "rag"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model (using a small, fast model)
        self.model_name = config.get("model", "all-MiniLM-L6-v2")
        self.embedding_function = None # Lazy load
        
        # ChromaDB setup
        self.client = chromadb.PersistentClient(path=str(self.base_dir))
        self.collection = self.client.get_or_create_collection(
            name=f"{self.agent_name}_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
    def _get_embedding_function(self):
        if not self.embedding_function:
            # Using sentence-transformers directly for better control
            model = SentenceTransformer(self.model_name)
            
            class LocalEmbeddingFunction:
                def __init__(self, model):
                    self.model = model
                def __call__(self, input: List[str]) -> List[List[float]]:
                    return self.model.encode(input).tolist()
            
            self.embedding_function = LocalEmbeddingFunction(model)
        return self.embedding_function

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks with overlap"""
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    @tool
    async def index_file(self, file_path: str) -> str:
        """Index a text file for RAG retrieval.
        
        Args:
            file_path: Path to the file to index.
            
        Returns:
            Status message.
        """
        if not RAG_AVAILABLE:
            return "Error: RAG dependencies not installed."
            
        # If relative path, try to resolve against agent's base directory (if available)
        # or current working directory as fallback
        
        # Check if we have an agent-specific base dir (e.g. from FilesystemSkill)
        # This skill doesn't know about FilesystemSkill directly, but we can try to be smart.
        
        # Try to resolve directly
        path = Path(file_path).expanduser().resolve()
        
        if not path.exists():
            # Try relative to cwd if resolved path doesn't exist (e.g. if file_path was just 'file.txt')
            # This is already covered by resolve() on a relative path, but let's double check logic
            # If the user is in /a/b/c and says 'file.txt', resolve() gives /a/b/c/file.txt
            
            # If still not found, return error
            return f"Error: File {file_path} not found at {path}."
            
        try:
            content = path.read_text(encoding='utf-8')
            chunks = self._chunk_text(content)
            
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            ef = self._get_embedding_function()
            embedded_chunks = ef(chunks)
            
            file_hash = hashlib.md5(path.name.encode()).hexdigest()
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embedded_chunks)):
                chunk_id = f"{file_hash}_{i}"
                ids.append(chunk_id)
                embeddings.append(embedding)
                documents.append(chunk)
                
                # Get file metadata
                stat = path.stat()
                import datetime
                
                metadatas.append({
                    "source": str(path),
                    "chunk_index": i,
                    "filename": path.name,
                    "created_at": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "modified_timestamp": int(stat.st_mtime), # Integer for range filtering
                    "size": stat.st_size,
                    "extension": path.suffix.lower()
                })
            
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            return f"Successfully indexed {len(chunks)} chunks from {file_path}."
            
        except Exception as e:
            return f"Error indexing file: {e}"

    @tool
    async def search_knowledge(self, query: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None) -> str:
        """Search the indexed knowledge base.
        
        Args:
            query: The search query.
            n_results: Number of results to return.
            where: Optional metadata filter (e.g., {"extension": ".md"}).
            
        Returns:
            Relevant chunks from indexed files.
        """
        if not RAG_AVAILABLE:
            return "Error: RAG dependencies not installed."
            
        try:
            ef = self._get_embedding_function()
            query_embedding = ef([query])
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where
            )
            
            if not results['documents'][0]:
                return "No relevant information found in knowledge base."
            
            output = []
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                source = meta.get('filename', 'unknown')
                mod_at = meta.get('modified_at', 'unknown')
                output.append(f"--- From {source} (Modified: {mod_at}) ---\n{doc}\n")
                
            return "\n".join(output)
            
        except Exception as e:
            return f"Error searching knowledge base: {e}"

    @tool
    async def clear_knowledge(self) -> str:
        """Clear all indexed knowledge for this agent."""
        if not RAG_AVAILABLE:
            return "Error: RAG dependencies not installed."
            
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=f"{self.agent_name}_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            return "Knowledge base cleared."
        except Exception as e:
            return f"Error clearing knowledge: {e}"
