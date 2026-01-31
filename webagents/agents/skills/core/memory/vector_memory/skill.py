"""
VectorMemorySkill - Milvus-backed vector memory for agent instructions/context

Features:
- Store and retrieve instruction documents (common or agent-specific)
- Tool to fetch relevant instructions for a problem (common + agent-specific)
- Owner-only tools to upload/remove documents

Env configuration:
- MILVUS_HOST (e.g., https://in03)
- MILVUS_PORT (e.g., 443)
- MILVUS_TOKEN (if required by Milvus/Cosmos)
- MILVUS_COLLECTION (default: webagents_memory)
- MILVUS_FORCE_RECREATE (truthy to recreate collection on init)
- EMBEDDING_MODEL (default: text-embedding-3-small)
- LITELLM_BASE_URL (default: http://localhost:2225)
- LITELLM_API_KEY or WEBAGENTS_API_KEY (bearer for embeddings)
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt
from webagents.utils.logging import get_logger, log_skill_event, log_tool_execution

try:
    from pymilvus import (
        connections,
        FieldSchema, CollectionSchema, DataType, Collection,
        utility as milvus_utility,
    )
    _MILVUS_AVAILABLE = True
except Exception:
    _MILVUS_AVAILABLE = False


def _is_truthy(val: Optional[str]) -> bool:
    return str(val).lower() in {"1", "true", "yes", "y", "on"}


def _get_env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    return v if v and str(v).strip() != "" else default


class VectorMemorySkill(Skill):
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config or {}, scope="all")
        self.config = config or {}
        self.logger = None
        self.collection: Optional[Collection] = None
        self.dim: int = int(os.getenv("EMBEDDING_DIM", "1536"))
        # Milvus config
        self.milvus_host = _get_env_str("MILVUS_HOST", "http://localhost")
        self.milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
        self.milvus_token = _get_env_str("MILVUS_TOKEN")
        self.milvus_collection = _get_env_str("MILVUS_COLLECTION", "webagents_memory")
        self.milvus_force_recreate = _is_truthy(os.getenv("MILVUS_FORCE_RECREATE", "false"))
        # Embeddings config
        self.embed_model = _get_env_str("EMBEDDING_MODEL", "text-embedding-3-small")
        self.litellm_base = _get_env_str("LITELLM_BASE_URL", "http://localhost:2225")
        self.litellm_key = _get_env_str("LITELLM_API_KEY") or _get_env_str("WEBAGENTS_API_KEY")

    async def initialize(self, agent) -> None:
        self.agent = agent
        self.logger = get_logger('skill.webagents.vector_memory', agent.name)
        if not _MILVUS_AVAILABLE:
            self.logger.warning("pymilvus not available; VectorMemorySkill disabled")
            return
        try:
            connections.connect(
                alias="default",
                uri=self.milvus_host if self.milvus_host.startswith("http") else None,
                host=None if self.milvus_host.startswith("http") else self.milvus_host,
                port=str(self.milvus_port),
                token=self.milvus_token,
            )
        except Exception as e:
            self.logger.warning(f"Failed to connect to Milvus: {e}")
            return

        try:
            if self.milvus_force_recreate and milvus_utility.has_collection(self.milvus_collection):
                milvus_utility.drop_collection(self.milvus_collection)

            if not milvus_utility.has_collection(self.milvus_collection):
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                    FieldSchema(name="agent_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="owner_user_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                ]
                schema = CollectionSchema(fields=fields, description="WebAgents Vector Memory")
                self.collection = Collection(name=self.milvus_collection, schema=schema)
                try:
                    self.collection.create_index(
                        field_name="vector",
                        index_params={
                            "index_type": "IVF_FLAT",
                            "metric_type": "IP",
                            "params": {"nlist": 1024},
                        },
                    )
                except Exception:
                    pass
                self.collection.load()
            else:
                self.collection = Collection(self.milvus_collection)
                try:
                    self.collection.load()
                except Exception:
                    pass
            self.logger.info(f"VectorMemorySkill initialized; collection={self.milvus_collection}")
            try:
                log_skill_event(self.agent.name, 'vector_memory', 'initialized', {
                    'milvus_host': (self.milvus_host or '')[:40],
                    'milvus_port': self.milvus_port,
                    'collection': self.milvus_collection,
                    'force_recreate': self.milvus_force_recreate,
                    'embed_model': self.embed_model,
                })
            except Exception:
                pass
        except Exception as e:
            self.logger.warning(f"Milvus collection init failed: {e}")
            self.collection = None

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            import httpx
            headers = {"Content-Type": "application/json"}
            if self.litellm_key:
                headers["Authorization"] = f"Bearer {self.litellm_key}"
            url = f"{self.litellm_base.rstrip('/')}/v1/embeddings"
            payload = {"model": self.embed_model, "input": texts}
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json().get("data", [])
                return [item.get("embedding", []) for item in data]
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Embedding failed: {e}")
            return [[0.0] * self.dim for _ in texts]

    def _ensure_collection(self) -> bool:
        return bool(_MILVUS_AVAILABLE and self.collection)

    @tool(description="Retrieve relevant instruction documents (common + agent-specific) for the current problem.")
    async def fetch_instructions_tool(self, problem: str, top_k: int = 3, context: Any = None) -> str:
        if not self._ensure_collection():
            return "❌ Vector memory unavailable"
        try:
            agent_id = getattr(self.agent, 'id', None)
            vectors = await self._embed([problem])
            if not vectors:
                return ""
            qvec = vectors[0]
            expr = "agent_id == ''"
            if agent_id:
                expr = f"({expr}) or agent_id == '{agent_id}'"
            res = self.collection.search(
                data=[qvec],
                anns_field="vector",
                param={"nprobe": 16},
                limit=max(1, int(top_k)),
                output_fields=["id", "title", "content", "agent_id"],
                expr=expr,
            )
            hits = res[0] if res else []
            docs: List[Tuple[str, str]] = []
            for hit in hits:
                row = hit.entity
                title = row.get("title") or "Instruction"
                content = row.get("content") or ""
                if content:
                    docs.append((title, content))
            if not docs:
                try:
                    log_tool_execution(self.agent.name, 'vector_memory.fetch_instructions_tool', 0, success=True)
                except Exception:
                    pass
                return ""
            combined = [f"# {t}\n{c}" for t, c in docs]
            try:
                log_tool_execution(self.agent.name, 'vector_memory.fetch_instructions_tool', 0, success=True)
            except Exception:
                pass
            return "\n\n".join(combined)
        except Exception as e:
            if self.logger:
                self.logger.error(f"fetch_instructions_tool error: {e}")
            try:
                log_tool_execution(self.agent.name, 'vector_memory.fetch_instructions_tool', 0, success=False)
            except Exception:
                pass
            return ""

    # @tool(description="Owner-only: upload an instruction document to the knowledge base (common or agent-specific).", scope="owner")
    async def upload_instruction(self, title: str, content: str, agent_specific: bool = True, context: Any = None) -> str:
        if not self._ensure_collection():
            return "❌ Vector memory unavailable"
        try:
            agent_id = getattr(self.agent, 'id', None) if agent_specific else ""
            owner_user_id = getattr(self.agent, 'owner_user_id', None) or ""
            vecs = await self._embed([f"{title}\n\n{content}"])
            vec = vecs[0] if vecs else [0.0] * self.dim
            doc_id = uuid.uuid4().hex
            entities = [
                [doc_id],
                [agent_id or ""],
                [owner_user_id],
                [title[:255]],
                [content[:8000]],
                [vec],
            ]
            self.collection.insert(entities)
            try:
                self.collection.flush()
            except Exception:
                pass
            try:
                log_tool_execution(self.agent.name, 'vector_memory.upload_instruction', 0, success=True)
            except Exception:
                pass
            return f"✅ Uploaded instruction (id={doc_id})"
        except Exception as e:
            if self.logger:
                self.logger.error(f"upload_instruction error: {e}")
            try:
                log_tool_execution(self.agent.name, 'vector_memory.upload_instruction', 0, success=False)
            except Exception:
                pass
            return f"❌ Upload failed: {e}"

    # @tool(description="Owner-only: remove an instruction document by id.", scope="owner")
    async def remove_instruction(self, doc_id: str, context: Any = None) -> str:
        if not self._ensure_collection():
            return "❌ Vector memory unavailable"
        try:
            expr = f"id == '{doc_id}'"
            self.collection.delete(expr)
            try:
                self.collection.flush()
            except Exception:
                pass
            try:
                log_tool_execution(self.agent.name, 'vector_memory.remove_instruction', 0, success=True)
            except Exception:
                pass
            return f"🗑️ Removed instruction (id={doc_id})"
        except Exception as e:
            if self.logger:
                self.logger.error(f"remove_instruction error: {e}")
            try:
                log_tool_execution(self.agent.name, 'vector_memory.remove_instruction', 0, success=False)
            except Exception:
                pass
            return f"❌ Remove failed: {e}"

    # ---------------- Admin tools for common instructions ----------------
    # @tool(description="Admin: list common instruction documents (agent_id == '')", scope="admin")
    async def list_common_instructions(self, limit: int = 50, context: Any = None) -> str:
        if not self._ensure_collection():
            return "❌ Vector memory unavailable"
        try:
            # Query all common docs (agent_id == '')
            # Use a vector-less query via query API
            results = self.collection.query(
                expr="agent_id == ''",
                output_fields=["id", "title", "owner_user_id"],
                limit=max(1, int(limit))
            )
            if not results:
                try:
                    log_tool_execution(self.agent.name, 'vector_memory.list_common_instructions', 0, success=True)
                except Exception:
                    pass
                return "(no common instructions)"
            lines = [f"- {r.get('id')}: {r.get('title') or 'Untitled'} (owner={r.get('owner_user_id') or ''})" for r in results]
            try:
                log_tool_execution(self.agent.name, 'vector_memory.list_common_instructions', 0, success=True)
            except Exception:
                pass
            return "\n".join(lines)
        except Exception as e:
            if self.logger:
                self.logger.error(f"list_common_instructions error: {e}")
            try:
                log_tool_execution(self.agent.name, 'vector_memory.list_common_instructions', 0, success=False)
            except Exception:
                pass
            return "❌ List failed"

    # @tool(description="Admin: create a common instruction document", scope="admin")
    async def create_common_instruction(self, title: str, content: str, context: Any = None) -> str:
        if not self._ensure_collection():
            return "❌ Vector memory unavailable"
        try:
            vecs = await self._embed([f"{title}\n\n{content}"])
            vec = vecs[0] if vecs else [0.0] * self.dim
            doc_id = uuid.uuid4().hex
            entities = [
                [doc_id],
                [""],  # agent_id empty => common
                [getattr(self.agent, 'owner_user_id', '') or ''],
                [title[:255]],
                [content[:8000]],
                [vec],
            ]
            self.collection.insert(entities)
            try:
                self.collection.flush()
            except Exception:
                pass
            try:
                log_tool_execution(self.agent.name, 'vector_memory.create_common_instruction', 0, success=True)
            except Exception:
                pass
            return f"✅ Created common instruction (id={doc_id})"
        except Exception as e:
            if self.logger:
                self.logger.error(f"create_common_instruction error: {e}")
            try:
                log_tool_execution(self.agent.name, 'vector_memory.create_common_instruction', 0, success=False)
            except Exception:
                pass
            return f"❌ Create failed: {e}"

    # @tool(description="Admin: update a common instruction document (title and/or content)", scope="admin")
    async def update_common_instruction(self, doc_id: str, title: Optional[str] = None, content: Optional[str] = None, context: Any = None) -> str:
        if not self._ensure_collection():
            return "❌ Vector memory unavailable"
        if not title and not content:
            return "Nothing to update"
        try:
            # Fetch existing doc to preserve fields
            rows = self.collection.query(expr=f"id == '{doc_id}' and agent_id == ''", output_fields=["id", "title", "content"], limit=1)
            if not rows:
                return "❌ Document not found or not common"
            current = rows[0]
            new_title = title if title is not None else current.get('title')
            new_content = content if content is not None else current.get('content')
            vecs = await self._embed([f"{new_title}\n\n{new_content}"])
            vec = vecs[0] if vecs else [0.0] * self.dim
            # Delete + insert (Milvus doesn't support partial update of vectors)
            self.collection.delete(f"id == '{doc_id}'")
            try:
                self.collection.flush()
            except Exception:
                pass
            entities = [
                [doc_id],
                [""],
                [getattr(self.agent, 'owner_user_id', '') or ''],
                [new_title[:255]],
                [new_content[:8000]],
                [vec],
            ]
            self.collection.insert(entities)
            try:
                self.collection.flush()
            except Exception:
                pass
            try:
                log_tool_execution(self.agent.name, 'vector_memory.update_common_instruction', 0, success=True)
            except Exception:
                pass
            return "✅ Updated"
        except Exception as e:
            if self.logger:
                self.logger.error(f"update_common_instruction error: {e}")
            try:
                log_tool_execution(self.agent.name, 'vector_memory.update_common_instruction', 0, success=False)
            except Exception:
                pass
            return f"❌ Update failed: {e}"

    # @tool(description="Admin: delete a common instruction document by id", scope="admin")
    async def delete_common_instruction(self, doc_id: str, context: Any = None) -> str:
        if not self._ensure_collection():
            return "❌ Vector memory unavailable"
        try:
            self.collection.delete(f"id == '{doc_id}' and agent_id == ''")
            try:
                self.collection.flush()
            except Exception:
                pass
            try:
                log_tool_execution(self.agent.name, 'vector_memory.delete_common_instruction', 0, success=True)
            except Exception:
                pass
            return "🗑️ Deleted"
        except Exception as e:
            if self.logger:
                self.logger.error(f"delete_common_instruction error: {e}")
            try:
                log_tool_execution(self.agent.name, 'vector_memory.delete_common_instruction', 0, success=False)
            except Exception:
                pass
            return f"❌ Delete failed: {e}"

    def get_guidance_prompt(self) -> str:
        return (
            "You have vector memory (common & agent-specific).\n"
            "- MAY call fetch_instructions_tool ONCE per conversation for domain guidance.\n"
            "- Use relevant instructions; ignore generic ones.\n"
            "- Owner/admin: use upload_instruction/remove_instruction or create/update/delete_common_instruction.\n"
        )

    def get_tool_prompt(self) -> str:
        """Detailed prompt block that can be injected into the system prompt."""
        return (
            "- fetch_instructions_tool(problem, top_k=1): retrieve instruction documents\n"
            "Use ONCE per conversation if needed.\n"
        )

    # @prompt blocks to auto-inject scoped guidance
    @prompt(priority=20, scope="all")
    def vector_memory_general_prompt(self, context: Any = None) -> str:
        return self.get_guidance_prompt() + "\n\n" + self.get_tool_prompt()

    @prompt(priority=25, scope="owner")
    def vector_memory_owner_prompt(self, context: Any = None) -> str:
        return "OWNER: Curate using upload_instruction(title, content, agent_specific=True) / remove_instruction(doc_id).\n"

    @prompt(priority=25, scope="admin")
    def vector_memory_admin_prompt(self, context: Any = None) -> str:
        return "ADMIN: Manage using list/create/update/delete_common_instruction tools.\n"


