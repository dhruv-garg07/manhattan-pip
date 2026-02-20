"""
Coding Vector Store

Dedicated vector storage for code chunks in .gitmem_coding/.
Mirrors gitmem's LocalVectorStore pattern: stores vectors in a separate
vectors.json per agent, keyed by chunk hash_id.

Storage Structure:
    .gitmem_coding/
    └── agents/{agent_id}/
        └── vectors.json   # {hash_id: [float, float, ...]}
"""

import os
import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..gitmem.embedding import (
    RemoteEmbeddingClient,
    get_embedding_client,
    create_vector,
)
import logging

logger = logging.getLogger(__name__)


class CodingVectorStore:
    """
    Local vector storage for coding chunks using a dedicated vectors.json file.
    
    Each agent gets its own vectors.json inside .gitmem_coding/agents/{agent_id}/.
    Vectors are keyed by the chunk's hash_id for deduplication.
    """

    def __init__(
        self,
        root_path: str = "./.gitmem_coding",
        embedding_client: RemoteEmbeddingClient = None,
    ):
        self.root_path = Path(root_path).absolute()
        self._lock = threading.RLock()
        # In-memory cache: {agent_id: {hash_id: vector_list}}
        self._vector_cache: Dict[str, Dict[str, List[float]]] = {}

        # Reuse global embedding client singleton
        cache_path = self.root_path / ".embedding_cache"
        self.embedding_client = embedding_client or get_embedding_client(
            cache_path=str(cache_path)
        )

    # -------------------------------------------------------------------------
    # Path helpers
    # -------------------------------------------------------------------------
    def _get_agent_path(self, agent_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in agent_id)
        return self.root_path / "agents" / safe_id

    def _get_vectors_path(self, agent_id: str) -> Path:
        return self._get_agent_path(agent_id) / "vectors.json"

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _to_list(vector) -> List[float]:
        """Convert a vector (numpy or list) to a plain list for JSON."""
        if hasattr(vector, "tolist"):
            return vector.tolist()
        return list(vector)

    # -------------------------------------------------------------------------
    # Load / Save
    # -------------------------------------------------------------------------
    def _load_vectors(self, agent_id: str) -> Dict[str, List[float]]:
        """Load vectors.json for an agent (with in-memory caching)."""
        if agent_id in self._vector_cache:
            cached = self._vector_cache[agent_id]
            if cached is not None:
                return cached

        vectors_path = self._get_vectors_path(agent_id)
        if vectors_path.exists():
            try:
                with open(vectors_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if data is None:
                    data = {}

                self._vector_cache[agent_id] = data
                return data
            except Exception as e:
                logger.error(f"[CodingVectorStore] Failed to load vectors: {e}")
                return {}

        return {}

    def _save_vectors(self, agent_id: str, vectors: Dict[str, List[float]]):
        """Persist vectors.json for an agent."""
        vectors_path = self._get_vectors_path(agent_id)
        vectors_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            try:
                with open(vectors_path, "w", encoding="utf-8") as f:
                    json.dump(vectors, f)
                self._vector_cache[agent_id] = vectors
            except Exception as e:
                logger.error(f"[CodingVectorStore] Failed to save vectors: {e}")

    # -------------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------------
    def add_vector(
        self, agent_id: str, hash_id: str, text: str
    ) -> Optional[List[float]]:
        """
        Generate an embedding for *text* and store it keyed by *hash_id*.
        Returns the vector list, or None on failure.
        """
        try:
            embedding = self.embedding_client.embed(text)
            vec_list = self._to_list(embedding)

            vectors = self._load_vectors(agent_id)
            vectors[hash_id] = vec_list
            self._save_vectors(agent_id, vectors)
            return vec_list
        except Exception as e:
            logger.error(f"[CodingVectorStore] Failed to add vector for {hash_id}: {e}")
            return None

    def add_vector_raw(
        self, agent_id: str, hash_id: str, vector: List[float]
    ):
        """Store a pre-computed vector (no embedding call)."""
        vec_list = self._to_list(vector)
        vectors = self._load_vectors(agent_id)
        vectors[hash_id] = vec_list
        self._save_vectors(agent_id, vectors)

    def get_vector(self, agent_id: str, hash_id: str) -> Optional[List[float]]:
        """Retrieve a single vector by hash_id."""
        vectors = self._load_vectors(agent_id)
        return vectors.get(hash_id)

    def get_vectors_batch(
        self, agent_id: str, hash_ids: List[str]
    ) -> Dict[str, List[float]]:
        """Retrieve multiple vectors at once."""
        vectors = self._load_vectors(agent_id)
        return {h: vectors[h] for h in hash_ids if h in vectors}

    def delete_vector(self, agent_id: str, hash_id: str) -> bool:
        """Delete a vector entry."""
        vectors = self._load_vectors(agent_id)
        if hash_id in vectors:
            del vectors[hash_id]
            self._save_vectors(agent_id, vectors)
            return True
        return False
    
    def delete_vectors(self, agent_id: str, hash_ids: List[str]) -> int:
        """Bulk delete vector entries. Returns count of deleted vectors."""
        if not hash_ids:
            return 0
        vectors = self._load_vectors(agent_id)
        deleted = 0
        for hid in hash_ids:
            if hid in vectors:
                del vectors[hid]
                deleted += 1
        if deleted > 0:
            self._save_vectors(agent_id, vectors)
        return deleted

    def clear_vectors(self, agent_id: str):
        """Remove all vectors for an agent."""
        vectors_path = self._get_vectors_path(agent_id)
        if vectors_path.exists():
            vectors_path.unlink()
        self._vector_cache.pop(agent_id, None)

    def get_stats(self, agent_id: str) -> Dict[str, Any]:
        """Return basic statistics."""
        vectors = self._load_vectors(agent_id)
        dim = 0
        if vectors:
            first = next(iter(vectors.values()))
            dim = len(first) if first else 0
        return {
            "agent_id": agent_id,
            "vector_count": len(vectors),
            "dimension": dim,
        }
