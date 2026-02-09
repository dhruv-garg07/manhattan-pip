"""
GitMem Local - Vector Store

Local vector storage engine that stores embeddings in JSON files.
No external database required - all vectors stored alongside memories.

Features:
- JSON-based vector storage (no ChromaDB dependency)
- Semantic search using cosine similarity
- Keyword/lexical search using BM25-like scoring
- Hybrid search combining both approaches
- Thread-safe operations
- No external dependencies (works without numpy/requests)
- Global singleton pattern for efficiency
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import threading
import hashlib

# Import vector utilities from embedding module (works with or without numpy)
from .embedding import (
    RemoteEmbeddingClient,
    create_vector,
    zeros_vector,
    vector_norm,
    vector_dot,
    HAS_NUMPY,
    get_embedding_client
)


# =============================================================================
# Global Singleton Pattern for Vector Store
# =============================================================================

_vector_store_singleton: Optional['LocalVectorStore'] = None
_vector_store_lock = threading.Lock()


def get_vector_store(root_path: str = "./.gitmem_data") -> 'LocalVectorStore':
    """
    Get or create the global vector store singleton.
    
    This avoids repeated initialization overhead.
    
    Args:
        root_path: Root path for gitmem data (only used on first creation)
    
    Returns:
        The global LocalVectorStore instance
    """
    global _vector_store_singleton
    
    if _vector_store_singleton is None:
        with _vector_store_lock:
            if _vector_store_singleton is None:
                _vector_store_singleton = LocalVectorStore(root_path=root_path)
    
    return _vector_store_singleton


def reset_vector_store():
    """Reset the global vector store (for testing or reconfiguration)."""
    global _vector_store_singleton
    with _vector_store_lock:
        _vector_store_singleton = None


class LocalVectorStore:
    """
    Local vector storage engine using JSON files.
    
    Stores embeddings alongside memories in the gitmem data directory.
    Supports semantic, keyword, and hybrid search.
    
    Uses a global singleton embedding client for efficiency.
    
    Storage Structure:
        .gitmem_data/
        ├── agents/{agent_id}/
        │   ├── memories.json     # Memory entries (with vector field)
        │   ├── vectors.json      # Vector index (separate for efficiency)
        │   └── ...
        └── embedding_cache.json  # Shared embedding cache
    """
    
    def __init__(
        self,
        root_path: str = "./.gitmem_data",
        embedding_client: RemoteEmbeddingClient = None,
        auto_embed: bool = True
    ):
        """
        Initialize the local vector store.
        
        Args:
            root_path: Root path for gitmem data
            embedding_client: Optional custom embedding client (uses global singleton if None)
            auto_embed: Whether to automatically generate embeddings for new entries
        """
        self.root_path = Path(root_path).absolute()
        self.auto_embed = auto_embed
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Vector caches per agent (stores Python lists or numpy arrays)
        self._vector_cache: Dict[str, Dict[str, Any]] = {}
        
        # Use provided client or global singleton
        cache_path = self.root_path / ".embedding_cache"
        self.embedding_client = embedding_client or get_embedding_client(
            cache_path=str(cache_path)
        )
        
        # Ensure directories exist
        self.root_path.mkdir(parents=True, exist_ok=True)
    
    def _get_agent_path(self, agent_id: str) -> Path:
        """Get the storage path for an agent."""
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in agent_id)
        return self.root_path / "agents" / safe_id
    
    def _get_vectors_path(self, agent_id: str) -> Path:
        """Get the vectors file path for an agent."""
        return self._get_agent_path(agent_id) / "vectors.json"
    
    def _to_list(self, vector) -> List[float]:
        """Convert a vector to a list for JSON serialization."""
        if hasattr(vector, 'tolist'):
            return vector.tolist()
        elif hasattr(vector, 'data'):
            return vector.data
        return list(vector)
    
    def _load_vectors(self, agent_id: str) -> Dict[str, List[float]]:
        """Load vector index for an agent."""
        vectors_path = self._get_vectors_path(agent_id)
        
        if agent_id in self._vector_cache:
            return {k: self._to_list(v) for k, v in self._vector_cache[agent_id].items()}
        
        if vectors_path.exists():
            try:
                with open(vectors_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Cache as vectors (numpy or pure Python)
                    self._vector_cache[agent_id] = {
                        k: create_vector(v) for k, v in data.items()
                    }
                    return data
            except Exception as e:
                print(f"[VectorStore] Failed to load vectors: {e}")
        
        return {}
    
    def _save_vectors(self, agent_id: str, vectors: Dict[str, List[float]]):
        """Save vector index for an agent."""
        vectors_path = self._get_vectors_path(agent_id)
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            try:
                with open(vectors_path, 'w', encoding='utf-8') as f:
                    json.dump(vectors, f)
                
                # Update cache
                self._vector_cache[agent_id] = {
                    k: create_vector(v) for k, v in vectors.items()
                }
            except Exception as e:
                print(f"[VectorStore] Failed to save vectors: {e}")
    
    def add_vector(self, agent_id: str, entry_id: str, text: str) -> Optional[List[float]]:
        """
        Generate and store a vector embedding for text.
        
        Args:
            agent_id: The agent ID
            entry_id: The memory entry ID
            text: Text to embed
        
        Returns:
            The embedding vector as a list, or None if failed
        """
        try:
            # Generate embedding
            embedding = self.embedding_client.embed(text)
            embedding_list = self._to_list(embedding)
            
            # Load existing vectors
            vectors = self._load_vectors(agent_id)
            
            # Add new vector
            vectors[entry_id] = embedding_list
            
            # Save back
            self._save_vectors(agent_id, vectors)
            
            return embedding_list
            
        except Exception as e:
            print(f"[VectorStore] Failed to add vector: {e}")
            return None
    
    def add_vectors_batch(
        self,
        agent_id: str,
        entries: List[Dict[str, Any]]
    ) -> int:
        """
        Add vectors for multiple entries in batch.
        
        Args:
            agent_id: The agent ID
            entries: List of entries with 'id' and 'content' or 'lossless_restatement'
        
        Returns:
            Number of vectors added
        """
        if not entries:
            return 0
        
        # Load existing vectors
        vectors = self._load_vectors(agent_id)
        added = 0
        
        for entry in entries:
            entry_id = entry.get("id")
            if not entry_id:
                continue
            
            # Skip if already has vector
            if entry_id in vectors:
                continue
            
            # Get text to embed
            text = entry.get("lossless_restatement") or entry.get("content", "")
            if not text:
                continue
            
            try:
                embedding = self.embedding_client.embed(text)
                vectors[entry_id] = self._to_list(embedding)
                added += 1
            except Exception as e:
                print(f"[VectorStore] Failed to embed {entry_id}: {e}")
        
        if added > 0:
            self._save_vectors(agent_id, vectors)
            print(f"[VectorStore] Added {added} vectors for {agent_id}")
        
        return added
    
    def get_vector(self, agent_id: str, entry_id: str):
        """Get the vector for a specific entry."""
        if agent_id in self._vector_cache and entry_id in self._vector_cache[agent_id]:
            return self._vector_cache[agent_id][entry_id]
        
        vectors = self._load_vectors(agent_id)
        if entry_id in vectors:
            return create_vector(vectors[entry_id])
        
        return None
    
    def delete_vector(self, agent_id: str, entry_id: str) -> bool:
        """Delete a vector for a specific entry."""
        vectors = self._load_vectors(agent_id)
        
        if entry_id in vectors:
            del vectors[entry_id]
            self._save_vectors(agent_id, vectors)
            return True
        
        return False
    
    def semantic_search(
        self,
        agent_id: str,
        query: str,
        memories: List[Dict[str, Any]],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using vector similarity.
        
        Args:
            agent_id: The agent ID
            query: Search query
            memories: List of memory entries to search
            top_k: Number of results to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of matching memories with similarity scores
        """
        if not memories:
            return []
        
        # Get query embedding
        try:
            query_embedding = self.embedding_client.embed(query)
        except Exception as e:
            print(f"[VectorStore] Failed to embed query: {e}")
            return []
        
        # Load vectors
        vectors_data = self._load_vectors(agent_id)
        
        # Compute similarities
        results = []
        for mem in memories:
            entry_id = mem.get("id")
            if not entry_id:
                continue
            
            # Get vector (from cache or generate)
            if entry_id in vectors_data:
                vector = create_vector(vectors_data[entry_id])
            elif self.auto_embed:
                # Generate embedding on the fly
                text = mem.get("lossless_restatement") or mem.get("content", "")
                if text:
                    vector_list = self.add_vector(agent_id, entry_id, text)
                    if vector_list:
                        vector = create_vector(vector_list)
                    else:
                        continue
                else:
                    continue
            else:
                continue
            
            # Calculate cosine similarity
            similarity = self.embedding_client.cosine_similarity(query_embedding, vector)
            
            if similarity >= threshold:
                results.append({
                    **mem,
                    "semantic_score": float(similarity)
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x["semantic_score"], reverse=True)
        
        return results[:top_k]
    
    def keyword_search(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Keyword/lexical search using BM25-like scoring.
        
        Args:
            query: Search query
            memories: List of memory entries to search
            top_k: Number of results to return
        
        Returns:
            List of matching memories with keyword scores
        """
        # Stopwords to filter
        STOPWORDS = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
            'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'and', 'but', 'if', 'or', 'because', 'until',
            'while', 'about', 'against', 'up', 'down', 'out', 'off', 'over', 'under',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
            'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who',
            'whom', 'this', 'that', 'these', 'those', 'am', 'been', 'being', 'how',
            'when', 'where', 'why', 'all', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'user', 'agent', 'memory', 'loves', 'like', 'likes'
        }
        
        # Tokenize query
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if w not in STOPWORDS and len(w) > 2]
        
        if not query_words:
            # Fall back to all words if too many filtered
            query_words = [w for w in query_lower.split() if len(w) > 2]
        
        if not query_words:
            return []
        
        results = []
        for mem in memories:
            content = (mem.get("lossless_restatement") or mem.get("content", "")).lower()
            keywords = [k.lower() for k in mem.get("keywords", [])]
            topic = mem.get("topic", "").lower()
            persons = [p.lower() for p in mem.get("persons", [])]
            
            score = 0.0
            matched = 0
            
            # Phrase match bonus
            if len(query_words) >= 2:
                phrase = " ".join(query_words)
                if phrase in content:
                    score += 2.0
            
            # Word matching
            for word in query_words:
                word_score = 0.0
                
                # Content match
                if word in content:
                    word_score += 0.3 + (len(word) * 0.05)
                    matched += 1
                
                # Keyword match
                for kw in keywords:
                    if word in kw or kw in word:
                        word_score += 0.6
                        break
                
                # Topic match
                if word in topic:
                    word_score += 0.4
                
                # Person match
                for person in persons:
                    if word in person:
                        word_score += 0.5
                        break
                
                score += word_score
            
            # Match ratio boost
            if query_words:
                match_ratio = matched / len(query_words)
                score *= (0.5 + match_ratio)
            
            # Importance boost
            score *= (1 + mem.get("importance", 0.5))
            
            if score > 0.3 and matched > 0:
                results.append({
                    **mem,
                    "keyword_score": round(score, 3)
                })
        
        # Sort by score
        results.sort(key=lambda x: x["keyword_score"], reverse=True)
        
        return results[:top_k]
    
    def hybrid_search(
        self,
        agent_id: str,
        query: str,
        memories: List[Dict[str, Any]],
        top_k: int = 5,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword matching.
        
        Paper Reference: Section 3.3 - Hybrid Scoring Function S(q, m_k)
        S(q, m_k) = λ₁·cos(e_q, v_k) + λ₂·BM25(q, S_k) + γ·metadata_match
        
        Args:
            agent_id: The agent ID
            query: Search query
            memories: List of memory entries to search
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity (λ₁)
            keyword_weight: Weight for keyword matching (λ₂)
            threshold: Minimum combined score threshold
        
        Returns:
            List of matching memories with hybrid scores
        """
        if not memories:
            return []
        
        # Get semantic results
        semantic_results = self.semantic_search(
            agent_id=agent_id,
            query=query,
            memories=memories,
            top_k=min(top_k * 3, len(memories)),  # Get more for merging
            threshold=0.0
        )
        
        # Get keyword results
        keyword_results = self.keyword_search(
            query=query,
            memories=memories,
            top_k=min(top_k * 3, len(memories))
        )
        
        # Merge and combine scores
        entry_scores: Dict[str, Dict[str, Any]] = {}
        
        # Add semantic results
        max_semantic = max([r.get("semantic_score", 0) for r in semantic_results], default=1.0) or 1.0
        for result in semantic_results:
            entry_id = result.get("id")
            if entry_id:
                normalized_score = result.get("semantic_score", 0) / max_semantic
                entry_scores[entry_id] = {
                    "entry": result,
                    "semantic": normalized_score,
                    "keyword": 0.0
                }
        
        # Add keyword results
        max_keyword = max([r.get("keyword_score", 0) for r in keyword_results], default=1.0) or 1.0
        for result in keyword_results:
            entry_id = result.get("id")
            if entry_id:
                normalized_score = result.get("keyword_score", 0) / max_keyword
                if entry_id in entry_scores:
                    entry_scores[entry_id]["keyword"] = normalized_score
                else:
                    entry_scores[entry_id] = {
                        "entry": result,
                        "semantic": 0.0,
                        "keyword": normalized_score
                    }
        
        # Calculate hybrid scores
        results = []
        for entry_id, scores in entry_scores.items():
            hybrid_score = (
                semantic_weight * scores["semantic"] +
                keyword_weight * scores["keyword"]
            )
            
            if hybrid_score >= threshold:
                entry = scores["entry"].copy()
                entry["hybrid_score"] = round(hybrid_score, 3)
                entry["semantic_score"] = round(scores["semantic"], 3)
                entry["keyword_score"] = round(scores["keyword"], 3)
                results.append(entry)
        
        # Sort by hybrid score
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return results[:top_k]
    
    def get_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get vector store statistics for an agent."""
        vectors = self._load_vectors(agent_id)
        
        return {
            "agent_id": agent_id,
            "vector_count": len(vectors),
            "dimension": self.embedding_client.dimension,
            "cache_size": len(self.embedding_client._cache)
        }
    
    def clear_vectors(self, agent_id: str):
        """Clear all vectors for an agent."""
        vectors_path = self._get_vectors_path(agent_id)
        
        if vectors_path.exists():
            vectors_path.unlink()
        
        if agent_id in self._vector_cache:
            del self._vector_cache[agent_id]
        
        print(f"[VectorStore] Cleared vectors for {agent_id}")


# Global instance (lazy initialization)
_vector_store: Optional[LocalVectorStore] = None


def get_vector_store(root_path: str = "./.gitmem_data") -> LocalVectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = LocalVectorStore(root_path=root_path)
    return _vector_store
