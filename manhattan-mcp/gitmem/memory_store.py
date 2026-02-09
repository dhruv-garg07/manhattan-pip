"""
GitMem Local - Memory Store

Local storage backend for AI context data using JSON files.
No external databases or web APIs required.

Storage Structure:
    .gitmem_data/
    ├── agents/
    │   └── {agent_id}/
    │       ├── memories.json
    │       ├── documents.json
    │       ├── checkpoints.json
    │       ├── logs.json
    │       ├── vectors.json     # Vector embeddings
    │       └── settings.json
    ├── index/
    │   └── keywords.json
    └── config.json

Features:
    - JSON-based local storage
    - Vector embeddings for semantic search
    - Hybrid retrieval (keyword + semantic)
    - Git-like version control via MemoryDAG
"""

import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading

from .object_store import MemoryDAG, MemoryBlob
from .models import MemoryEntry, Checkpoint, ActivityLog


class LocalMemoryStore:
    """
    Local JSON-based storage for AI context data.
    
    Thread-safe storage with automatic persistence.
    Supports vector embeddings for semantic search.
    
    Features:
        - Memory CRUD operations
        - Keyword-based search
        - Vector-based semantic search (optional)
        - Hybrid search combining both
        - Git-like version control
    """
    
    def __init__(
        self,
        root_path: str = "./.gitmem_data",
        enable_vectors: bool = True
    ):
        """
        Initialize the memory store.
        
        Args:
            root_path: Root directory for storage
            enable_vectors: Whether to enable vector embeddings for semantic search
        """
        self.root_path = Path(root_path).absolute()
        self.agents_path = self.root_path / "agents"
        self.index_path = self.root_path / "index"
        self.config_path = self.root_path / "config.json"
        self.enable_vectors = enable_vectors
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory caches
        self._agents_cache: Dict[str, Dict] = {}
        self._config: Dict = {}
        
        # Initialize
        self._ensure_dirs()
        self._load_config()
        
        # Initialize MemoryDAG for version control
        self.dag = MemoryDAG(str(self.root_path / ".gitmem"))
        
        # Initialize vector store (lazy loaded)
        self._vector_store = None
        self._hybrid_retriever = None
    
    def _ensure_dirs(self):
        """Create necessary directory structure."""
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.agents_path.mkdir(exist_ok=True)
        self.index_path.mkdir(exist_ok=True)
    
    def _load_config(self):
        """Load global configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        else:
            self._config = {
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "default_agent": "default"
            }
            self._save_config()
    
    def _save_config(self):
        """Save global configuration."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2)
    
    def _get_agent_path(self, agent_id: str) -> Path:
        """Get the storage path for an agent."""
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in agent_id)
        return self.agents_path / safe_id
    
    def _ensure_agent(self, agent_id: str):
        """Ensure agent directory and files exist."""
        agent_path = self._get_agent_path(agent_id)
        agent_path.mkdir(exist_ok=True)
        
        # Initialize default files
        defaults = {
            "memories.json": [],
            "documents.json": [],
            "checkpoints.json": [],
            "logs.json": [],
            "settings.json": {"created_at": datetime.now().isoformat()}
        }
        
        for filename, default_content in defaults.items():
            filepath = agent_path / filename
            if not filepath.exists():
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(default_content, f, indent=2)
    
    def _load_agent_data(self, agent_id: str, data_type: str) -> List[Dict]:
        """Load agent data from JSON file."""
        self._ensure_agent(agent_id)
        filepath = self._get_agent_path(agent_id) / f"{data_type}.json"
        
        with self._lock:
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        return []
    
    def _save_agent_data(self, agent_id: str, data_type: str, data: List[Dict]):
        """Save agent data to JSON file."""
        self._ensure_agent(agent_id)
        filepath = self._get_agent_path(agent_id) / f"{data_type}.json"
        
        with self._lock:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
    
    # =========================================================================
    # Memory Operations
    # =========================================================================
    
    def add_memory(self, agent_id: str, content: str, memory_type: str = "episodic",
                   lossless_restatement: str = None, keywords: List[str] = None,
                   persons: List[str] = None, entities: List[str] = None,
                   topic: str = None, importance: float = 0.5,
                   metadata: Dict = None) -> str:
        """
        Add a memory entry for an agent.
        
        Args:
            agent_id: The agent ID
            content: Raw content
            memory_type: episodic, semantic, procedural, working
            lossless_restatement: Clear restatement of the fact
            keywords: Searchable keywords
            persons: People mentioned
            entities: Entities mentioned
            topic: Topic category
            importance: 0-1 importance score
            metadata: Additional metadata
        
        Returns:
            Memory entry ID
        """
        entry_id = str(uuid.uuid4())
        
        memory = {
            "id": entry_id,
            "content": content,
            "lossless_restatement": lossless_restatement or content,
            "memory_type": memory_type,
            "keywords": keywords or [],
            "persons": persons or [],
            "entities": entities or [],
            "topic": topic or "",
            "importance": importance,
            "metadata": metadata or {},
            "agent_id": agent_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        memories = self._load_agent_data(agent_id, "memories")
        memories.append(memory)
        self._save_agent_data(agent_id, "memories", memories)
        
        # Also stage in DAG for version control
        self.dag.set_agent(agent_id)
        self.dag.add(
            content=lossless_restatement or content,
            memory_type=memory_type,
            importance=importance,
            keywords=keywords,
            persons=persons,
            metadata={"entry_id": entry_id, **(metadata or {})}
        )
        
        # Log activity
        self._log_activity(agent_id, "mutation", "create", "memory", entry_id, "agent", agent_id)
        
        return entry_id
    
    # Common stopwords to filter out from search queries
    STOPWORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
        'because', 'until', 'while', 'about', 'against', 'up', 'down', 'out',
        'off', 'over', 'any', 'both', 'this', 'that', 'these', 'those', 'am',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'does', 'doing', 'having', 'loves', 'like', 'likes', 'user', 'agent'
    }
    
    def search_memory(self, agent_id: str, query: str, top_k: int = 5,
                      memory_type: str = None) -> List[Dict]:
        """
        Search memories using improved keyword matching.
        
        Features:
        - Stopword filtering
        - Phrase matching bonus
        - TF-IDF-like weighting
        - Keyword and topic matching
        
        Args:
            agent_id: The agent ID
            query: Search query
            top_k: Number of results
            memory_type: Filter by type
        
        Returns:
            List of matching memories with scores
        """
        memories = self._load_agent_data(agent_id, "memories")
        query_lower = query.lower().strip()
        
        # Extract meaningful words (filter stopwords)
        all_words = query_lower.split()
        meaningful_words = [w for w in all_words if w not in self.STOPWORDS and len(w) > 2]
        
        # Fall back to all words if too many filtered
        if len(meaningful_words) == 0:
            meaningful_words = [w for w in all_words if len(w) > 2]
        
        results = []
        for mem in memories:
            # Filter by type if specified
            if memory_type and mem.get("memory_type") != memory_type:
                continue
            
            # Get searchable content
            content = (mem.get("lossless_restatement") or mem.get("content", "")).lower()
            keywords = [k.lower() for k in mem.get("keywords", [])]
            topic = mem.get("topic", "").lower()
            persons = [p.lower() for p in mem.get("persons", [])]
            
            score = 0.0
            matched_words = 0
            
            # 1. Phrase match bonus (highest priority)
            if len(meaningful_words) >= 2:
                # Check if consecutive words appear together
                phrase = " ".join(meaningful_words)
                if phrase in content:
                    score += 2.0  # Strong phrase match bonus
            
            # 2. Individual word matching with weighting
            for word in meaningful_words:
                word_score = 0.0
                
                # Content match - higher weight for less common words
                if word in content:
                    # Approximate TF-IDF: shorter words are more common
                    weight = 0.3 + (len(word) * 0.05)  # Longer words get more weight
                    word_score += weight
                    matched_words += 1
                
                # Keyword match (high priority - explicit keywords)
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
            
            # 3. Calculate match ratio (what % of query words matched)
            if meaningful_words:
                match_ratio = matched_words / len(meaningful_words)
                score *= (0.5 + match_ratio)  # Boost based on how many words matched
            
            # 4. Boost by importance
            score *= (1 + mem.get("importance", 0.5))
            
            # 5. Only include if we have a meaningful match
            if score > 0.3 and matched_words > 0:
                results.append({
                    **mem,
                    "score": round(score, 3),
                    "matched_words": matched_words,
                    "total_query_words": len(meaningful_words)
                })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    @property
    def vector_store(self):
        """
        Get the vector store instance (uses global singleton).
        
        The vector store enables semantic search using embeddings.
        Uses a global singleton to avoid repeated initialization.
        """
        if self._vector_store is None and self.enable_vectors:
            try:
                from .vector_store import get_vector_store
                # Use global singleton vector store
                self._vector_store = get_vector_store(root_path=str(self.root_path))
                print("[MemoryStore] Vector store initialized (singleton)")
            except ImportError as e:
                print(f"[MemoryStore] Vector store not available: {e}")
                self.enable_vectors = False
        return self._vector_store
    
    @property
    def hybrid_retriever(self):
        """
        Get the hybrid retriever instance (uses global singleton).
        
        The hybrid retriever combines semantic and keyword search.
        Uses a global singleton to avoid repeated initialization.
        """
        if self._hybrid_retriever is None and self.enable_vectors:
            try:
                from .hybrid_retriever import get_retriever, RetrievalConfig
                config = RetrievalConfig(
                    semantic_weight=0.6,
                    keyword_weight=0.4,
                    enable_reflection=False
                )
                # Use global singleton retriever
                self._hybrid_retriever = get_retriever(
                    vector_store=self.vector_store,
                    config=config
                )
                print("[MemoryStore] Hybrid retriever initialized (singleton)")
            except ImportError as e:
                print(f"[MemoryStore] Hybrid retriever not available: {e}")
        return self._hybrid_retriever
    
    def hybrid_search_memory(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        memory_type: str = None,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[Dict]:
        """
        Hybrid search combining semantic (vector) and keyword matching.
        
        This method provides the best search results by combining:
        - Semantic similarity using vector embeddings
        - Keyword/lexical matching using BM25-like scoring
        
        Falls back to keyword-only search if vectors are not available.
        
        Args:
            agent_id: The agent ID
            query: Search query
            top_k: Number of results to return
            memory_type: Optional filter by memory type
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
        
        Returns:
            List of matching memories with combined scores
        """
        memories = self._load_agent_data(agent_id, "memories")
        
        # Filter by type if specified
        if memory_type:
            memories = [m for m in memories if m.get("memory_type") == memory_type]
        
        if not memories:
            return []
        
        # Check if hybrid retriever is available
        if self.enable_vectors and self.hybrid_retriever:
            try:
                # Use hybrid retriever
                results = self.hybrid_retriever.retrieve(
                    agent_id=agent_id,
                    query=query,
                    memories=memories,
                    top_k=top_k
                )
                return results
            except Exception as e:
                print(f"[MemoryStore] Hybrid search failed, falling back to keyword: {e}")
        
        # Fallback to keyword search
        return self.search_memory(agent_id, query, top_k, memory_type)
    
    def semantic_search_memory(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        memory_type: str = None
    ) -> List[Dict]:
        """
        Pure semantic search using vector embeddings.
        
        Args:
            agent_id: The agent ID
            query: Search query
            top_k: Number of results to return
            memory_type: Optional filter by memory type
        
        Returns:
            List of matching memories with semantic scores
        """
        if not self.enable_vectors or not self.vector_store:
            print("[MemoryStore] Vectors not enabled, using keyword search")
            return self.search_memory(agent_id, query, top_k, memory_type)
        
        memories = self._load_agent_data(agent_id, "memories")
        
        # Filter by type if specified
        if memory_type:
            memories = [m for m in memories if m.get("memory_type") == memory_type]
        
        if not memories:
            return []
        
        try:
            return self.vector_store.semantic_search(
                agent_id=agent_id,
                query=query,
                memories=memories,
                top_k=top_k
            )
        except Exception as e:
            print(f"[MemoryStore] Semantic search failed: {e}")
            return self.search_memory(agent_id, query, top_k, memory_type)
    
    def ensure_memory_vectors(self, agent_id: str) -> int:
        """
        Ensure all memories have vector embeddings.
        
        Generates embeddings for memories that don't have them yet.
        
        Args:
            agent_id: The agent ID
        
        Returns:
            Number of vectors generated
        """
        if not self.enable_vectors or not self.vector_store:
            return 0
        
        memories = self._load_agent_data(agent_id, "memories")
        return self.vector_store.add_vectors_batch(agent_id, memories)
    
    def get_vector_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get vector storage statistics for an agent."""
        if not self.enable_vectors or not self.vector_store:
            return {"error": "Vectors not enabled"}
        return self.vector_store.get_stats(agent_id)
    
    def get_memory_by_id(self, agent_id: str, memory_id: str) -> Optional[Dict]:
        """Get a specific memory by ID."""
        memories = self._load_agent_data(agent_id, "memories")
        for mem in memories:
            if mem.get("id") == memory_id or str(mem.get("id", "")).startswith(memory_id):
                return mem
        return None
    
    def update_memory(self, agent_id: str, memory_id: str, updates: Dict) -> bool:
        """Update a memory entry."""
        memories = self._load_agent_data(agent_id, "memories")
        
        for i, mem in enumerate(memories):
            if mem.get("id") == memory_id:
                # Update fields
                for key, value in updates.items():
                    if key != "id":
                        memories[i][key] = value
                memories[i]["updated_at"] = datetime.now().isoformat()
                
                self._save_agent_data(agent_id, "memories", memories)
                self._log_activity(agent_id, "mutation", "update", "memory", memory_id, "agent", agent_id)
                return True
        
        return False
    
    def delete_memory(self, agent_id: str, memory_id: str) -> bool:
        """Delete a memory entry."""
        memories = self._load_agent_data(agent_id, "memories")
        
        for i, mem in enumerate(memories):
            if mem.get("id") == memory_id:
                del memories[i]
                self._save_agent_data(agent_id, "memories", memories)
                self._log_activity(agent_id, "mutation", "delete", "memory", memory_id, "agent", agent_id)
                return True
        
        return False
    
    def list_memories(self, agent_id: str, memory_type: str = None,
                      limit: int = 50, offset: int = 0) -> List[Dict]:
        """List memories with optional filtering."""
        memories = self._load_agent_data(agent_id, "memories")
        
        # Filter by type
        if memory_type:
            memories = [m for m in memories if m.get("memory_type") == memory_type]
        
        # Sort by created_at descending
        memories.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return memories[offset:offset + limit]
    
    def count_memories(self, agent_id: str, memory_type: str = None) -> int:
        """Count memories."""
        memories = self._load_agent_data(agent_id, "memories")
        if memory_type:
            return len([m for m in memories if m.get("memory_type") == memory_type])
        return len(memories)
    
    # =========================================================================
    # Document Operations
    # =========================================================================
    
    def add_document(self, agent_id: str, filename: str, content: str,
                     folder: str = "knowledge", content_type: str = "markdown",
                     description: str = "", tags: List[str] = None,
                     metadata: Dict = None) -> str:
        """Add a document."""
        doc_id = str(uuid.uuid4())
        
        document = {
            "id": doc_id,
            "filename": filename,
            "content": content,
            "folder": folder,
            "content_type": content_type,
            "size_bytes": len(content.encode('utf-8')),
            "description": description,
            "tags": tags or [],
            "metadata": metadata or {},
            "agent_id": agent_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        documents = self._load_agent_data(agent_id, "documents")
        documents.append(document)
        self._save_agent_data(agent_id, "documents", documents)
        
        self._log_activity(agent_id, "mutation", "create", "document", doc_id, "user", agent_id)
        
        return doc_id
    
    def get_document_by_id(self, agent_id: str, doc_id: str) -> Optional[Dict]:
        """Get a document by ID."""
        documents = self._load_agent_data(agent_id, "documents")
        for doc in documents:
            if doc.get("id") == doc_id or str(doc.get("id", "")).startswith(doc_id):
                return doc
        return None
    
    def list_documents(self, agent_id: str, folder: str = None, limit: int = 50) -> List[Dict]:
        """List documents."""
        documents = self._load_agent_data(agent_id, "documents")
        if folder:
            documents = [d for d in documents if d.get("folder") == folder]
        documents.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return documents[:limit]
    
    def delete_document(self, agent_id: str, doc_id: str) -> bool:
        """Delete a document."""
        documents = self._load_agent_data(agent_id, "documents")
        for i, doc in enumerate(documents):
            if doc.get("id") == doc_id:
                del documents[i]
                self._save_agent_data(agent_id, "documents", documents)
                self._log_activity(agent_id, "mutation", "delete", "document", doc_id, "user", agent_id)
                return True
        return False
    
    def count_documents(self, agent_id: str, folder: str = None) -> int:
        """Count documents."""
        documents = self._load_agent_data(agent_id, "documents")
        if folder:
            return len([d for d in documents if d.get("folder") == folder])
        return len(documents)
    
    # =========================================================================
    # Checkpoint Operations
    # =========================================================================
    
    def create_checkpoint(self, agent_id: str, name: str, checkpoint_type: str = "snapshot",
                          description: str = "", metadata: Dict = None) -> str:
        """Create a checkpoint."""
        cp_id = str(uuid.uuid4())
        
        # Get current memory counts
        memory_counts = {
            "episodic": self.count_memories(agent_id, "episodic"),
            "semantic": self.count_memories(agent_id, "semantic"),
            "procedural": self.count_memories(agent_id, "procedural"),
            "working": self.count_memories(agent_id, "working"),
        }
        
        # Commit staged memories in DAG
        self.dag.set_agent(agent_id)
        commit_hash = None
        try:
            if self.dag.index:
                commit_hash = self.dag.commit(f"Checkpoint: {name}")
        except ValueError:
            pass  # Nothing staged
        
        checkpoint = {
            "id": cp_id,
            "agent_id": agent_id,
            "checkpoint_type": checkpoint_type,
            "name": name,
            "description": description,
            "commit_hash": commit_hash,
            "memory_counts": memory_counts,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        
        checkpoints = self._load_agent_data(agent_id, "checkpoints")
        checkpoints.append(checkpoint)
        self._save_agent_data(agent_id, "checkpoints", checkpoints)
        
        self._log_activity(agent_id, "mutation", "create", "checkpoint", cp_id, "agent", agent_id)
        
        return cp_id
    
    def get_checkpoint_by_id(self, agent_id: str, cp_id: str) -> Optional[Dict]:
        """Get a checkpoint by ID."""
        checkpoints = self._load_agent_data(agent_id, "checkpoints")
        for cp in checkpoints:
            if cp.get("id") == cp_id or str(cp.get("id", "")).startswith(cp_id):
                return cp
        return None
    
    def list_checkpoints(self, agent_id: str, checkpoint_type: str = None, limit: int = 20) -> List[Dict]:
        """List checkpoints."""
        checkpoints = self._load_agent_data(agent_id, "checkpoints")
        if checkpoint_type:
            checkpoints = [cp for cp in checkpoints if cp.get("checkpoint_type") == checkpoint_type]
        checkpoints.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return checkpoints[:limit]
    
    def delete_checkpoint(self, agent_id: str, cp_id: str) -> bool:
        """Delete a checkpoint."""
        checkpoints = self._load_agent_data(agent_id, "checkpoints")
        for i, cp in enumerate(checkpoints):
            if cp.get("id") == cp_id:
                del checkpoints[i]
                self._save_agent_data(agent_id, "checkpoints", checkpoints)
                return True
        return False
    
    def count_checkpoints(self, agent_id: str, checkpoint_type: str = None) -> int:
        """Count checkpoints."""
        checkpoints = self._load_agent_data(agent_id, "checkpoints")
        if checkpoint_type:
            return len([cp for cp in checkpoints if cp.get("checkpoint_type") == checkpoint_type])
        return len(checkpoints)
    
    # =========================================================================
    # Log Operations
    # =========================================================================
    
    def _log_activity(self, agent_id: str, log_type: str, action: str,
                      resource_type: str, resource_id: str,
                      actor_type: str, actor_id: str, details: Dict = None):
        """Internal method to log activity."""
        log_entry = {
            "id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "log_type": log_type,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "actor_type": actor_type,
            "actor_id": actor_id,
            "details": details or {},
            "created_at": datetime.now().isoformat()
        }
        
        logs = self._load_agent_data(agent_id, "logs")
        logs.append(log_entry)
        
        # Keep only last 1000 logs
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        self._save_agent_data(agent_id, "logs", logs)
    
    def get_log_by_id(self, agent_id: str, log_id: str) -> Optional[Dict]:
        """Get a log by ID."""
        logs = self._load_agent_data(agent_id, "logs")
        for log in logs:
            if log.get("id") == log_id or str(log.get("id", "")).startswith(log_id):
                return log
        return None
    
    def list_logs(self, agent_id: str, log_type: str = None, limit: int = 100) -> List[Dict]:
        """List activity logs."""
        logs = self._load_agent_data(agent_id, "logs")
        if log_type:
            logs = [log for log in logs if log.get("log_type") == log_type]
        logs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return logs[:limit]
    
    def delete_log(self, agent_id: str, log_id: str) -> bool:
        """Delete a log entry."""
        logs = self._load_agent_data(agent_id, "logs")
        for i, log in enumerate(logs):
            if log.get("id") == log_id:
                del logs[i]
                self._save_agent_data(agent_id, "logs", logs)
                return True
        return False
    
    def count_logs(self, agent_id: str, log_type: str = None) -> int:
        """Count logs."""
        logs = self._load_agent_data(agent_id, "logs")
        if log_type:
            return len([log for log in logs if log.get("log_type") == log_type])
        return len(logs)
    
    # =========================================================================
    # Agent Operations
    # =========================================================================
    
    def list_agents(self) -> List[str]:
        """List all known agents."""
        agents = []
        if self.agents_path.exists():
            for path in self.agents_path.iterdir():
                if path.is_dir():
                    agents.append(path.name)
        return agents
    
    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics for an agent."""
        return {
            "agent_id": agent_id,
            "memories": {
                "total": self.count_memories(agent_id),
                "episodic": self.count_memories(agent_id, "episodic"),
                "semantic": self.count_memories(agent_id, "semantic"),
                "procedural": self.count_memories(agent_id, "procedural"),
                "working": self.count_memories(agent_id, "working"),
            },
            "documents": self.count_documents(agent_id),
            "checkpoints": self.count_checkpoints(agent_id),
            "logs": self.count_logs(agent_id),
            "dag_history": len(self.dag.log(limit=100))
        }
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent and all its data."""
        import shutil
        agent_path = self._get_agent_path(agent_id)
        if agent_path.exists():
            shutil.rmtree(agent_path)
            return True
        return False
    
    # =========================================================================
    # Export/Import
    # =========================================================================
    
    def export_memories(self, agent_id: str) -> Dict[str, Any]:
        """Export all memories for backup."""
        return {
            "version": "1.0.0",
            "agent_id": agent_id,
            "exported_at": datetime.now().isoformat(),
            "memories": self._load_agent_data(agent_id, "memories"),
            "documents": self._load_agent_data(agent_id, "documents"),
            "checkpoints": self._load_agent_data(agent_id, "checkpoints")
        }
    
    def import_memories(self, agent_id: str, export_data: Dict, 
                        merge_mode: str = "append") -> Dict[str, Any]:
        """Import memories from backup."""
        if merge_mode == "replace":
            # Clear existing
            self._save_agent_data(agent_id, "memories", [])
            self._save_agent_data(agent_id, "documents", [])
        
        imported_count = 0
        
        # Import memories
        for mem in export_data.get("memories", []):
            mem["id"] = str(uuid.uuid4())  # New ID
            mem["imported_at"] = datetime.now().isoformat()
            memories = self._load_agent_data(agent_id, "memories")
            memories.append(mem)
            self._save_agent_data(agent_id, "memories", memories)
            imported_count += 1
        
        return {
            "status": "success",
            "imported": imported_count
        }
    
    # =========================================================================
    # Commit Operations (Version Control)
    # =========================================================================
    
    def commit_state(self, agent_id: str, message: str) -> Optional[str]:
        """Commit current state with a message."""
        self.dag.set_agent(agent_id)
        try:
            return self.dag.commit(message)
        except ValueError:
            return None
    
    def get_history(self, agent_id: str, limit: int = 10) -> List[Dict]:
        """Get commit history for an agent."""
        self.dag.set_agent(agent_id)
        return self.dag.log(limit=limit)
    
    def rollback(self, agent_id: str, commit_sha: str) -> Dict:
        """Rollback to a previous commit."""
        self.dag.set_agent(agent_id)
        return self.dag.checkout(commit_sha)
