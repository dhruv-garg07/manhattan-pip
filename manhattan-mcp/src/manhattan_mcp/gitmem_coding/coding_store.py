"""
GitMem Coding - Coding Context Store

Local JSON-based storage backend for coding context data.
Caches file contents read by AI agents for cross-session retrieval,
reducing token usage by avoiding redundant file reads.

Storage Structure:
    .gitmem_coding/
    ├── agents/
    │   └── {agent_id}/
    │       ├── file_contexts.json    # Cached file contents + metadata
    │       ├── coding_sessions.json  # Session read/retrieve logs
    │       └── settings.json         # Agent-specific config
    ├── index/
    │   └── file_index.json           # Path-based lookup index
    └── config.json                   # Global config
"""

import os
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import threading


from .models import FileContext, CodingSession, ContextStatus, TOKENS_PER_CHAR_RATIO, CodeChunk
from .chunking_engine import ChunkingEngine
from .ast_skeleton import ASTSkeletonGenerator, detect_language

# Staleness threshold — files not accessed in this many days are considered stale
STALE_DAYS_THRESHOLD = 30


class CodingContextStore:
    """
    Local JSON-based storage for coding context data.
    
    Thread-safe storage with automatic persistence.
    Provides hash-based freshness detection and semantic chunking.
    
    Features:
        - File content CRUD operations
        - Semantic Chunking & Deduplication
        - Hash-based staleness detection
        - Keyword search across cached files
        - Token savings tracking
        - Session-level analytics
    """
    
    def __init__(self, root_path: str = "./.gitmem_coding"):
        """
        Initialize the coding context store.
        
        Args:
            root_path: Root directory for storage (defaults to .gitmem_coding)
        """
        self.root_path = Path(root_path).absolute()
        self.agents_path = self.root_path / "agents"
        self.index_path = self.root_path / "index"
        self.chunks_path = self.root_path / "chunks.json" # Global chunk registry
        self.config_path = self.root_path / "config.json"
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize directories
        self._ensure_dirs()
        self._load_config()
    
    # =========================================================================
    # Internal Setup
    # =========================================================================
    
    def _ensure_dirs(self):
        """Create necessary directory structure."""
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.agents_path.mkdir(exist_ok=True)
        self.index_path.mkdir(exist_ok=True)
        
        # Initialize global chunk registry if missing
        if not self.chunks_path.exists():
            with open(self.chunks_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def _load_config(self):
        """Load global configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        else:
            self._config = {
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "stale_days_threshold": STALE_DAYS_THRESHOLD,
                "module": "gitmem_coding"
            }
            self._save_config()
    
    def _save_config(self):
        """Save global configuration."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2)
            
    def _load_chunks(self) -> Dict[str, Any]:
        """Load global chunk registry."""
        with self._lock:
            if self.chunks_path.exists():
                try:
                    with open(self.chunks_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    return {}
            return {}

    def _save_chunks(self, chunks: Dict[str, Any]):
        """Save global chunk registry."""
        with self._lock:
            with open(self.chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2)

    def _get_agent_path(self, agent_id: str) -> Path:
        """Get the storage path for an agent."""
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in agent_id)
        return self.agents_path / safe_id
    
    def _ensure_agent(self, agent_id: str):
        """Ensure agent directory and files exist."""
        agent_path = self._get_agent_path(agent_id)
        agent_path.mkdir(exist_ok=True)
        
        defaults = {
            "file_contexts.json": [],
            "coding_sessions.json": [],
            "settings.json": {"created_at": datetime.now().isoformat()}
        }
        
        for filename, default_content in defaults.items():
            filepath = agent_path / filename
            if not filepath.exists():
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(default_content, f, indent=2)
    
    def _load_agent_data(self, agent_id: str, data_type: str) -> list:
        """Load agent data from JSON file."""
        self._ensure_agent(agent_id)
        filepath = self._get_agent_path(agent_id) / f"{data_type}.json"
        
        with self._lock:
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        return []
    
    def _save_agent_data(self, agent_id: str, data_type: str, data: list):
        """Save agent data to JSON file."""
        self._ensure_agent(agent_id)
        filepath = self._get_agent_path(agent_id) / f"{data_type}.json"
        
        with self._lock:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
    
    # =========================================================================
    # File Context Operations
    # =========================================================================
    
    def store_file_context(
        self,
        agent_id: str,
        file_path: str,
        content: str,
        language: str = "other",
        session_id: str = "",
        keywords: List[str] = None,
        content_summary: str = "",
        storage_mode: str = "skeleton"
    ) -> Dict[str, Any]:
        """
        Store or update file content in the coding context cache.
        
        In 'skeleton' mode (default), generates a compact AST skeleton and
        stores ONLY the skeleton — NOT the full content. This achieves
        70-85% token reduction while preserving structural information.
        
        In 'full' mode, stores the raw content (legacy behavior).
        
        The original content hash is always stored for freshness detection.
        
        Args:
            agent_id: The agent ID
            file_path: Absolute path to the file
            content: Full file content
            language: Programming language
            session_id: Current session identifier
            keywords: Optional searchable keywords
            content_summary: Optional brief description
            storage_mode: 'skeleton' (compact AST, default) or 'full' (raw content)
        
        Returns:
            Dict with store status, context_id, compression stats
        """
        content_hash = FileContext.compute_hash(content)
        file_name = os.path.basename(file_path)
        normalized_path = os.path.normpath(file_path)
        original_token_estimate = FileContext.estimate_tokens(content)
        
        # ── Generate AST skeleton if in skeleton mode ──
        compact_skeleton = ""
        skeleton_token_estimate = 0
        compression_ratio = 0.0
        
        if storage_mode == "skeleton":
            skeleton_gen = ASTSkeletonGenerator()
            compact_skeleton = skeleton_gen.generate_skeleton(content, language, file_path)
            skeleton_token_estimate = FileContext.estimate_tokens(compact_skeleton)
            if original_token_estimate > 0:
                compression_ratio = round(
                    1.0 - (skeleton_token_estimate / original_token_estimate), 4
                )
        
        # Get file modification time if accessible
        file_mod_time = ""
        try:
            if os.path.exists(normalized_path):
                file_mod_time = datetime.fromtimestamp(
                    os.path.getmtime(normalized_path)
                ).isoformat()
        except (OSError, ValueError):
            pass
        
        # --- Chunking & Deduplication ---
        chunker = ChunkingEngine.get_chunker(language)
        chunks = chunker.chunk_file(content, file_path)
        
        chunk_registry = self._load_chunks()
        new_chunks_count = 0
        file_chunk_hashes = []
        
        for chunk in chunks:
            file_chunk_hashes.append(chunk.hash_id)
            if chunk.hash_id not in chunk_registry:
                chunk_registry[chunk.hash_id] = chunk.to_dict()
                new_chunks_count += 1
        
        if new_chunks_count > 0:
            self._save_chunks(chunk_registry)
        
        # ── Determine what to persist ──
        stored_content = content if storage_mode == "full" else ""
        effective_token_estimate = (
            original_token_estimate if storage_mode == "full" 
            else skeleton_token_estimate
        )
        
        # --- File Context Update ---
        contexts = self._load_agent_data(agent_id, "file_contexts")
        
        existing_idx = None
        for i, ctx in enumerate(contexts):
            if os.path.normpath(ctx.get("file_path", "")) == normalized_path:
                existing_idx = i
                break
        
        now = datetime.now().isoformat()
        
        if existing_idx is not None:
            existing = contexts[existing_idx]
            was_same_content = existing.get("content_hash") == content_hash
            
            existing["content"] = stored_content
            existing["content_hash"] = content_hash
            existing["compact_skeleton"] = compact_skeleton
            existing["storage_mode"] = storage_mode
            existing["size_bytes"] = len(content.encode('utf-8'))
            existing["line_count"] = content.count('\n') + 1
            existing["token_estimate"] = effective_token_estimate
            existing["original_token_estimate"] = original_token_estimate
            existing["skeleton_token_estimate"] = skeleton_token_estimate
            existing["compression_ratio"] = compression_ratio
            existing["last_accessed_at"] = now
            existing["access_count"] = existing.get("access_count", 0) + 1
            existing["file_modified_at"] = file_mod_time
            existing["language"] = language
            existing["session_id"] = session_id
            existing["chunk_hashes"] = file_chunk_hashes
            
            if content_summary:
                existing["content_summary"] = content_summary
            if keywords:
                old_kw = set(existing.get("keywords", []))
                old_kw.update(keywords)
                existing["keywords"] = list(old_kw)
            
            contexts[existing_idx] = existing
            self._save_agent_data(agent_id, "file_contexts", contexts)
            
            return {
                "status": "updated",
                "context_id": existing["id"],
                "file_path": normalized_path,
                "content_changed": not was_same_content,
                "storage_mode": storage_mode,
                "token_estimate": effective_token_estimate,
                "original_token_estimate": original_token_estimate,
                "skeleton_token_estimate": skeleton_token_estimate,
                "compression_ratio": compression_ratio,
                "tokens_saved": original_token_estimate - effective_token_estimate,
                "chunks_processed": len(chunks),
                "new_unique_chunks": new_chunks_count,
                "message": f"File context updated ({storage_mode} mode)."
            }
        else:
            context_id = str(uuid.uuid4())
            
            new_context = {
                "id": context_id,
                "file_path": normalized_path,
                "file_name": file_name,
                "relative_path": "",
                "content": stored_content,
                "content_hash": content_hash,
                "content_summary": content_summary,
                "compact_skeleton": compact_skeleton,
                "storage_mode": storage_mode,
                "language": language,
                "line_count": content.count('\n') + 1,
                "size_bytes": len(content.encode('utf-8')),
                "agent_id": agent_id,
                "session_id": session_id,
                "access_count": 1,
                "token_estimate": effective_token_estimate,
                "original_token_estimate": original_token_estimate,
                "skeleton_token_estimate": skeleton_token_estimate,
                "compression_ratio": compression_ratio,
                "created_at": now,
                "last_accessed_at": now,
                "file_modified_at": file_mod_time,
                "keywords": keywords or [],
                "tags": [],
                "chunk_hashes": file_chunk_hashes
            }
            
            contexts.append(new_context)
            self._save_agent_data(agent_id, "file_contexts", contexts)
            
            return {
                "status": "created",
                "context_id": context_id,
                "file_path": normalized_path,
                "storage_mode": storage_mode,
                "token_estimate": effective_token_estimate,
                "original_token_estimate": original_token_estimate,
                "skeleton_token_estimate": skeleton_token_estimate,
                "compression_ratio": compression_ratio,
                "tokens_saved": original_token_estimate - effective_token_estimate,
                "chunks_processed": len(chunks),
                "new_unique_chunks": new_chunks_count,
                "message": f"File context stored ({storage_mode} mode)."
            }
    
    def store_file_context_from_path(
        self,
        agent_id: str,
        file_path: str,
        language: str = "auto",
        session_id: str = "",
        keywords: List[str] = None,
        content_summary: str = "",
        storage_mode: str = "skeleton"
    ) -> Dict[str, Any]:
        """
        Store a file by reading it from disk — no content parameter needed.
        
        Reads the file server-side, auto-detects language, generates AST
        skeleton, and stores. This avoids sending full file content through
        the LLM context window.
        
        Args:
            agent_id: The agent ID
            file_path: Absolute path to the file on disk
            language: Programming language (or 'auto' to detect from extension)
            session_id: Current session identifier
            keywords: Optional searchable keywords
            content_summary: Optional brief description
            storage_mode: 'skeleton' (compact AST, default) or 'full'
        
        Returns:
            Dict with store status, context_id, compression stats
        """
        normalized_path = os.path.normpath(file_path)
        
        if not os.path.exists(normalized_path):
            return {
                "status": "error",
                "error": f"File not found: {normalized_path}",
                "file_path": normalized_path
            }
        
        try:
            with open(normalized_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except (OSError, IOError) as e:
            return {
                "status": "error",
                "error": f"Failed to read file: {e}",
                "file_path": normalized_path
            }
        
        # Auto-detect language from extension
        if language == "auto":
            language = detect_language(file_path)
        
        return self.store_file_context(
            agent_id=agent_id,
            file_path=file_path,
            content=content,
            language=language,
            session_id=session_id,
            keywords=keywords,
            content_summary=content_summary,
            storage_mode=storage_mode
        )
    
    def store_file_context_from_ast(
        self,
        agent_id: str,
        file_path: str,
        context_ast: str,
        language: str = "auto",
        session_id: str = "",
        keywords: List[str] = None,
        content_summary: str = ""
    ) -> Dict[str, Any]:
        """
        Store a file using a pre-computed AST skeleton provided by the LLM/agent.
        
        The agent provides the AST skeleton directly (e.g. from its own analysis).
        This method still reads the file from disk to compute the content hash
        (for freshness detection) and metadata (size, line count), but does NOT
        regenerate the skeleton — it uses the one provided.
        
        Args:
            agent_id: The agent ID
            file_path: Absolute path to the file on disk
            context_ast: The AST skeleton string provided by the LLM/coding agent
            language: Programming language (or 'auto' to detect from extension)
            session_id: Current session identifier
            keywords: Optional searchable keywords
            content_summary: Optional brief description
        
        Returns:
            Dict with store status, context_id, token stats
        """
        normalized_path = os.path.normpath(file_path)
        
        # Read file from disk for hash and metadata only
        if not os.path.exists(normalized_path):
            return {
                "status": "error",
                "error": f"File not found: {normalized_path}",
                "file_path": normalized_path
            }
        
        try:
            with open(normalized_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except (OSError, IOError) as e:
            return {
                "status": "error",
                "error": f"Failed to read file: {e}",
                "file_path": normalized_path
            }
        
        # Auto-detect language from extension
        if language == "auto":
            language = detect_language(file_path)
        
        # Compute metadata from original content
        content_hash = FileContext.compute_hash(content)
        file_name = os.path.basename(file_path)
        original_token_estimate = FileContext.estimate_tokens(content)
        skeleton_token_estimate = FileContext.estimate_tokens(context_ast)
        compression_ratio = 0.0
        if original_token_estimate > 0:
            compression_ratio = round(
                1.0 - (skeleton_token_estimate / original_token_estimate), 4
            )
        
        # Get file modification time
        file_mod_time = ""
        try:
            file_mod_time = datetime.fromtimestamp(
                os.path.getmtime(normalized_path)
            ).isoformat()
        except (OSError, ValueError):
            pass
        
        # Chunking from original content
        chunker = ChunkingEngine.get_chunker(language)
        chunks = chunker.chunk_file(content, file_path)
        
        chunk_registry = self._load_chunks()
        new_chunks_count = 0
        file_chunk_hashes = []
        
        for chunk in chunks:
            file_chunk_hashes.append(chunk.hash_id)
            if chunk.hash_id not in chunk_registry:
                chunk_registry[chunk.hash_id] = chunk.to_dict()
                new_chunks_count += 1
        
        if new_chunks_count > 0:
            self._save_chunks(chunk_registry)
        
        effective_token_estimate = skeleton_token_estimate
        
        # --- File Context Update ---
        contexts = self._load_agent_data(agent_id, "file_contexts")
        
        existing_idx = None
        for i, ctx in enumerate(contexts):
            if os.path.normpath(ctx.get("file_path", "")) == normalized_path:
                existing_idx = i
                break
        
        now = datetime.now().isoformat()
        
        if existing_idx is not None:
            existing = contexts[existing_idx]
            was_same_content = existing.get("content_hash") == content_hash
            
            existing["content"] = ""  # AST mode — no full content stored
            existing["content_hash"] = content_hash
            existing["compact_skeleton"] = context_ast
            existing["storage_mode"] = "ast"
            existing["size_bytes"] = len(content.encode('utf-8'))
            existing["line_count"] = content.count('\n') + 1
            existing["token_estimate"] = effective_token_estimate
            existing["original_token_estimate"] = original_token_estimate
            existing["skeleton_token_estimate"] = skeleton_token_estimate
            existing["compression_ratio"] = compression_ratio
            existing["last_accessed_at"] = now
            existing["access_count"] = existing.get("access_count", 0) + 1
            existing["file_modified_at"] = file_mod_time
            existing["language"] = language
            existing["session_id"] = session_id
            existing["chunk_hashes"] = file_chunk_hashes
            
            if content_summary:
                existing["content_summary"] = content_summary
            if keywords:
                old_kw = set(existing.get("keywords", []))
                old_kw.update(keywords)
                existing["keywords"] = list(old_kw)
            
            contexts[existing_idx] = existing
            self._save_agent_data(agent_id, "file_contexts", contexts)
            
            return {
                "status": "updated",
                "context_id": existing["id"],
                "file_path": normalized_path,
                "content_changed": not was_same_content,
                "storage_mode": "ast",
                "token_estimate": effective_token_estimate,
                "original_token_estimate": original_token_estimate,
                "skeleton_token_estimate": skeleton_token_estimate,
                "compression_ratio": compression_ratio,
                "tokens_saved": original_token_estimate - effective_token_estimate,
                "chunks_processed": len(chunks),
                "new_unique_chunks": new_chunks_count,
                "message": "File context updated with agent-provided AST."
            }
        else:
            context_id = str(uuid.uuid4())
            
            new_context = {
                "id": context_id,
                "file_path": normalized_path,
                "file_name": file_name,
                "relative_path": "",
                "content": "",
                "content_hash": content_hash,
                "content_summary": content_summary,
                "compact_skeleton": context_ast,
                "storage_mode": "ast",
                "language": language,
                "line_count": content.count('\n') + 1,
                "size_bytes": len(content.encode('utf-8')),
                "agent_id": agent_id,
                "session_id": session_id,
                "access_count": 1,
                "token_estimate": effective_token_estimate,
                "original_token_estimate": original_token_estimate,
                "skeleton_token_estimate": skeleton_token_estimate,
                "compression_ratio": compression_ratio,
                "created_at": now,
                "last_accessed_at": now,
                "file_modified_at": file_mod_time,
                "keywords": keywords or [],
                "tags": [],
                "chunk_hashes": file_chunk_hashes
            }
            
            contexts.append(new_context)
            self._save_agent_data(agent_id, "file_contexts", contexts)
            
            return {
                "status": "created",
                "context_id": context_id,
                "file_path": normalized_path,
                "storage_mode": "ast",
                "token_estimate": effective_token_estimate,
                "original_token_estimate": original_token_estimate,
                "skeleton_token_estimate": skeleton_token_estimate,
                "compression_ratio": compression_ratio,
                "tokens_saved": original_token_estimate - effective_token_estimate,
                "chunks_processed": len(chunks),
                "new_unique_chunks": new_chunks_count,
                "message": "File context stored with agent-provided AST."
            }
    
    def retrieve_file_context(
        self,
        agent_id: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Retrieve cached file content with freshness check.
        
        Compares stored content hash with current file hash to determine
        if the cached version is still fresh or has become stale.
        
        Args:
            agent_id: The agent ID
            file_path: Path to the file to retrieve
        
        Returns:
            Dict with content, status (fresh/stale/missing), and token savings info
        """
        normalized_path = os.path.normpath(file_path)
        contexts = self._load_agent_data(agent_id, "file_contexts")
        
        # Find matching entry
        found = None
        found_idx = None
        for i, ctx in enumerate(contexts):
            if os.path.normpath(ctx.get("file_path", "")) == normalized_path:
                found = ctx
                found_idx = i
                break
        
        if found is None:
            return {
                "status": "cache_miss",
                "file_path": normalized_path,
                "content": None,
                "message": "File not found in coding context cache.",
                "token_savings": 0
            }
        
        # Determine freshness
        freshness_status = ContextStatus.UNKNOWN.value
        current_hash = None
        
        try:
            if os.path.exists(normalized_path):
                with open(normalized_path, 'r', encoding='utf-8', errors='replace') as f:
                    current_content = f.read()
                current_hash = FileContext.compute_hash(current_content)
                
                if current_hash == found.get("content_hash"):
                    freshness_status = ContextStatus.FRESH.value
                else:
                    freshness_status = ContextStatus.STALE.value
            else:
                freshness_status = ContextStatus.MISSING.value
        except (OSError, UnicodeDecodeError):
            freshness_status = ContextStatus.UNKNOWN.value
        
        # Update access metadata
        now = datetime.now().isoformat()
        found["last_accessed_at"] = now
        found["access_count"] = found.get("access_count", 0) + 1
        contexts[found_idx] = found
        self._save_agent_data(agent_id, "file_contexts", contexts)
        
        token_estimate = found.get("token_estimate", 0)
        
        # Determine what to return based on storage_mode
        storage_mode = found.get("storage_mode", "full")
        compact_skeleton = found.get("compact_skeleton", "")
        original_token_estimate = found.get("original_token_estimate", token_estimate)
        skeleton_token_estimate = found.get("skeleton_token_estimate", 0)
        compression_ratio = found.get("compression_ratio", 0.0)
        
        # Primary content: skeleton if available, else full content
        primary_content = compact_skeleton if compact_skeleton else found.get("content", "")
        
        # Token savings: difference between full file tokens and what we return
        effective_savings = (
            original_token_estimate - skeleton_token_estimate
            if freshness_status == ContextStatus.FRESH.value and compact_skeleton
            else 0
        )
        
        return {
            "status": "cache_hit",
            "freshness": freshness_status,
            "file_path": normalized_path,
            "file_name": found.get("file_name", ""),
            "content": primary_content,
            "compact_skeleton": compact_skeleton,
            "content_hash": found.get("content_hash", ""),
            "storage_mode": storage_mode,
            "language": found.get("language", "other"),
            "line_count": found.get("line_count", 0),
            "size_bytes": found.get("size_bytes", 0),
            "token_estimate": token_estimate,
            "original_token_estimate": original_token_estimate,
            "skeleton_token_estimate": skeleton_token_estimate,
            "compression_ratio": compression_ratio,
            "token_savings": effective_savings,
            "access_count": found.get("access_count", 0),
            "created_at": found.get("created_at", ""),
            "last_accessed_at": now,
            "content_summary": found.get("content_summary", ""),
            "chunk_hashes": found.get("chunk_hashes", []),
            "message": f"File retrieved from cache ({storage_mode} mode). Status: {freshness_status}."
        }
    
    def search_contexts(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search across stored file contexts by path, filename, or keywords.
        """
        return self.search_contexts(agent_id, query, top_k)

    def _get_preview_snippet(self, content: str, query_words: List[str], max_chars: int = 2000) -> str:
        """Get a snippet of content centered around the first match."""
        if not content or len(content) <= max_chars:
            return content
        
        lower_content = content.lower()
        best_pos = -1
        
        # Find first occurrence of any query word
        for word in query_words:
            if len(word) < 3: continue # Skip short words
            pos = lower_content.find(word)
            if pos != -1:
                best_pos = pos
                break
        
        if best_pos == -1:
            return content[:max_chars] + "..."
            
        # Center the window around the match
        half_window = max_chars // 2
        start = max(0, best_pos - half_window)
        end = min(len(content), start + max_chars)
        
        # Adjust start if end hit the boundary
        if end == len(content):
            start = max(0, end - max_chars)
            
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
            
        return snippet

    def search_contexts(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Search for file contexts relevant to the query.
        Searches metadata, summaries, and skeleton content.
        """
        contexts = self._load_agent_data(agent_id, "file_contexts")
        if not query or not contexts:
            return {"results": [], "total_count": 0}
            
        query_words = query.lower().split()
        results = []
        
        for ctx in contexts:
            score = 0
            file_path = ctx.get("file_path", "").lower()
            file_name = ctx.get("file_name", "").lower()
            language = ctx.get("language", "").lower()
            keywords = [k.lower() for k in ctx.get("keywords", [])]
            summary = ctx.get("content_summary", "").lower()
            skeleton = ctx.get("compact_skeleton", "") # Keep original case for preview
            content = ctx.get("content", "") # Keep original case for preview
            
            # Use lower case for searching
            skeleton_search = skeleton.lower()
            content_search = content.lower()
            
            for word in query_words:
                # Filename match (highest priority)
                if word in file_name:
                    score += 3.0
                
                # File path match
                if word in file_path:
                    score += 1.5
                
                # Language match
                if word == language:
                    score += 2.0
                
                # Keyword match
                for kw in keywords:
                    if word in kw or kw in word:
                        score += 1.0
                        break
                
                # Summary match
                if word in summary:
                    score += 0.5
                
                # Skeleton/Content match (lower priority but enables finding functions)
                if word in skeleton_search:
                    score += 1.0
                elif word in content_search:
                    score += 1.0
            
            if score > 0:
                # Determine raw preview source
                raw_preview = skeleton if skeleton else content
                
                # Get windowed snippet around the match
                preview_snippet = self._get_preview_snippet(raw_preview, query_words)

                # Return summary and skeleton preview for efficiency
                result = {
                    "context_id": ctx.get("id"), # Changed from "context_id" to "id" to match new_context
                    "file_path": ctx.get("file_path"),
                    "file_name": ctx.get("file_name"),
                    "score": score,
                    "summary": summary,
                    "preview": preview_snippet, # Contextual snippet
                    "language": ctx.get("language"),
                    "line_count": ctx.get("line_count", 0),
                    "size_bytes": ctx.get("size_bytes", 0),
                    "token_estimate": ctx.get("token_estimate", 0),
                    "access_count": ctx.get("access_count", 0),
                    "created_at": ctx.get("created_at"),
                    "last_accessed_at": ctx.get("last_accessed_at"),
                    "content_summary": ctx.get("content_summary", ""),
                    "keywords": ctx.get("keywords", []),
                    "chunk_hashes": ctx.get("chunk_hashes", []), # Add to search result
                    "score": round(score, 3)
                }
                results.append(result)
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def list_contexts(
        self,
        agent_id: str,
        limit: int = 50,
        offset: int = 0,
        language: str = None
    ) -> Dict[str, Any]:
        """
        List all cached file contexts for an agent.
        
        Returns summaries (no full content) for efficiency.
        
        Args:
            agent_id: The agent ID
            limit: Maximum items to return
            offset: Pagination offset
            language: Optional filter by language
        
        Returns:
            Dict with paginated context summaries
        """
        contexts = self._load_agent_data(agent_id, "file_contexts")
        
        if language:
            contexts = [c for c in contexts if c.get("language") == language]
        
        # Sort by last_accessed_at descending (most recent first)
        contexts.sort(key=lambda x: x.get("last_accessed_at", ""), reverse=True)
        
        page = contexts[offset:offset + limit]
        
        # Strip content for listing
        summaries = []
        for ctx in page:
            summary = {k: v for k, v in ctx.items() if k != "content"}
            summary["has_content"] = bool(ctx.get("content"))
            summaries.append(summary)
        
        return {
            "contexts": summaries,
            "total": len(contexts),
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < len(contexts)
        }
    
    def delete_context(
        self,
        agent_id: str,
        file_path: str = None,
        context_id: str = None
    ) -> Dict[str, Any]:
        """
        Delete a cached file context.
        
        Can delete by file_path or context_id.
        
        Args:
            agent_id: The agent ID
            file_path: Path of file to remove from cache
            context_id: ID of context entry to remove
        
        Returns:
            Dict with deletion status
        """
        contexts = self._load_agent_data(agent_id, "file_contexts")
        original_count = len(contexts)
        
        if file_path:
            normalized_path = os.path.normpath(file_path)
            contexts = [
                c for c in contexts 
                if os.path.normpath(c.get("file_path", "")) != normalized_path
            ]
        elif context_id:
            contexts = [c for c in contexts if c.get("id") != context_id]
        else:
            return {"status": "error", "message": "Provide file_path or context_id."}
        
        deleted_count = original_count - len(contexts)
        self._save_agent_data(agent_id, "file_contexts", contexts)
        
        return {
            "status": "ok",
            "deleted_count": deleted_count,
            "remaining_count": len(contexts)
        }
    
    def get_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        Get coding context storage statistics for an agent.
        
        Args:
            agent_id: The agent ID
        
        Returns:
            Dict with comprehensive statistics
        """
        contexts = self._load_agent_data(agent_id, "file_contexts")
        sessions = self._load_agent_data(agent_id, "coding_sessions")
        chunks = self._load_chunks()
        
        total_size = sum(c.get("size_bytes", 0) for c in contexts)
        total_tokens = sum(c.get("token_estimate", 0) for c in contexts)
        total_accesses = sum(c.get("access_count", 0) for c in contexts)
        
        # Count by language
        lang_counts = {}
        for ctx in contexts:
            lang = ctx.get("language", "other")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        # Estimate total token savings (access_count - 1 for each = times served from cache)
        total_cache_serves = sum(max(0, c.get("access_count", 0) - 1) for c in contexts)
        avg_tokens_per_file = total_tokens / len(contexts) if contexts else 0
        estimated_token_savings = int(total_cache_serves * avg_tokens_per_file)
        
        # Freshness stats
        fresh_count = 0
        stale_count = 0
        missing_count = 0
        for ctx in contexts:
            try:
                fp = ctx.get("file_path", "")
                if os.path.exists(fp):
                    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                        current_hash = FileContext.compute_hash(f.read())
                    if current_hash == ctx.get("content_hash"):
                        fresh_count += 1
                    else:
                        stale_count += 1
                else:
                    missing_count += 1
            except (OSError, UnicodeDecodeError):
                pass  # Skip files we can't check
        
        return {
            "agent_id": agent_id,
            "total_files_cached": len(contexts),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_token_estimate": total_tokens,
            "total_accesses": total_accesses,
            "estimated_token_savings": estimated_token_savings,
            "freshness": {
                "fresh": fresh_count,
                "stale": stale_count,
                "missing": missing_count,
                "unchecked": len(contexts) - fresh_count - stale_count - missing_count
            },
            "languages": lang_counts,
            "sessions_recorded": len(sessions),
            "global_unique_chunks": len(chunks)
        }
    
    # =========================================================================
    # Session Tracking
    # =========================================================================
    
    def record_session_activity(
        self,
        agent_id: str,
        session_id: str,
        action: str,
        file_path: str,
        tokens: int = 0
    ):
        """
        Record a file store/retrieve action within a session.
        
        Args:
            agent_id: The agent ID
            session_id: Current session ID
            action: "store" or "retrieve"
            file_path: File path involved
            tokens: Token count for the operation
        """
        sessions = self._load_agent_data(agent_id, "coding_sessions")
        
        # Find or create session
        session = None
        session_idx = None
        for i, s in enumerate(sessions):
            if s.get("session_id") == session_id:
                session = s
                session_idx = i
                break
        
        if session is None:
            session = {
                "session_id": session_id,
                "agent_id": agent_id,
                "files_stored": [],
                "files_retrieved": [],
                "tokens_stored": 0,
                "tokens_retrieved": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "started_at": datetime.now().isoformat(),
                "ended_at": ""
            }
            sessions.append(session)
            session_idx = len(sessions) - 1
        
        normalized_path = os.path.normpath(file_path)
        
        if action == "store":
            if normalized_path not in session["files_stored"]:
                session["files_stored"].append(normalized_path)
            session["tokens_stored"] += tokens
        elif action == "retrieve":
            if normalized_path not in session["files_retrieved"]:
                session["files_retrieved"].append(normalized_path)
            session["tokens_retrieved"] += tokens
            session["cache_hits"] += 1
        elif action == "miss":
            session["cache_misses"] += 1
        
        sessions[session_idx] = session
        
        # Keep only last 100 sessions
        if len(sessions) > 100:
            sessions = sessions[-100:]
        
        self._save_agent_data(agent_id, "coding_sessions", sessions)
    
    def get_token_savings_report(self, agent_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive token savings report.
        
        Args:
            agent_id: The agent ID
        
        Returns:
            Dict with token savings breakdown
        """
        sessions = self._load_agent_data(agent_id, "coding_sessions")
        contexts = self._load_agent_data(agent_id, "file_contexts")
        
        total_tokens_stored = sum(s.get("tokens_stored", 0) for s in sessions)
        total_tokens_retrieved = sum(s.get("tokens_retrieved", 0) for s in sessions)
        total_cache_hits = sum(s.get("cache_hits", 0) for s in sessions)
        total_cache_misses = sum(s.get("cache_misses", 0) for s in sessions)
        
        # Current cache value
        cache_token_value = sum(c.get("token_estimate", 0) for c in contexts)
        
        return {
            "total_tokens_stored": total_tokens_stored,
            "total_tokens_retrieved_from_cache": total_tokens_retrieved,
            "estimated_token_savings": total_tokens_retrieved,
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses,
            "hit_rate": round(
                (total_cache_hits / (total_cache_hits + total_cache_misses) * 100)
                if (total_cache_hits + total_cache_misses) > 0 else 0, 2
            ),
            "current_cache_token_value": cache_token_value,
            "files_in_cache": len(contexts),
            "sessions_tracked": len(sessions)
        }
    
    # =========================================================================
    # Cleanup Operations
    # =========================================================================
    
    def cleanup_stale_contexts(
        self,
        agent_id: str,
        days_threshold: int = None
    ) -> Dict[str, Any]:
        """
        Remove cached contexts not accessed within the threshold period.
        
        Args:
            agent_id: The agent ID
            days_threshold: Days since last access to consider stale
        
        Returns:
            Dict with cleanup results
        """
        threshold = days_threshold or self._config.get("stale_days_threshold", STALE_DAYS_THRESHOLD)
        cutoff = (datetime.now() - timedelta(days=threshold)).isoformat()
        
        contexts = self._load_agent_data(agent_id, "file_contexts")
        original_count = len(contexts)
        
        # Keep contexts accessed after the cutoff
        fresh = [c for c in contexts if c.get("last_accessed_at", "") >= cutoff]
        removed_count = original_count - len(fresh)
        
        if removed_count > 0:
            self._save_agent_data(agent_id, "file_contexts", fresh)
        
        return {
            "status": "ok",
            "removed_count": removed_count,
            "remaining_count": len(fresh),
            "threshold_days": threshold
        }
    
    # =========================================================================
    # Agent Operations
    # =========================================================================
    
    def list_agents(self) -> List[str]:
        """List all agents with coding context data."""
        agents = []
        if self.agents_path.exists():
            for path in self.agents_path.iterdir():
                if path.is_dir():
                    agents.append(path.name)
        return agents
    
    def delete_agent_data(self, agent_id: str) -> bool:
        """Delete all coding context data for an agent."""
        import shutil
        agent_path = self._get_agent_path(agent_id)
        if agent_path.exists():
            shutil.rmtree(agent_path)
            return True
        return False

