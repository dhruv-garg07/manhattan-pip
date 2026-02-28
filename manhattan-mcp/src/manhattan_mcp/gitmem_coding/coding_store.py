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


from .chunking_engine import ChunkingEngine
from .ast_skeleton import ContextTreeBuilder, BSTIndex

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
        self.global_index_path = self.index_path / "global_index.json"
        
        self._load_config()
        self._global_index = self._load_global_index()
    
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

    def _load_global_index(self) -> Dict[str, List[str]]:
        """Load global inverted index (Symbol -> [FilePaths])."""
        with self._lock:
            if self.global_index_path.exists():
                try:
                    with open(self.global_index_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    return {}
            return {}

    def _save_global_index(self):
        """Save global inverted index."""
        with self._lock:
            with open(self.global_index_path, 'w', encoding='utf-8') as f:
                json.dump(self._global_index, f, indent=2)

    def _update_global_index(self, file_path: str, symbols: List[str]):
        """
        Update global index for a file.
        1. Remove file from all entries (cleanup old).
        2. Add file to new symbol entries.
        """
        normalized_path = os.path.normpath(file_path)
        
        # 1. Cleanup old
        self._remove_from_global_index(normalized_path, save=False)
        
        # 2. Add new
        for symbol in symbols:
            if symbol not in self._global_index:
                self._global_index[symbol] = []
            if normalized_path not in self._global_index[symbol]:
                self._global_index[symbol].append(normalized_path)
        
        self._save_global_index()

    def _remove_from_global_index(self, file_path: str, save: bool = True):
        """Remove a file from the global index."""
        normalized_path = os.path.normpath(file_path)
        empty_keys = []
        
        for symbol, files in self._global_index.items():
            if normalized_path in files:
                files.remove(normalized_path)
                if not files:
                    empty_keys.append(symbol)
        
        for k in empty_keys:
            del self._global_index[k]
            
        if save:
            self._save_global_index()

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
    
    def store_file_chunks(
        self,
        agent_id: str,
        file_path: str,
        chunks: List[Dict[str, Any]],
        language: str = "auto",
        session_id: str = ""
    ) -> Dict[str, Any]:
        """
        Store file context from pre-computed chunks.
        """
        normalized_path = os.path.normpath(file_path)
        file_name = os.path.basename(file_path)
        
        # 1. Build Tree & Index from chunks
        builder = ContextTreeBuilder()
        flow_data = builder.build(chunks, file_path)
        compact_skeleton = json.dumps(flow_data)
        
        # 2. Update Global Index
        if "index" in flow_data:
            # Index is now a flat Dict[Symbol, List[NodeIDs]]
            all_symbols = list(flow_data["index"].keys())
            self._update_global_index(normalized_path, all_symbols)
            
        # 3. Process Chunks for Storage
        # We store the full chunks in the file context
        # We also might want to update the global chunk registry if we were using it for deduplication across files
        # But for now, let's strictly follow the file context structure
        
        now = datetime.now().isoformat()
        
        # Calculate stats
        total_tokens = sum(c.get("token_count", 0) for c in chunks)
        content_summary = chunks[0].get("summary", "") if chunks else ""
        
        contexts = self._load_agent_data(agent_id, "file_contexts")
        existing_idx = next((i for i, c in enumerate(contexts) if os.path.normpath(c.get("file_path", "")) == normalized_path), None)
        
        # Compute content hash for freshness detection
        file_content_hash = ""
        try:
            if os.path.exists(normalized_path):
                with open(normalized_path, 'r', encoding='utf-8', errors='replace') as f:
                    file_content = f.read()
                file_content_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()
        except Exception:
            pass

        context_data = {
            "file_path": normalized_path,
            "file_name": file_name,
            "compact_skeleton": compact_skeleton,
            "chunks": chunks, # Store full chunks
            "language": language,
            "agent_id": agent_id,
            "session_id": session_id,
            "last_accessed_at": now,
            "token_estimate": total_tokens,
            "file_modified_at": now, # Approximation
            "content_hash": file_content_hash,  # SHA-256 for freshness checks
        }

        if existing_idx is not None:
            # Update existing
            existing = contexts[existing_idx]
            existing.update(context_data)
            existing["access_count"] = existing.get("access_count", 0) + 1
            contexts[existing_idx] = existing
            status = "updated"
            ctx_id = existing["id"]
        else:
            # Create new
            ctx_id = str(uuid.uuid4())
            context_data["id"] = ctx_id
            context_data["created_at"] = now
            context_data["access_count"] = 1
            contexts.append(context_data)
            status = "created"
            
        self._save_agent_data(agent_id, "file_contexts", contexts)
        
        return {
            "file_path": normalized_path,
            "message": f"File chunks stored. Count: {len(chunks)}"
        }

    def get_cached_chunk(self, hash_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chunk from the global registry by hash."""
        chunks = self._load_chunks()
        return chunks.get(hash_id)

    def cache_chunk(self, chunk_data: Dict[str, Any]):
        """Cache a chunk in the global registry (without inline vectors)."""
        hash_id = chunk_data.get("hash_id")
        if not hash_id:
            return
        
        # Strip inline vector — vectors live in vectors.json
        clean_chunk = {k: v for k, v in chunk_data.items() if k != "vector"}
        
        with self._lock:
            chunks = self._load_chunks()
            if hash_id not in chunks:
                chunks[hash_id] = clean_chunk
                self._save_chunks(chunks)



    def retrieve_file_context(
        self,
        agent_id: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Retrieve cached file content (Code Flow Tree) with freshness check.
        Updates access metrics.
        """
        from datetime import datetime
        normalized_path = os.path.normpath(file_path)
        contexts = self._load_agent_data(agent_id, "file_contexts")
        
        found_idx = next((i for i, ctx in enumerate(contexts) if os.path.normpath(ctx.get("file_path", "")) == normalized_path), None)
        
        if found_idx is None:
            return {
                "status": "cache_miss",
                "file_path": normalized_path,
                "message": "File not found in coding context cache."
            }
        
        found = contexts[found_idx]
        
        # Check freshness
        freshness_status = "unknown"
        try:
             if os.path.exists(normalized_path):
                with open(normalized_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    current_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    
                    stored_hash = found.get("content_hash")
                    freshness_status = "fresh" if current_hash == stored_hash else "stale"
             else:
                freshness_status = "missing"
        except Exception:
            pass

        # Update access metrics
        try:
            found["access_count"] = found.get("access_count", 0) + 1
            found["last_accessed_at"] = datetime.now().isoformat()
            contexts[found_idx] = found
            self._save_agent_data(agent_id, "file_contexts", contexts)
        except Exception:
            pass

        return {
            "status": "cache_hit",
            "freshness": freshness_status,
            "file_path": normalized_path,
            "code_flow": json.loads(found.get("compact_skeleton", "{}")), 
            "storage_mode": found.get("storage_mode"),
            "message": f"Code Mem retrieved. Status: {freshness_status}"
        }

    # =========================================================================
    # Retrieval Operations
    # =========================================================================

    # search_chunks and search_code_flow have been moved to CodingHybridRetriever
    # to separate logic from storage mechanism.

    def list_code_mems(self, agent_id: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List all stored code mem structures."""
        contexts = self._load_agent_data(agent_id, "file_contexts")
        total = len(contexts)
        sliced = contexts[offset : offset + limit]
        
        summaries = []
        for ctx in sliced:
            summaries.append({
                "context_id": ctx.get("id"),
                "file_path": ctx.get("file_path"),
                "language": ctx.get("language"),
                "last_accessed": ctx.get("last_accessed_at"),
                "size_bytes": ctx.get("size_bytes", 0)
            })
            
        return {
            "total": total,
            "items": summaries
        }

    def delete_code_mem(self, agent_id: str, file_path: str) -> bool:
        """Delete a code mem entry."""
        contexts = self._load_agent_data(agent_id, "file_contexts")
        normalized_path = os.path.normpath(file_path)
        
        initial_len = len(contexts)
        contexts = [c for c in contexts if os.path.normpath(c.get("file_path", "")) != normalized_path]
        
        if len(contexts) < initial_len:
            self._save_agent_data(agent_id, "file_contexts", contexts)
            self._remove_from_global_index(normalized_path)
            return True
        return False

    
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
    # Cross-Reference & Dependency Methods (Tier 1)
    # =========================================================================
    
    def find_symbol_references(self, agent_id: str, symbol: str) -> Dict[str, Any]:
        """
        Find all references to a symbol across indexed files.
        Uses global_index for file-level lookup, then scans chunks for details.
        """
        symbol_lower = symbol.lower()
        
        # 1. Search global index for files containing this symbol
        matching_files = set()
        for key, files in self._global_index.items():
            if symbol_lower in key.lower() or key.lower() in symbol_lower:
                matching_files.update(files)
        
        # 2. Scan chunks in matching files for detailed references
        contexts = self._load_agent_data(agent_id, "file_contexts")
        references = []
        
        for ctx in contexts:
            ctx_path = os.path.normpath(ctx.get("file_path", ""))
            chunks = ctx.get("chunks", [])
            
            # Check if file is in global index matches OR chunks contain symbol
            file_matches = ctx_path in matching_files
            
            for chunk in chunks:
                chunk_name = chunk.get("name", "").lower()
                chunk_keywords = [kw.lower() for kw in chunk.get("keywords", [])]
                chunk_content = chunk.get("content", "").lower()
                
                # Match by: name contains symbol, symbol in keywords, or symbol in content
                if (symbol_lower in chunk_name or 
                    symbol_lower in chunk_keywords or
                    (file_matches and symbol_lower in chunk_content)):
                    
                    references.append({
                        "file_path": ctx.get("file_path", ""),
                        "chunk_name": chunk.get("name", "unknown"),
                        "chunk_type": chunk.get("type", "unknown"),
                        "start_line": chunk.get("start_line", 0),
                        "end_line": chunk.get("end_line", 0),
                        "context": (chunk.get("content", "")[:200] + "...") 
                                   if len(chunk.get("content", "")) > 200 
                                   else chunk.get("content", ""),
                        "match_reason": "definition" if symbol_lower == chunk_name 
                                       else "keyword" if symbol_lower in chunk_keywords
                                       else "usage"
                    })
        
        return {
            "symbol": symbol,
            "total_references": len(references),
            "files_matched": len(set(r["file_path"] for r in references)),
            "references": references
        }
    
    def get_file_imports(self, agent_id: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract import information from a file's stored chunks.
        Returns list of import details with module names.
        """
        import re as _re
        normalized = os.path.normpath(file_path)
        contexts = self._load_agent_data(agent_id, "file_contexts")
        found = next(
            (ctx for ctx in contexts 
             if os.path.normpath(ctx.get("file_path", "")) == normalized),
            None
        )
        
        if found is None:
            return []
        
        imports = []
        # Robust regex to handle 'line:X' prefix and find imports anywhere in the chunk content
        # Group 1: from X import, Group 2: import X
        import_pattern = _re.compile(r'(?:line:\d+\s+)?(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))', _re.MULTILINE)
        
        for chunk in found.get("chunks", []):
            content = chunk.get("content", "")
            matches = import_pattern.finditer(content)
            
            for match in matches:
                from_mod = match.group(1)
                direct_mod = match.group(2)
                module = from_mod or direct_mod
                
                if not module:
                     continue
                     
                import_info = {
                    "raw": match.group(0).strip(),
                    "module": module,
                    "type": "from_import" if from_mod else "direct_import",
                    "line": chunk.get("start_line", 0)
                }
                
                # Deduplicate imports within the same file
                if not any(imp["module"] == module for imp in imports):
                    imports.append(import_info)
        
        return imports
    
    def find_importers(self, agent_id: str, module_name: str) -> List[Dict[str, Any]]:
        """
        Find all files that import a given module name.
        Reverse lookup: "who imports X?"
        """
        module_lower = module_name.lower()
        # Also match the last segment (e.g., "coding_store" matches ".coding_store")
        module_basename = module_name.split(".")[-1].lower()
        
        contexts = self._load_agent_data(agent_id, "file_contexts")
        importers = []
        
        for ctx in contexts:
            for chunk in ctx.get("chunks", []):
                # Search all chunks as imports might be grouped into 'mixed' type
                content = chunk.get("content", "").lower()
                if module_lower in content or module_basename in content:
                    # Double check it's actually an import statement to avoid false positives
                    if "import " in content:
                        importers.append({
                            "file_path": ctx.get("file_path", ""),
                            "import_statement": chunk.get("content", "").strip(),
                            "line": chunk.get("start_line", 0)
                        })
                        break  # One match per file is enough
        
        return importers
    
    def get_detailed_cache_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics with per-file freshness checks.
        """
        contexts = self._load_agent_data(agent_id, "file_contexts")
        sessions = self._load_agent_data(agent_id, "coding_sessions")
        
        # Per-file stats with freshness
        per_file = []
        freshness_counts = {"fresh": 0, "stale": 0, "missing": 0, "unknown": 0}
        total_chunks = 0
        
        for ctx in contexts:
            file_path = ctx.get("file_path", "")
            normalized = os.path.normpath(file_path) if file_path else ""
            chunk_count = len(ctx.get("chunks", []))
            total_chunks += chunk_count
            
            # Check freshness
            freshness = "unknown"
            try:
                if normalized and os.path.exists(normalized):
                    with open(normalized, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    current_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    stored_hash = ctx.get("content_hash")
                    freshness = "fresh" if current_hash == stored_hash else "stale"
                elif normalized:
                    freshness = "missing"
            except Exception:
                pass
            
            freshness_counts[freshness] = freshness_counts.get(freshness, 0) + 1
            
            per_file.append({
                "file": os.path.basename(file_path) if file_path else "unknown",
                "file_path": file_path,
                "chunks": chunk_count,
                "tokens": ctx.get("token_estimate", 0),
                "language": ctx.get("language", "unknown"),
                "freshness": freshness,
                "last_accessed": ctx.get("last_accessed_at", ""),
                "access_count": ctx.get("access_count", 0)
            })
        
        # Sort by token count descending
        per_file.sort(key=lambda x: x["tokens"], reverse=True)
        
        # Aggregate stats
        total_tokens = sum(f["tokens"] for f in per_file)
        total_cache_hits = sum(s.get("cache_hits", 0) for s in sessions)
        total_cache_misses = sum(s.get("cache_misses", 0) for s in sessions)
        hit_rate = round(
            (total_cache_hits / (total_cache_hits + total_cache_misses) * 100)
            if (total_cache_hits + total_cache_misses) > 0 else 0, 2
        )
        
        # Generate recommendations
        recommendations = []
        if freshness_counts["stale"] > 0:
            recommendations.append(
                f"{freshness_counts['stale']} file(s) are stale — consider running delta_update"
            )
        if freshness_counts["missing"] > 0:
            recommendations.append(
                f"{freshness_counts['missing']} file(s) missing from disk — consider remove_index"
            )
        if total_cache_hits == 0 and len(contexts) > 0:
            recommendations.append(
                "No cache hits recorded yet — use summarize_context instead of view_file"
            )
        
        return {
            "overview": {
                "total_files": len(contexts),
                "total_chunks": total_chunks,
                "total_tokens_cached": total_tokens,
                "cache_hit_rate": hit_rate,
                "sessions_tracked": len(sessions)
            },
            "freshness": freshness_counts,
            "per_file": per_file,
            "recommendations": recommendations
        }
    
    # =========================================================================
    # Invalidation & Usage Analytics (Tier 2)
    # =========================================================================
    
    def invalidate_with_vectors(
        self, agent_id: str, file_path: str
    ) -> Dict[str, Any]:
        """
        Remove a file context and return its chunk hash_ids for vector cleanup.
        """
        normalized = os.path.normpath(file_path)
        contexts = self._load_agent_data(agent_id, "file_contexts")
        
        found_idx = next(
            (i for i, c in enumerate(contexts)
             if os.path.normpath(c.get("file_path", "")) == normalized),
            None
        )
        
        if found_idx is None:
            return {"removed": False, "hash_ids": []}
        
        ctx = contexts[found_idx]
        hash_ids = [
            ch.get("hash_id") for ch in ctx.get("chunks", [])
            if ch.get("hash_id")
        ]
        
        # Remove from global index
        self._remove_from_global_index(normalized)
        
        # Remove from contexts
        contexts.pop(found_idx)
        self._save_agent_data(agent_id, "file_contexts", contexts)
        
        return {"removed": True, "hash_ids": hash_ids, "file_path": normalized}
    
    def invalidate_stale(self, agent_id: str) -> Dict[str, Any]:
        """
        Find and remove all stale file contexts + return their vector hash_ids.
        """
        contexts = self._load_agent_data(agent_id, "file_contexts")
        stale_files = []
        all_hash_ids = []
        
        for ctx in contexts:
            fp = ctx.get("file_path", "")
            normalized = os.path.normpath(fp) if fp else ""
            
            try:
                if normalized and os.path.exists(normalized):
                    with open(normalized, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    current_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    stored_hash = ctx.get("content_hash")
                    if current_hash != stored_hash:
                        stale_files.append(ctx)
                        all_hash_ids.extend(
                            ch.get("hash_id") for ch in ctx.get("chunks", []) if ch.get("hash_id")
                        )
                elif normalized:
                    # File missing from disk — also stale
                    stale_files.append(ctx)
                    all_hash_ids.extend(
                        ch.get("hash_id") for ch in ctx.get("chunks", []) if ch.get("hash_id")
                    )
            except Exception:
                pass
        
        if stale_files:
            stale_paths = {os.path.normpath(s.get("file_path", "")) for s in stale_files}
            fresh = [c for c in contexts if os.path.normpath(c.get("file_path", "")) not in stale_paths]
            self._save_agent_data(agent_id, "file_contexts", fresh)
            for sp in stale_paths:
                self._remove_from_global_index(sp)
        
        return {
            "invalidated": len(stale_files),
            "hash_ids": all_hash_ids,
            "files": [s.get("file_path", "") for s in stale_files]
        }
    
    def get_usage_report(self, agent_id: str) -> Dict[str, Any]:
        """
        Aggregate usage analytics: access counts, tokens, indexing stats.
        """
        contexts = self._load_agent_data(agent_id, "file_contexts")
        sessions = self._load_agent_data(agent_id, "coding_sessions")
        
        # Most accessed files
        most_accessed = sorted(
            contexts,
            key=lambda c: c.get("access_count", 0),
            reverse=True
        )[:10]
        
        # Language distribution
        lang_dist = {}
        for ctx in contexts:
            lang = ctx.get("language", "unknown")
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        
        # Chunk stats
        total_chunks = sum(len(ctx.get("chunks", [])) for ctx in contexts)
        avg_chunks = round(total_chunks / len(contexts), 1) if contexts else 0
        
        return {
            "sessions": {
                "total": len(sessions),
                "total_tokens_stored": sum(s.get("tokens_stored", 0) for s in sessions),
                "total_tokens_retrieved": sum(s.get("tokens_retrieved", 0) for s in sessions),
                "total_cache_hits": sum(s.get("cache_hits", 0) for s in sessions),
                "total_cache_misses": sum(s.get("cache_misses", 0) for s in sessions),
            },
            "most_accessed_files": [
                {
                    "file": os.path.basename(c.get("file_path", "")),
                    "file_path": c.get("file_path", ""),
                    "access_count": c.get("access_count", 0),
                    "tokens": c.get("token_estimate", 0),
                    "language": c.get("language", "unknown"),
                }
                for c in most_accessed
            ],
            "indexing_activity": {
                "total_files_indexed": len(contexts),
                "total_chunks": total_chunks,
                "avg_chunks_per_file": avg_chunks,
                "language_distribution": lang_dist,
            },
            "total_tokens_cached": sum(c.get("token_estimate", 0) for c in contexts),
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

