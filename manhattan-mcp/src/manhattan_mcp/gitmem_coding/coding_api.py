"""
GitMem Coding - Coding Context API

High-level API for code memory management.
Provides VFS navigation tools + CRUD operations for agent-facing MCP tools.
"""

from typing import Dict, Any, List, Optional
from .coding_store import CodingContextStore
from .coding_vector_store import CodingVectorStore
from .coding_memory_builder import CodingMemoryBuilder
from .coding_hybrid_retriever import CodingHybridRetriever
from .coding_file_system import CodingFileSystem
import os
import json
import hashlib


class CodingAPI:
    """
    High-level API for coding context storage.
    
    VFS Navigation Tools (for MCP):
    1. read_file_context(file_path) — cached read with auto-index
    2. get_file_outline(file_path) — structural outline only
    3. list_directory(path) — browse indexed files
    4. search_codebase(query) — hybrid semantic+keyword search
    5. get_token_savings() — savings report
    
    CRUD Operations:
    6. index_file(file_path) — index/create
    7. reindex_file(file_path) — re-index/update
    8. remove_index(file_path) — delete
    9. list_indexed_files() — list all
    """
    
    def __init__(self, root_path: str = "./.gitmem_coding"):
        """Initialize the Coding API."""
        self.store = CodingContextStore(root_path=root_path)
        self.vector_store = CodingVectorStore(root_path=root_path)
        self.builder = CodingMemoryBuilder(self.store, self.vector_store)
        self.retriever = CodingHybridRetriever(self.store, self.vector_store)
        self.filesystem = CodingFileSystem(self.store)
    
    # =========================================================================
    # VFS Navigation Tools (NEW — for agent-facing MCP tools)
    # =========================================================================
    
    def read_file_context(self, agent_id: str, file_path: str) -> Dict[str, Any]:
        """
        Read a file's compressed context from cache, or auto-index if not cached.
        
        Returns:
            Dict with compressed context (chunks, outline, metadata), token info.
        """
        normalized = os.path.normpath(file_path)
        
        # 1. Check cache
        cached = self.store.retrieve_file_context(agent_id, normalized)
        
        if cached.get("status") == "cache_hit":
            # Calculate token savings
            original_tokens = self._estimate_file_tokens(normalized)
            cached_tokens = sum(
                len(str(c)) // 4 
                for c in cached.get("code_flow", {}).get("tree", [])
            ) if cached.get("code_flow") else 0
            
            return {
                "status": "cache_hit",
                "freshness": cached.get("freshness", "unknown"),
                "file_path": normalized,
                "code_flow": cached.get("code_flow", {}),
                "message": f"Returning compressed context from cache.",
                "_token_info": {
                    "tokens_this_call": cached_tokens,
                    "tokens_if_raw_read": original_tokens,
                    "tokens_saved": max(0, original_tokens - cached_tokens),
                    "hint": f"Saved ~{max(0, original_tokens - cached_tokens)} tokens by using cached context"
                }
            }
        
        # 2. Cache miss — read real file and auto-index
        if not os.path.exists(normalized):
            return {
                "status": "error",
                "message": f"File not found: {normalized}",
                "file_path": normalized
            }
        
        try:
            with open(normalized, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            original_tokens = int(len(content) * 0.25)
            
            # Auto-index the file
            index_result = self.index_file(agent_id, normalized)
            
            # Now retrieve the cached version
            cached = self.store.retrieve_file_context(agent_id, normalized)
            code_flow = cached.get("code_flow", {}) if cached.get("status") == "cache_hit" else {}
            
            cached_tokens = sum(
                len(str(c)) // 4 
                for c in code_flow.get("tree", [])
            ) if code_flow else original_tokens
            
            return {
                "status": "auto_indexed",
                "file_path": normalized,
                "code_flow": code_flow,
                "message": f"File was not cached. Auto-indexed and returning compressed context.",
                "_token_info": {
                    "tokens_this_call": cached_tokens,
                    "tokens_if_raw_read": original_tokens,
                    "tokens_saved": max(0, original_tokens - cached_tokens),
                    "hint": f"File auto-indexed. Future reads will save ~{max(0, original_tokens - cached_tokens)} tokens."
                }
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to read file: {str(e)}"}
    
    def get_file_outline(self, agent_id: str, file_path: str) -> Dict[str, Any]:
        """
        Get structural outline of a file — chunk names, types, signatures, line ranges.
        No full content. Auto-indexes if not cached.
        """
        normalized = os.path.normpath(file_path)
        
        # Check if indexed
        contexts = self.store._load_agent_data(agent_id, "file_contexts")
        found = next(
            (ctx for ctx in contexts 
             if os.path.normpath(ctx.get("file_path", "")) == normalized), 
            None
        )
        
        if found is None:
            # Auto-index first
            if os.path.exists(normalized):
                self.index_file(agent_id, normalized)
                contexts = self.store._load_agent_data(agent_id, "file_contexts")
                found = next(
                    (ctx for ctx in contexts 
                     if os.path.normpath(ctx.get("file_path", "")) == normalized), 
                    None
                )
        
        if found is None:
            return {
                "status": "error",
                "message": f"File not found and could not be indexed: {normalized}"
            }
        
        # Extract outline from chunks
        chunks = found.get("chunks", [])
        outline_items = []
        for chunk in chunks:
            item = {
                "name": chunk.get("name", "unknown"),
                "type": chunk.get("type", "unknown"),
                "start_line": chunk.get("start_line", 0),
                "end_line": chunk.get("end_line", 0),
            }
            # Include summary if available (compact)
            if chunk.get("summary"):
                item["summary"] = chunk["summary"]
            # Include signature (content first line only)
            content = chunk.get("content", "")
            if content:
                first_line = content.split("\n")[0].strip()
                item["signature"] = first_line
            # Include keywords
            if chunk.get("keywords"):
                item["keywords"] = chunk["keywords"]
            outline_items.append(item)
        
        original_tokens = self._estimate_file_tokens(normalized)
        outline_tokens = int(len(json.dumps(outline_items)) * 0.25)
        
        return {
            "status": "ok",
            "file_path": normalized,
            "language": found.get("language", "unknown"),
            "total_chunks": len(outline_items),
            "outline": outline_items,
            "_token_info": {
                "tokens_this_call": outline_tokens,
                "tokens_if_raw_read": original_tokens,
                "tokens_saved": max(0, original_tokens - outline_tokens),
                "hint": f"Outline uses ~{min(100, max(1, int(outline_tokens / max(1, original_tokens) * 100)))}% of raw file tokens"
            }
        }
    
    def list_directory(self, agent_id: str, path: str = "") -> Dict[str, Any]:
        """
        List indexed files through the VFS, organized by language.
        """
        nodes = self.filesystem.list_dir(agent_id, path)
        return {
            "status": "ok",
            "path": path or "/",
            "items": nodes,
            "count": len(nodes)
        }
    
    def get_token_savings(self, agent_id: str) -> Dict[str, Any]:
        """Get the token savings report."""
        return self.store.get_token_savings_report(agent_id)
    
    def _estimate_file_tokens(self, file_path: str) -> int:
        """Estimate tokens for a file on disk."""
        try:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                return int(size * 0.25)  # ~4 chars per token
        except Exception:
            pass
        return 0
    
    # =========================================================================
    # CRUD Operations (renamed for agent-facing tools)
    # =========================================================================
    
    def index_file(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Index a file (create Code Mem). Alias: create_mem.
        If chunks are not provided, falls back to local AST parsing.
        """
        return self._ingest_file(agent_id, file_path, chunks)

    def reindex_file(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Re-index a file (update Code Mem). Alias: update_mem.
        If chunks are not provided, falls back to local AST parsing.
        """
        return self._ingest_file(agent_id, file_path, chunks)

    def search_codebase(self, agent_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search the indexed codebase with hybrid semantic + keyword search.
        Alias: get_mem.
        """
        return self.retriever.search(agent_id, query, top_k=top_k)

    def remove_index(self, agent_id: str, file_path: str) -> bool:
        """Remove a file's index. Alias: delete_mem."""
        return self.store.delete_code_mem(agent_id, file_path)

    def list_indexed_files(self, agent_id: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List all indexed files. Alias: list_mems."""
        return self.store.list_code_mems(agent_id, limit, offset)

    # =========================================================================
    # Backward-compatible aliases
    # =========================================================================
    
    def create_mem(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Backward-compatible alias for index_file."""
        return self.index_file(agent_id, file_path, chunks)
    
    def get_mem(self, agent_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Backward-compatible alias for search_codebase."""
        return self.search_codebase(agent_id, query, top_k)
    
    def update_mem(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Backward-compatible alias for reindex_file."""
        return self.reindex_file(agent_id, file_path, chunks)
    
    def delete_mem(self, agent_id: str, file_path: str) -> bool:
        """Backward-compatible alias for remove_index."""
        return self.remove_index(agent_id, file_path)
    
    def list_mems(self, agent_id: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Backward-compatible alias for list_indexed_files."""
        return self.list_indexed_files(agent_id, limit, offset)
    
    # =========================================================================
    # Internal helpers
    # =========================================================================
    
    def _ingest_file(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Shared logic for index_file and reindex_file."""
        from .chunking_engine import detect_language
        
        # Detect language for VFS categorization
        lang = detect_language(file_path)
        
        if chunks is None:
            # Fallback to local AST parsing
            if not os.path.exists(file_path):
                return {"status": "error", "message": f"File not found for auto-chunking: {file_path}"}
            
            from .chunking_engine import ChunkingEngine
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                chunker = ChunkingEngine.get_chunker(lang)
                code_chunks = chunker.chunk_file(content, file_path)
                
                # Convert CodeChunk objects to dicts
                chunks = []
                for cc in code_chunks:
                    c_dict = cc.to_dict()
                    if not c_dict.get("summary"):
                        c_dict["summary"] = f"Code unit: {c_dict.get('name', 'unnamed')}"
                    chunks.append(c_dict)
                    
            except Exception as e:
                return {"status": "error", "message": f"Auto-chunking failed: {str(e)}"}
            
        return self.builder.process_file_chunks(
            agent_id=agent_id,
            file_path=file_path,
            chunks=chunks,
            language=lang,
        )
