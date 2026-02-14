"""
GitMem Coding - Coding Context API

Simplified High-level API for code flow management.
Provides exactly 4 CRUD operations to manage the Code Flow structure.
"""

from typing import Dict, Any, List
from .coding_store import CodingContextStore
from .coding_memory_builder import CodingMemoryBuilder
from .coding_hybrid_retriever import CodingHybridRetriever
from .chunking_engine import ChunkingEngine, detect_language
import os

class CodingAPI:
    """
    High-level API for coding context storage.
    
    Exposes only 4 core CRUD operations:
    1. create_flow(file_path)
    2. get_flow(query)
    3. update_flow(file_path)
    4. delete_flow(file_path)
    """
    
    def __init__(self, root_path: str = "./.gitmem_coding"):
        """Initialize the Coding API."""
        self.store = CodingContextStore(root_path=root_path)
        self.builder = CodingMemoryBuilder(self.store)
        self.retriever = CodingHybridRetriever(self.store)
    
    # 1. Create
    def create_flow(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a Code Flow structure for a file.
        If chunks are provided, uses them.
        If chunks are NOT provided, reads file from disk and auto-chunks.
        
        ALWAYS passes through CodingMemoryBuilder to ensure embeddings are generated.
        """
        if chunks is None:
            chunks = self._read_and_chunk_file(file_path)
            
        return self.builder.process_file_chunks(
            agent_id=agent_id,
            file_path=file_path,
            chunks=chunks
        )

    # 2. Read (Query or ID)
    def get_flow(self, agent_id: str, query: str) -> Dict[str, Any]:
        """
        Retrieve Code Flow/Context based on query.
        
        If query looks like a file path, attempts to return the full tree.
        Otherwise, performs a hybrid search on chunks.
        """
        # Try interpreting query as a file path first for backward compat/debugging
        if "/" in query or "." in query.split("/")[-1]:
             if os.path.exists(query):
                result = self.store.retrieve_file_context(agent_id, query)
                if result.get("status") != "cache_miss":
                    return result
        
        # Hybrid Search on Chunks via Retriever
        results = self.retriever.search(agent_id, query)
        
        return results

    # 3. Update
    def update_flow(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update the Code Flow for a file.
        """
        if chunks is None:
            chunks = self._read_and_chunk_file(file_path)
            
        return self.builder.process_file_chunks(
            agent_id=agent_id,
            file_path=file_path,
            chunks=chunks
        )
    
    # 4. Delete
    def delete_flow(self, agent_id: str, file_path: str) -> bool:
        """
        Delete the Code Flow for a file.
        """
        return self.store.delete_code_flow(agent_id, file_path)

    # Additional utility (optional, for listing)
    def list_flows(self, agent_id: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List all stored code flows."""
        return self.store.list_code_flows(agent_id, limit, offset)

    def _read_and_chunk_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Helper: Read file from disk and generate chunks locally.
        """
        normalized_path = os.path.normpath(file_path)
        
        if not os.path.exists(normalized_path):
            raise FileNotFoundError(f"File not found: {normalized_path}")
            
        try:
            with open(normalized_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except OSError as e:
            raise IOError(f"Failed to read file: {e}")
            
        language = detect_language(normalized_path)
        chunker = ChunkingEngine.get_chunker(language)
        chunks_objs = chunker.chunk_file(content, file_path)
        
        return [c.to_dict() for c in chunks_objs]
