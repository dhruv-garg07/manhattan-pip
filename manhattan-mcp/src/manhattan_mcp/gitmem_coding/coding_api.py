"""
GitMem Coding - Coding Context API

Simplified High-level API for code flow management.
Provides exactly 4 CRUD operations to manage the Code Flow structure.
"""

from typing import Dict, Any, List
from .coding_store import CodingContextStore
from .coding_vector_store import CodingVectorStore
from .coding_memory_builder import CodingMemoryBuilder
from .coding_hybrid_retriever import CodingHybridRetriever
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
        self.vector_store = CodingVectorStore(root_path=root_path)
        self.builder = CodingMemoryBuilder(self.store, self.vector_store)
        self.retriever = CodingHybridRetriever(self.store, self.vector_store)
    
    # 1. Create
    def create_flow(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a Code Flow structure for a file.
        Chunks MUST be provided by the caller (Coding Agent).
        """
        if chunks is None:
            raise ValueError("Chunks must be provided explicitly. Auto-chunking is disabled.")
            
        return self.builder.process_file_chunks(
            agent_id=agent_id,
            file_path=file_path,
            chunks=chunks
        )

    # 2. Read (Query or ID)
    def get_flow(self, agent_id: str, query: str) -> Dict[str, Any]:
        """
        Retrieve Code Flow/Context based on query.
        
        Performs a hybrid search on chunks.
        """
        # Hybrid Search on Chunks via Retriever
        results = self.retriever.search(agent_id, query)
        
        return results

    # 3. Update
    def update_flow(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update the Code Flow for a file.
        Chunks MUST be provided explicitly.
        """
        if chunks is None:
            raise ValueError("Chunks must be provided explicitly. Auto-chunking is disabled.")
            
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
