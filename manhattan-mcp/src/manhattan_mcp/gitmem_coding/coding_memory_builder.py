"""
Coding Memory Builder

Handles the ingestion of code chunks, ensuring they have vector embeddings
before storage. Acts as the construction phase for the Coding Context.
"""
from typing import List, Dict, Any, Optional
import os
import uuid
from .models import CodeChunk
from ..gitmem.embedding import RemoteEmbeddingClient
from .coding_store import CodingContextStore
import logging

# Configure logger
logger = logging.getLogger(__name__)

class CodingMemoryBuilder:
    """
    Orchestrates the ingestion of code contexts.
    
    Responsibilities:
    1. Accepts code chunks (from file or pre-computed).
    2. Ensures every chunk has a vector embedding (calls RemoteEmbeddingClient).
    3. Persists the enriched chunks to CodingContextStore.
    """
    def __init__(
        self,
        store: CodingContextStore,
        embedding_client: Optional[RemoteEmbeddingClient] = None
    ):
        self.store = store
        self.embedding_client = embedding_client or RemoteEmbeddingClient()
        
    def process_file_chunks(
        self,
        agent_id: str,
        file_path: str,
        chunks: List[Dict[str, Any]],
        language: str = "auto",
        session_id: str = ""
    ) -> Dict[str, Any]:
        """
        Process and store chunks for a file.
        Ensures all chunks have embeddings before storage.
        """
        enriched_chunks = []
        
        # 1. Enrich chunks with embeddings
        for chunk_data in chunks:
            # Create a shallow copy to avoid modifying original if needed, 
            # but we likely want to modify it for storage.
            chunk = chunk_data.copy()
            
            # Check if vector is missing or empty
            if "vector" not in chunk or not chunk["vector"]:
                # 1.1 TRY CACHE DEDUPLICATION
                hash_id = chunk.get("hash_id")
                if hash_id:
                    cached_chunk = self.store.get_cached_chunk(hash_id)
                    if cached_chunk and cached_chunk.get("vector"):
                        chunk["vector"] = cached_chunk["vector"]
                        # logger.info(f"Cache hit for chunk {hash_id[:8]}")
                        enriched_chunks.append(chunk)
                        continue

                # 1.2 Generate embedding (Cache Miss)
                content_to_embed = self._prepare_embedding_text(chunk)
                if content_to_embed:
                    try:
                        vector = self.embedding_client.embed(content_to_embed)
                        # Ensure serialization: convert to list if it's a vector object
                        if hasattr(vector, 'tolist'):
                            vector = vector.tolist()
                        elif not isinstance(vector, list):
                            vector = list(vector)
                            
                        chunk["vector"] = vector
                        
                        # 1.3 Cache the new chunk
                        if hash_id:
                            self.store.cache_chunk(chunk)
                            
                    except Exception as e:
                        logger.error(f"Failed to generate embedding for chunk {chunk.get('name', 'unknown')}: {e}")
                        # Depending on policy, we might skip or store without vector
                        chunk["vector"] = []
            
            enriched_chunks.append(chunk)

        # 2. Store enriched chunks
        return self.store.store_file_chunks(
            agent_id=agent_id,
            file_path=file_path,
            chunks=enriched_chunks,
            language=language,
            session_id=session_id
        )

    def _prepare_embedding_text(self, chunk: Dict[str, Any]) -> str:
        """
        Prepare the text to be embedded for a chunk.
        Prefers 'summary' if available, otherwise 'content'.
        Combines with 'keywords' for richer context.
        """
        text_parts = []
        
        # 1. Summary or Content
        if chunk.get("summary"):
            text_parts.append(chunk["summary"])
        elif chunk.get("content"):
            # Limit content length to avoid exceeding token limits of embedding model
            content = chunk["content"]
            text_parts.append(content[:1000]) # Truncate if too long?
        
        # 2. Keywords
        keywords = chunk.get("keywords", [])
        if keywords:
            text_parts.append(f"Keywords: {', '.join(keywords)}")
            
        # 3. Name/Type
        name = chunk.get("name", "")
        type_ = chunk.get("type", "")
        if name or type_:
            text_parts.append(f"Context: {type_} {name}")
            
        return "\n".join(text_parts)
