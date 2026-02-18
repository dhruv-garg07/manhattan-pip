"""
Coding Memory Builder

Handles the ingestion of code chunks, ensuring they have vector embeddings
stored in the dedicated CodingVectorStore (vectors.json) before storage.
"""
from typing import List, Dict, Any, Optional
import os
import uuid
from .models import CodeChunk
from ..gitmem.embedding import RemoteEmbeddingClient
from .coding_store import CodingContextStore
from .coding_vector_store import CodingVectorStore
import logging

# Configure logger
logger = logging.getLogger(__name__)

class CodingMemoryBuilder:
    """
    Orchestrates the ingestion of code contexts.
    
    Responsibilities:
    1. Accepts code chunks (from file or pre-computed).
    2. Ensures every chunk has a vector embedding stored in CodingVectorStore.
    3. Persists the chunks (without inline vectors) to CodingContextStore.
    """
    def __init__(
        self,
        store: CodingContextStore,
        vector_store: CodingVectorStore,
        embedding_client: Optional[RemoteEmbeddingClient] = None
    ):
        self.store = store
        self.vector_store = vector_store
        self.embedding_client = embedding_client or vector_store.embedding_client
        
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
        Ensures all chunks have embeddings in vectors.json before storage.
        """
        import hashlib
        import re

        # Phase 1: Preparation & Deduplication
        chunks_to_embed = []
        embedding_indices = []
        enriched_chunks = []
        
        for i, chunk_data in enumerate(chunks):
            chunk = chunk_data.copy()
            
            # Ensure hash_id exists
            if not chunk.get("hash_id") and chunk.get("content"):
                normalized = re.sub(r'\s+', ' ', chunk["content"]).strip()
                chunk["hash_id"] = hashlib.sha256(normalized.encode('utf-8')).hexdigest()

            hash_id = chunk.get("hash_id")
            existing_vec = None
            
            if hash_id:
                existing_vec = self.vector_store.get_vector(agent_id, hash_id)
                if not existing_vec:
                    cached_chunk = self.store.get_cached_chunk(hash_id)
                    if cached_chunk and cached_chunk.get("vector"):
                        self.vector_store.add_vector_raw(agent_id, hash_id, cached_chunk["vector"])
                        existing_vec = cached_chunk["vector"]

            if not existing_vec:
                content_to_embed = self._prepare_embedding_text(chunk)
                if content_to_embed:
                    chunks_to_embed.append(content_to_embed)
                    embedding_indices.append(i)
            
            # Set embedding_id if we have a hash_id
            if hash_id:
                chunk["embedding_id"] = hash_id
            
            chunk.pop("vector", None)
            enriched_chunks.append(chunk)

        # Phase 2: Batch Embedding
        if chunks_to_embed:
            try:
                vectors = self.embedding_client.embed_batch(chunks_to_embed)
                
                # Store new vectors
                for idx, vector in zip(embedding_indices, vectors):
                    if hasattr(vector, 'tolist'):
                        vector = vector.tolist()
                    
                    hash_id = enriched_chunks[idx].get("hash_id")
                    if hash_id:
                        self.vector_store.add_vector_raw(agent_id, hash_id, vector)
                        
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")

        # Phase 3: Final Caching
        for chunk in enriched_chunks:
            hash_id = chunk.get("hash_id")
            if hash_id:
                self.store.cache_chunk(chunk)

        # Store chunks (no inline vectors)
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
        Combines summary, content, keywords, and metadata for rich embedding.
        
        Strategy:
        - Summary provides high-level semantic meaning
        - Content snippet provides exact code patterns (decorators, function calls)
        - Keywords provide searchable concepts
        - Type/Name/Lines provide structural context
        """
        import re
        text_parts = []
        
        # 1. Name & Type header (important for direct name queries)
        name = chunk.get("name", "")
        type_ = chunk.get("type", "")
        if name or type_:
            text_parts.append(f"{type_} {name}".strip())
        
        # 2. Summary (primary semantic content)
        if chunk.get("summary"):
            text_parts.append(chunk["summary"])
        
        # 3. Content snippet (critical for exact pattern matching)
        if chunk.get("content"):
            content = chunk["content"]
            # Include first 500 chars of content for exact pattern matches
            text_parts.append(content[:500])
            
            # 4. Extract decorators & important patterns from content
            decorators = re.findall(r'@\w+[\.\w]*(?:\([^)]*\))?', content)
            if decorators:
                text_parts.append(f"Decorators: {', '.join(decorators)}")
            
            # Extract HTTP methods from route decorators
            methods_match = re.findall(r"methods\s*=\s*\[([^\]]+)\]", content)
            if methods_match:
                text_parts.append(f"HTTP methods: {', '.join(methods_match)}")
        
        # 5. Keywords
        keywords = chunk.get("keywords", [])
        if keywords:
            text_parts.append(f"Keywords: {', '.join(keywords)}")
        
        # 6. Line range (for positional queries)
        start_line = chunk.get("start_line", 0)
        end_line = chunk.get("end_line", 0)
        if start_line or end_line:
            text_parts.append(f"Lines: {start_line}-{end_line}")
            
        return "\n".join(text_parts)
