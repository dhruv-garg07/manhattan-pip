"""
Coding Hybrid Retriever

Handles the retrieval of code chunks using a hybrid approach (Vector + Keyword).
"""
from typing import List, Dict, Any, Optional
import os
import json
from ..gitmem.embedding import RemoteEmbeddingClient
from .coding_store import CodingContextStore
from .ast_skeleton import retrieve_path
import logging

logger = logging.getLogger(__name__)

class CodingHybridRetriever:
    """
    Centralized retrieval logic for coding contexts.
    Combines semantic search (Vectors) and exact matching (Keywords/Symbols).
    """
    def __init__(
        self,
        store: CodingContextStore,
        embedding_client: Optional[RemoteEmbeddingClient] = None
    ):
        self.store = store
        self.embedding_client = embedding_client or RemoteEmbeddingClient()

    def search(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        hybrid_alpha: float = 0.7 # Weight for vector score (0.0 to 1.0)
    ) -> Dict[str, Any]:
        """
        Perform a hybrid search for the query.
        
        If query looks like a file path, attempts to retrieve that file's context.
        Otherwise, performs hybrid search across all chunks.
        """
        # 1. File Path / ID Check (Direct Lookup)
        # This preserves existing functionality for direct file access
        if "/" in query or "." in query.split("/")[-1]:
             if os.path.exists(query):
                result = self.store.retrieve_file_context(agent_id, query)
                if result.get("status") != "cache_miss":
                    return result
        
        # 2. Semantic/Keyword Search
        results = self._hybrid_search_chunks(agent_id, query, top_k, hybrid_alpha)
        
        return {
            "status": "search_results",
            "query": query,
            "results": results,
            "count": len(results)
        }

    def _hybrid_search_chunks(
        self,
        agent_id: str,
        query: str,
        top_k: int,
        alpha: float
    ) -> List[Dict[str, Any]]:
        """
        Execute the hybrid search logic.
        """
        # A. Prepare Query Representations
        
        # Vector embedding
        query_vector = []
        try:
             query_vector = self.embedding_client.embed(query)
        except Exception as e:
            logger.warning(f"Embedding generation failed for query '{query}': {e}")
            
        # Keyword tokens (simple split)
        query_keywords = set(query.lower().split())

        scored_chunks = []
        # We need to access stored contexts. 
        # Ideally store provides an iterator or we load all (memory constraints?)
        # For now, load all file contexts for the agent.
        contexts = self.store._load_agent_data(agent_id, "file_contexts")
        
        for ctx in contexts:
            file_path = ctx.get("file_path", "")
            chunks = ctx.get("chunks", [])
            
            for chunk in chunks:
                # 1. Vector Score (Cosine Similarity)
                vec_score = 0.0
                chunk_vec = chunk.get("vector")
                if query_vector and chunk_vec:
                    # Dot product (assuming normalized vectors)
                    try:
                        vec_score = sum(a*b for a, b in zip(query_vector, chunk_vec))
                    except:
                        vec_score = 0.0
                
                # 2. Keyword Score (Jaccard-ish)
                kw_score = 0.0
                chunk_kws = set(chunk.get("keywords", []))
                # Add name to keywords for matching
                if chunk.get("name"):
                    chunk_kws.add(chunk.get("name").lower())
                    
                if chunk_kws and query_keywords:
                    overlap = len(query_keywords.intersection(chunk_kws))
                    if overlap > 0:
                        kw_score = overlap / len(query_keywords) 
                        # Adjust denominator? Jaccard is union. 
                        # But here we care about how much of query is covered.
                
                # 3. Hybrid Combination
                # Final = alpha * Vector + (1-alpha) * Keyword
                final_score = (vec_score * alpha) + (kw_score * (1.0 - alpha))
                
                if final_score > 0.01: # Threshold to filter noise
                    scored_chunks.append({
                        "file_path": file_path,
                        "chunk": chunk,
                        "score": final_score,
                        "match_type": "hybrid",
                        "vector_score": vec_score,
                        "keyword_score": kw_score
                    })

        # B. Fallback / Enhancement: Global Symbol Index Search
        # If hybrid search yields low results, or specifically for symbol lookup
        # We can also check the global index for exact matches of the query
        # This was the "Symbol Search" features.
        
        # (Optional: Merge Global Index results if they point to files not in top chunks?)
        # For now, let's trust the hybrid search on chunks since chunks contain names/keywords.
        
        # Sort and return top_k
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
