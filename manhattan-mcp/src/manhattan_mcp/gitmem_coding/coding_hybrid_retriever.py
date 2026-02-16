"""
Coding Hybrid Retriever

Handles the retrieval of code chunks using a hybrid approach (Vector + Keyword).
Loads vectors from the dedicated CodingVectorStore (vectors.json) rather than
reading inline vector fields from chunks.
"""
from typing import List, Dict, Any, Optional
import os
import re
import json
import string
from ..gitmem.embedding import RemoteEmbeddingClient
from .coding_store import CodingContextStore
from .coding_vector_store import CodingVectorStore
from .ast_skeleton import retrieve_path
import logging

logger = logging.getLogger(__name__)

class CodingHybridRetriever:
    """
    Centralized retrieval logic for coding contexts.
    Combines semantic search (Vectors from vectors.json) and exact matching (Keywords/Symbols).
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

    def search(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        hybrid_alpha: float = 0.7 
    ) -> Dict[str, Any]:
        """
        Perform a hybrid search for the query with metadata filtering.
        """
        # 1. Metadata Filtering Extraction
        file_filter = None
        clean_query = query
        route_tokens = []  # Collect route-style tokens for keyword enrichment
        
        # Known file extensions for distinguishing file paths from URL routes
        FILE_EXTENSIONS = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs',
            '.rb', '.php', '.c', '.cpp', '.h', '.hpp', '.cs', '.swift',
            '.kt', '.scala', '.vue', '.svelte', '.html', '.css', '.scss',
            '.json', '.yaml', '.yml', '.toml', '.xml', '.md', '.txt',
            '.sh', '.bash', '.sql', '.r', '.m', '.lua', '.zig',
        }
        
        tokens = query.split()
        for token in tokens:
            clean_token = token.strip(string.punctuation + "'\"")
            if not clean_token:
                continue
            
            # Check if token looks like a file path (must have a file extension)
            _, ext = os.path.splitext(clean_token)
            if ext.lower() in FILE_EXTENSIONS:
                file_filter = clean_token
                clean_query = query.replace(token, "").strip()
                break
            
            # Collect route-style tokens (e.g., /api/keys) for keyword enrichment
            if '/' in clean_token and not ext:
                route_tokens.append(clean_token)
        
        if not clean_query:
            clean_query = "summary overview" 

        # 2. Semantic/Keyword Search with Filter
        results = self._hybrid_search_chunks(
            agent_id, 
            clean_query, 
            top_k, 
            hybrid_alpha,
            file_filter=file_filter,
            route_tokens=route_tokens
        )
        
        return {
            "status": "search_results",
            "query": clean_query,
            "filter": file_filter,
            "results": results,
            "count": len(results)
        }

    def _hybrid_search_chunks(
        self,
        agent_id: str,
        query: str,
        top_k: int,
        alpha: float,
        file_filter: str = None,
        route_tokens: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute the hybrid search logic.
        Loads vectors from vectors.json via CodingVectorStore.
        """
        # A. Prepare Query Representations
        
        # Vector embedding for query
        query_vector = []
        try:
             query_vector = self.embedding_client.embed(query)
             if hasattr(query_vector, 'tolist'):
                 query_vector = query_vector.tolist()
        except Exception as e:
            logger.warning(f"Embedding generation failed for query '{query}': {e}")
            
        # Keyword tokens (simple split, removing small noise words)
        STOP_WORDS = {"is", "the", "a", "an", "and", "or", "in", "on", "with", "how", "what", "where", "to", "for", "of", "can", "you", "explain", "from", "pull", "context", "only"}
        query_keywords = {w.lower().strip(string.punctuation + "'\"")
                          for w in query.split()
                          if w.lower().strip(string.punctuation + "'\"") not in STOP_WORDS}
        query_keywords.discard("")
        
        # Decompose route-style tokens into individual keywords
        # e.g., "/api/keys" -> {"api", "keys"}
        route_patterns = []  # Original route strings for content matching
        if route_tokens:
            for rt in route_tokens:
                clean_rt = rt.strip(string.punctuation + "'\"")
                route_patterns.append(clean_rt)
                segments = [seg for seg in clean_rt.split('/') if seg and seg not in STOP_WORDS]
                query_keywords.update(s.lower() for s in segments)

        scored_chunks = []
        contexts = self.store._load_agent_data(agent_id, "file_contexts")
        
        # ... (vector loading same)
        all_vectors = self.vector_store._load_vectors(agent_id)
        if all_vectors is None:
            all_vectors = {}
        
        for ctx in contexts:
            file_path = ctx.get("file_path", "")
            
            # Metadata Filtering
            if file_filter:
                normalized_filter = file_filter.replace("/", os.sep).replace("\\", os.sep).lower()
                path_lower = file_path.lower()
                
                # Strict check first
                if normalized_filter in path_lower:
                    pass
                # Lenient check: if filename matches, allow it even if directories differ
                elif os.path.basename(normalized_filter) in os.path.basename(path_lower):
                    pass
                else:
                    continue

            chunks = ctx.get("chunks", [])
            for chunk in chunks:
                # 1. Vector Score (Cosine Similarity)
                vec_score = 0.0
                hash_id = chunk.get("hash_id") or chunk.get("embedding_id")
                chunk_vec = all_vectors.get(hash_id) if hash_id else None
                
                if (query_vector is not None and len(query_vector) > 0 
                        and chunk_vec and len(chunk_vec) > 0):
                    try:
                        vec_score = sum(a*b for a, b in zip(query_vector, chunk_vec))
                    except:
                        vec_score = 0.0
                
                # 2. Keyword Score (Enhanced)
                kw_score = 0.0
                chunk_kws = {kw.lower() for kw in chunk.get("keywords", [])}
                chunk_name = chunk.get("name", "").lower()
                chunk_summary = chunk.get("summary", "").lower()
                chunk_content = chunk.get("content", "").lower()
                
                if query_keywords:
                    # A. Keyword overlap
                    overlap = query_keywords.intersection(chunk_kws)
                    base_kw = len(overlap) / len(query_keywords) if query_keywords else 0.0
                    
                    # B. Name Match Boost
                    name_match_boost = 0.0
                    for qkw in query_keywords:
                        if qkw == chunk_name:
                            name_match_boost = 0.8  # Exact name match is extremely strong
                            break
                        elif qkw in chunk_name:
                            name_match_boost = max(name_match_boost, 0.4) # Partial name match
                    
                    # C. Summary Match Boost
                    summary_match_boost = 0.0
                    summary_overlap = sum(1 for qkw in query_keywords if qkw in chunk_summary)
                    if summary_overlap > 0:
                        summary_match_boost = (summary_overlap / len(query_keywords)) * 0.3 # Max 0.3 boost from summary
                    
                    # D. Route/Content Match Boost
                    # Check if the original route pattern appears in content (e.g., @app.route('/api/keys'))
                    route_match_boost = 0.0
                    if route_patterns:
                        for rp in route_patterns:
                            if rp in chunk_content or rp in chunk_summary:
                                route_match_boost = 0.6  # Strong boost for exact route match in content
                                break
                    
                    # E. Content keyword overlap (fallback for keywords found in content)
                    content_match_boost = 0.0
                    if chunk_content and not overlap:
                        content_hits = sum(1 for qkw in query_keywords if qkw in chunk_content)
                        if content_hits > 0:
                            content_match_boost = (content_hits / len(query_keywords)) * 0.2
                    
                    # Combine Keyword Scores
                    kw_score = min(1.0, base_kw + name_match_boost + summary_match_boost + route_match_boost + content_match_boost)
                
                # 3. Hybrid Combination
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

        # Sort and return top_k
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
