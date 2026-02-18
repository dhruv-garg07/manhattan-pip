"""
Coding Hybrid Retriever

Handles the retrieval of code chunks using a hybrid approach (Vector + Keyword).
Loads vectors from the dedicated CodingVectorStore (vectors.json) rather than
reading inline vector fields from chunks.
"""
from typing import List, Dict, Any, Optional, Set
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

    # Concept expansion: maps abstract terms to concrete keywords present in chunks
    CONCEPT_MAP = {
        # Security concepts
        "security": {"hash", "sha256", "token", "authentication", "password", "secret", "login", "secure", "permissions", "credentials", "hashed_key"},
        "auth": {"login", "authentication", "password", "token", "session", "oauth", "credentials", "sign_in_with_password"},
        "authentication": {"login", "password", "oauth", "token", "session", "credentials", "sign_in_with_password", "flask-login"},
        "authorization": {"oauth", "permissions", "token", "rls", "role", "login_required"},
        
        # Infrastructure concepts
        "infrastructure": {"flask", "gevent", "socketio", "server", "render", "daemon", "blueprint", "initialization", "monkey_patch", "keep-alive"},
        "devops": {"render", "daemon", "ping", "keep-alive", "server", "health", "timer", "background", "thread"},
        "deployment": {"render", "server", "keep-alive", "daemon", "health", "ping"},
        
        # Data concepts
        "database": {"supabase", "table", "query", "insert", "upsert", "select", "rls", "profiles", "client"},
        "persistence": {"supabase", "database", "table", "insert", "store", "save", "json"},
        "storage": {"supabase", "database", "json", "file", "upload", "save", "memory", "rag"},
        
        # UX concepts
        "onboarding": {"register", "signup", "login", "profile", "dashboard", "welcome"},
        "experience": {"user", "page", "template", "render", "form", "dashboard", "explore"},
        
        # Validation concepts
        "validation": {"validate", "regex", "email", "password", "check", "form", "username_unique", "whitelist", "extension_validation"},
        "sanitization": {"strip", "lower", "clean", "validate", "escape", "html_escape", "regex"},
        
        # API concepts
        "api": {"endpoint", "route", "json", "post", "get", "request", "response", "api_key", "rest"},
        "rest": {"endpoint", "route", "json", "post", "get", "request", "response"},
        "json": {"jsonify", "json", "response", "api", "parse", "data"},
        "response": {"return", "jsonify", "render_template", "redirect", "flash"},
        "redirect": {"redirect", "url_for", "flash", "agent_detail", "login_google"},
    }
    
    def _expand_concepts(self, keywords: Set[str]) -> Set[str]:
        """Expand abstract concept keywords into concrete searchable terms."""
        expanded = set(keywords)
        for kw in keywords:
            if kw in self.CONCEPT_MAP:
                expanded.update(self.CONCEPT_MAP[kw])
        return expanded

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

    @staticmethod
    def _decompose_compound(token: str) -> Set[str]:
        """
        Decompose compound identifiers into sub-tokens.
        e.g., 'login_required' -> {'login', 'required'}
              'camelCase' -> {'camel', 'case'}
              '@decorator' -> {'decorator'}
        """
        parts = set()
        # Strip common prefixes
        clean = token.lstrip('@').strip(string.punctuation)
        if not clean:
            return parts
        
        # Split on underscores
        if '_' in clean:
            for seg in clean.split('_'):
                seg = seg.lower().strip()
                if len(seg) > 1:
                    parts.add(seg)
        
        # Split camelCase
        camel_parts = re.findall(r'[a-z]+|[A-Z][a-z]*|[A-Z]+(?=[A-Z][a-z]|$)', clean)
        for cp in camel_parts:
            cp = cp.lower().strip()
            if len(cp) > 1:
                parts.add(cp)
        
        # Always include the original (lowered)
        parts.add(clean.lower())
        return parts

    @staticmethod
    def _extract_line_numbers(query: str) -> List[int]:
        """Extract line numbers from query like 'line 447' or 'at line 100'."""
        line_nums = []
        # Pattern: 'line <number>' or 'line: <number>' or 'lines <n>-<m>'
        for m in re.finditer(r'\blines?\s*:?\s*(\d+)', query, re.IGNORECASE):
            line_nums.append(int(m.group(1)))
        # Also catch standalone 3+ digit numbers that look like line references
        for m in re.finditer(r'\b(\d{3,})\b', query):
            num = int(m.group(1))
            if num < 50000:  # Reasonable line number
                line_nums.append(num)
        return line_nums

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
        STOP_WORDS = {"is", "the", "a", "an", "and", "or", "in", "on", "with", 
                      "how", "what", "where", "to", "for", "of", "can", "you", 
                      "explain", "from", "pull", "context", "only", "which",
                      "does", "do", "that", "this", "it", "but", "not", "at",
                      "by", "are", "be", "been", "has", "have", "was", "were"}
        
        raw_tokens = query.split()
        query_keywords = set()
        for w in raw_tokens:
            cleaned = w.lower().strip(string.punctuation + "'\"")
            if cleaned and cleaned not in STOP_WORDS:
                query_keywords.add(cleaned)
                # Also decompose compound tokens
                query_keywords.update(self._decompose_compound(cleaned))
        query_keywords.discard("")
        
        # Expand abstract concepts into concrete keywords
        query_keywords = self._expand_concepts(query_keywords)
        
        # Extract line numbers for positional matching
        query_line_numbers = self._extract_line_numbers(query)
        
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
                # Build expanded keyword set for chunk (includes decomposed form)
                chunk_kws = set()
                for kw in chunk.get("keywords", []):
                    chunk_kws.add(kw.lower())
                    chunk_kws.update(self._decompose_compound(kw))
                
                chunk_name = chunk.get("name", "").lower()
                chunk_name_parts = self._decompose_compound(chunk_name)
                chunk_summary = chunk.get("summary", "").lower()
                chunk_content = chunk.get("content", "").lower()
                chunk_type = chunk.get("type", "").lower()
                
                # Tokenize summary into words for precise matching
                summary_words = set(re.findall(r'[a-z0-9_]+', chunk_summary))
                
                if query_keywords:
                    # A. Keyword overlap (with decomposed keywords)
                    overlap = query_keywords.intersection(chunk_kws)
                    base_kw = len(overlap) / len(query_keywords) if query_keywords else 0.0
                    
                    # B. Name Match Boost
                    name_match_boost = 0.0
                    for qkw in query_keywords:
                        if qkw == chunk_name:
                            name_match_boost = 0.8  # Exact name match
                            break
                        elif qkw in chunk_name_parts:
                            name_match_boost = max(name_match_boost, 0.5)  # Part of name
                        elif qkw in chunk_name:
                            name_match_boost = max(name_match_boost, 0.4)  # Substring of name
                    
                    # C. Summary Match Boost (word-level, not substring)
                    summary_match_boost = 0.0
                    summary_overlap = query_keywords.intersection(summary_words)
                    if summary_overlap:
                        summary_match_boost = (len(summary_overlap) / len(query_keywords)) * 0.35
                    
                    # D. Route/Content Match Boost
                    route_match_boost = 0.0
                    if route_patterns:
                        for rp in route_patterns:
                            if rp in chunk_content or rp in chunk_summary:
                                route_match_boost = 0.6
                                break
                    
                    # E. Content keyword overlap (ALWAYS checked, not just fallback)
                    content_match_boost = 0.0
                    if chunk_content:
                        content_words = set(re.findall(r'[a-z0-9_]+', chunk_content))
                        content_hits = query_keywords.intersection(content_words)
                        if content_hits:
                            # Scale: more hits = more boost, but cap at 0.25
                            content_match_boost = min(0.25, (len(content_hits) / len(query_keywords)) * 0.25)
                    
                    # F. Type Match Boost (query mentions 'class', 'function', etc.)
                    type_match_boost = 0.0
                    type_query_words = {"class", "function", "module", "block", "import",
                                        "classes", "functions", "modules", "blocks"}
                    query_type_mentions = query_keywords.intersection(type_query_words)
                    if query_type_mentions and chunk_type:
                        # Check if chunk type matches any type word in query
                        for qt in query_type_mentions:
                            if qt.rstrip('s') == chunk_type or qt == chunk_type:
                                type_match_boost = 0.2
                                break
                    
                    # Combine Keyword Scores
                    kw_score = min(1.0, base_kw + name_match_boost + summary_match_boost 
                                   + route_match_boost + content_match_boost + type_match_boost)
                
                # 3. Line Number Matching
                line_match_boost = 0.0
                if query_line_numbers:
                    start_line = chunk.get("start_line", 0)
                    end_line = chunk.get("end_line", 0)
                    if start_line or end_line:
                        for ln in query_line_numbers:
                            if start_line <= ln <= end_line:
                                line_match_boost = 0.9  # Very strong: exact line match
                                break
                            elif abs(ln - start_line) <= 10 or abs(ln - end_line) <= 10:
                                line_match_boost = max(line_match_boost, 0.4)  # Near match
                
                # 4. Hybrid Combination
                final_score = (vec_score * alpha) + (kw_score * (1.0 - alpha)) + line_match_boost
                if final_score > 0.01:  # Threshold to filter noise
                    # Filter out full content, keep only summary and metadata
                    summary_chunk = {k: v for k, v in chunk.items() if k != "hash_id" and k != "embedding_id"}
                    
                    scored_chunks.append({
                        "file_path": file_path,
                        "chunk": summary_chunk,
                        "score": final_score,
                        "match_type": "hybrid",
                        "vector_score": vec_score,
                        "keyword_score": kw_score
                    })

        # Sort and return top_k
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
