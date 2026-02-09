"""
GitMem Local - Hybrid Retriever

Adaptive Query-Aware Retrieval with Pruning.
Combines semantic and keyword search for optimal memory retrieval.

Paper Reference: Section 3.3 - Adaptive Query-Aware Retrieval with Pruning
Implements:
- Hybrid scoring function S(q, m_k)
- Query Complexity estimation C_q
- Dynamic retrieval depth k_dyn
- Complexity-Aware Pruning
- Global singleton pattern for efficiency
"""

import re
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import concurrent.futures

from .vector_store import LocalVectorStore, get_vector_store
from .embedding import RemoteEmbeddingClient


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval."""
    semantic_top_k: int = 10
    keyword_top_k: int = 10
    final_top_k: int = 5
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    enable_query_expansion: bool = False
    enable_reflection: bool = False
    max_reflection_rounds: int = 2
    min_score_threshold: float = 0.1


# =============================================================================
# Global Singleton Pattern for Hybrid Retriever
# =============================================================================

_retriever_singleton: Optional['HybridRetriever'] = None
_retriever_lock = threading.Lock()


def get_retriever(
    vector_store: LocalVectorStore = None,
    config: RetrievalConfig = None
) -> 'HybridRetriever':
    """
    Get or create the global hybrid retriever singleton.
    
    This avoids repeated initialization overhead.
    
    Args:
        vector_store: Optional vector store (uses global singleton if None)
        config: Optional retrieval config (only used on first creation)
    
    Returns:
        The global HybridRetriever instance
    """
    global _retriever_singleton
    
    if _retriever_singleton is None:
        with _retriever_lock:
            if _retriever_singleton is None:
                vs = vector_store or get_vector_store()
                cfg = config or RetrievalConfig()
                _retriever_singleton = HybridRetriever(vector_store=vs, config=cfg)
    
    return _retriever_singleton


def reset_retriever():
    """Reset the global retriever (for testing or reconfiguration)."""
    global _retriever_singleton
    with _retriever_lock:
        _retriever_singleton = None


class HybridRetriever:
    """
    Hybrid Retriever for adaptive, query-aware memory retrieval.
    
    Combines three retrieval strategies:
    1. Semantic Layer: Dense vector similarity (embeddings)
    2. Lexical Layer: Sparse keyword matching (BM25-like)
    3. Symbolic Layer: Metadata filtering (persons, topics, time)
    
    Features:
    - Query complexity estimation for adaptive depth
    - Multi-query decomposition for comprehensive retrieval
    - Reflection-based additional retrieval
    - Result merging and deduplication
    """
    
    def __init__(
        self,
        vector_store: LocalVectorStore = None,
        config: RetrievalConfig = None
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_store: LocalVectorStore instance
            config: Retrieval configuration
        """
        self.vector_store = vector_store or get_vector_store()
        self.config = config or RetrievalConfig()
    
    def retrieve(
        self,
        agent_id: str,
        query: str,
        memories: List[Dict[str, Any]],
        top_k: int = None,
        enable_reflection: bool = None
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid retrieval with optional query expansion and reflection.
        
        Args:
            agent_id: The agent ID
            query: Search query
            memories: List of memory entries to search
            top_k: Number of results (overrides config)
            enable_reflection: Override reflection setting
        
        Returns:
            List of relevant memories with scores
        """
        top_k = top_k or self.config.final_top_k
        
        if not memories:
            return []
        
        # Analyze query complexity
        query_analysis = self._analyze_query(query)
        
        # Adjust retrieval depth based on complexity
        adjusted_k = self._compute_dynamic_k(query_analysis, top_k)
        
        # Execute hybrid search
        results = self.vector_store.hybrid_search(
            agent_id=agent_id,
            query=query,
            memories=memories,
            top_k=adjusted_k,
            semantic_weight=self.config.semantic_weight,
            keyword_weight=self.config.keyword_weight
        )
        
        # Optional: Apply symbolic layer filtering
        if query_analysis.get("persons") or query_analysis.get("time_filter"):
            results = self._apply_symbolic_filter(results, query_analysis)
        
        # Optional: Reflection-based additional retrieval
        use_reflection = enable_reflection if enable_reflection is not None else self.config.enable_reflection
        if use_reflection and len(results) < top_k:
            results = self._retrieve_with_reflection(
                agent_id, query, memories, results, query_analysis, top_k
            )
        
        return results[:top_k]
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to extract structured information.
        
        Extracts:
        - keywords: Main search terms
        - persons: Person names mentioned
        - time_filter: Time constraints
        - entities: Other named entities
        - complexity: Query complexity score (0-1)
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        # Extract potential person names (capitalized words)
        persons = []
        for word in query.split():
            if word[0].isupper() and len(word) > 1 and word.lower() not in self._get_stopwords():
                persons.append(word)
        
        # Extract time expressions
        time_filter = self._extract_time_filter(query_lower)
        
        # Extract keywords (non-stopwords)
        stopwords = self._get_stopwords()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Estimate complexity
        complexity = self._estimate_complexity(query, keywords, persons, time_filter)
        
        return {
            "keywords": keywords,
            "persons": persons,
            "time_filter": time_filter,
            "entities": [],  # Could be extended with NER
            "complexity": complexity,
            "original_query": query
        }
    
    def _get_stopwords(self) -> set:
        """Get the set of stopwords to filter."""
        return {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
            'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'what', 'when',
            'where', 'why', 'how', 'who', 'which', 'that', 'this', 'these', 'those',
            'and', 'but', 'or', 'if', 'because', 'until', 'while', 'about', 'against',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
            'her', 'it', 'its', 'they', 'them', 'their', 'am', 'all', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'user', 'agent',
            'tell', 'know', 'remember', 'recall', 'find', 'search', 'get', 'show'
        }
    
    def _extract_time_filter(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract time-based filters from query."""
        time_patterns = {
            r'\btoday\b': ('today', timedelta(days=0)),
            r'\byesterday\b': ('yesterday', timedelta(days=1)),
            r'\blast\s+week\b': ('last_week', timedelta(weeks=1)),
            r'\blast\s+month\b': ('last_month', timedelta(days=30)),
            r'\bthis\s+week\b': ('this_week', timedelta(weeks=1)),
            r'\bthis\s+month\b': ('this_month', timedelta(days=30)),
            r'\brecent(ly)?\b': ('recent', timedelta(days=7)),
        }
        
        for pattern, (name, delta) in time_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                now = datetime.now()
                return {
                    "type": name,
                    "start": (now - delta).isoformat(),
                    "end": now.isoformat()
                }
        
        return None
    
    def _estimate_complexity(
        self,
        query: str,
        keywords: List[str],
        persons: List[str],
        time_filter: Optional[Dict]
    ) -> float:
        """
        Estimate query complexity for adaptive retrieval depth.
        
        Paper Reference: Section 3.3 - Query Complexity C_q
        
        Returns:
            Complexity score between 0 and 1
        """
        complexity = 0.0
        
        # Length contribution
        word_count = len(query.split())
        complexity += min(word_count / 20, 0.3)  # Max 0.3 for length
        
        # Keyword diversity
        keyword_count = len(keywords)
        complexity += min(keyword_count / 10, 0.2)  # Max 0.2 for keywords
        
        # Entity complexity
        if persons:
            complexity += min(len(persons) / 5, 0.2)  # Max 0.2 for persons
        
        # Temporal complexity
        if time_filter:
            complexity += 0.1
        
        # Question words indicate higher complexity
        question_words = ['what', 'when', 'where', 'why', 'how', 'who', 'which']
        if any(w in query.lower() for w in question_words):
            complexity += 0.1
        
        # Multi-part questions
        if ' and ' in query.lower() or ',' in query:
            complexity += 0.1
        
        return min(complexity, 1.0)
    
    def _compute_dynamic_k(self, query_analysis: Dict[str, Any], base_k: int) -> int:
        """
        Compute dynamic retrieval depth based on query complexity.
        
        Paper Reference: Section 3.3 - k_dyn = k_base · (1 + δ · C_q)
        
        Args:
            query_analysis: Query analysis result
            base_k: Base number of results
        
        Returns:
            Adjusted k value
        """
        complexity = query_analysis.get("complexity", 0.5)
        delta = 1.0  # Complexity scaling factor
        
        # k_dyn = k_base · (1 + δ · C_q)
        k_dyn = int(base_k * (1 + delta * complexity))
        
        # Clamp to reasonable bounds
        return max(base_k, min(k_dyn, base_k * 3))
    
    def _apply_symbolic_filter(
        self,
        results: List[Dict[str, Any]],
        query_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply symbolic layer filtering based on metadata.
        
        Paper Reference: Section 3.3 - γ · 𝕀(R_k ⊨ C_meta)
        """
        filtered = results.copy()
        
        # Filter by persons
        target_persons = query_analysis.get("persons", [])
        if target_persons:
            target_lower = [p.lower() for p in target_persons]
            scored = []
            for result in filtered:
                memory_persons = [p.lower() for p in result.get("persons", [])]
                person_match = sum(1 for p in target_lower if p in memory_persons)
                if person_match > 0:
                    result = result.copy()
                    # Boost score for person matches
                    result["hybrid_score"] = result.get("hybrid_score", 0) + (0.2 * person_match)
                    scored.append(result)
                else:
                    # Keep but don't boost
                    scored.append(result)
            filtered = scored
        
        # Filter by time
        time_filter = query_analysis.get("time_filter")
        if time_filter:
            start_time = datetime.fromisoformat(time_filter["start"])
            end_time = datetime.fromisoformat(time_filter["end"])
            
            time_filtered = []
            for result in filtered:
                timestamp = result.get("created_at") or result.get("timestamp")
                if timestamp:
                    try:
                        result_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        # Compare dates only
                        if start_time.date() <= result_time.date() <= end_time.date():
                            time_filtered.append(result)
                            continue
                    except (ValueError, AttributeError):
                        pass
                # Include if no timestamp to not lose relevant results
                time_filtered.append(result)
            
            filtered = time_filtered
        
        # Re-sort by updated scores
        filtered.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        
        return filtered
    
    def _retrieve_with_reflection(
        self,
        agent_id: str,
        original_query: str,
        memories: List[Dict[str, Any]],
        initial_results: List[Dict[str, Any]],
        query_analysis: Dict[str, Any],
        target_k: int
    ) -> List[Dict[str, Any]]:
        """
        Execute reflection-based additional retrieval.
        
        If initial results are insufficient, generate additional queries
        to find more relevant information.
        """
        current_results = initial_results.copy()
        seen_ids = {r.get("id") for r in current_results if r.get("id")}
        
        for round_num in range(self.config.max_reflection_rounds):
            # Check if we have enough results
            if len(current_results) >= target_k:
                break
            
            # Generate additional queries based on what's missing
            additional_queries = self._generate_additional_queries(
                original_query, query_analysis, current_results
            )
            
            if not additional_queries:
                break
            
            # Execute additional searches
            for add_query in additional_queries:
                add_results = self.vector_store.hybrid_search(
                    agent_id=agent_id,
                    query=add_query,
                    memories=memories,
                    top_k=target_k,
                    semantic_weight=self.config.semantic_weight,
                    keyword_weight=self.config.keyword_weight
                )
                
                # Merge new results
                for result in add_results:
                    result_id = result.get("id")
                    if result_id and result_id not in seen_ids:
                        # Apply slight penalty for reflection results
                        result["hybrid_score"] = result.get("hybrid_score", 0) * 0.9
                        current_results.append(result)
                        seen_ids.add(result_id)
            
            # Re-sort
            current_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        
        return current_results
    
    def _generate_additional_queries(
        self,
        original_query: str,
        query_analysis: Dict[str, Any],
        current_results: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate additional search queries based on what's missing.
        
        Uses simple heuristics to create variant queries.
        """
        additional = []
        keywords = query_analysis.get("keywords", [])
        
        if not keywords:
            return []
        
        # Try individual important keywords
        for kw in keywords[:3]:
            if len(kw) > 3:
                additional.append(kw)
        
        # Try keyword combinations
        if len(keywords) >= 2:
            additional.append(f"{keywords[0]} {keywords[1]}")
        
        # Try with persons
        persons = query_analysis.get("persons", [])
        if persons:
            for person in persons[:2]:
                additional.append(person)
        
        return additional[:3]  # Limit to 3 additional queries
    
    def search_semantic_only(
        self,
        agent_id: str,
        query: str,
        memories: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Execute pure semantic search."""
        return self.vector_store.semantic_search(
            agent_id=agent_id,
            query=query,
            memories=memories,
            top_k=top_k
        )
    
    def search_keyword_only(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Execute pure keyword search."""
        return self.vector_store.keyword_search(
            query=query,
            memories=memories,
            top_k=top_k
        )


# Global instance (lazy initialization)
_retriever: Optional[HybridRetriever] = None


def get_retriever(
    vector_store: LocalVectorStore = None,
    config: RetrievalConfig = None
) -> HybridRetriever:
    """Get or create the global hybrid retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(vector_store=vector_store, config=config)
    return _retriever
