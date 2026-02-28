"""
Coding Hybrid Retriever – v2 (Optimized for Q&A about code)

Handles retrieval of code chunks using a multi-signal hybrid approach:
  1. Vector similarity  (semantic)
  2. Keyword overlap     (lexical)
  3. Summary NLP match   (natural-language overlap)
  4. Content grep match  (exact symbol/pattern match)
  5. Name / Type match   (structural)
  6. Line-number proximity
  7. Parent-child class/method boost
  8. Query-intent detection (broad vs narrow)

Loads vectors from the dedicated CodingVectorStore (vectors.json).
"""
from typing import List, Dict, Any, Optional, Set, Tuple
import os
import re
import json
import math
import string
from collections import Counter
from ..gitmem.embedding import RemoteEmbeddingClient
from .coding_store import CodingContextStore
from .coding_vector_store import CodingVectorStore
from .ast_skeleton import retrieve_path
import logging

logger = logging.getLogger(__name__)


# ─── Helpers ────────────────────────────────────────────────────────────────

_STOP_WORDS: Set[str] = {
    "is", "the", "a", "an", "and", "or", "in", "on", "with",
    "how", "what", "where", "to", "for", "of", "can", "you",
    "explain", "from", "pull", "context", "only", "which",
    "does", "do", "that", "this", "it", "but", "not", "at",
    "by", "are", "be", "been", "has", "have", "was", "were",
    "show", "me", "all", "about", "tell", "describe", "give",
    "find", "list", "get", "there", "when", "if", "then",
    "so", "also", "my", "i", "we", "us", "our", "your",
    "its", "any", "some", "each", "every", "just", "need",
    "want", "like", "use", "used", "using", "would", "could",
    "should", "will", "shall", "may", "might", "must",
}

# Broad-intent signals: queries containing these words ask for multiple results
_BROAD_INTENT_SIGNALS = {
    "all", "every", "list", "show", "related", "overview",
    "relevant", "associated", "connected", "involved",
}


class CodingHybridRetriever:
    """
    Centralized retrieval logic for coding contexts.
    Combines semantic search (Vectors) and multi-signal exact matching.
    """

    def __init__(
        self,
        store: CodingContextStore,
        vector_store: CodingVectorStore,
        embedding_client: Optional[RemoteEmbeddingClient] = None,
    ):
        self.store = store
        self.vector_store = vector_store
        self.embedding_client = embedding_client or vector_store.embedding_client

    # =====================================================================
    # Concept expansion (abstract → concrete)
    # =====================================================================
    CONCEPT_MAP: Dict[str, Set[str]] = {
        # Security
        "security": {
            "hash", "sha256", "token", "authentication", "password", "secret",
            "login", "secure", "permissions", "credentials", "hashed_key",
            "bcrypt", "jwt", "rate_limit", "throttle", "validation", "oauth",
            "encrypt", "decrypt", "ssl", "tls", "certificate", "verify",
            "authorize", "authorisation", "authorization",
        },
        "auth": {
            "login", "authentication", "password", "token", "session", "oauth",
            "credentials", "sign_in_with_password", "jwt", "bcrypt", "verify",
        },
        "authentication": {
            "login", "password", "oauth", "token", "session", "credentials",
            "sign_in_with_password", "flask-login", "jwt", "bcrypt",
        },
        "authorization": {
            "oauth", "permissions", "token", "rls", "role", "login_required",
        },

        # Infrastructure
        "infrastructure": {
            "flask", "gevent", "socketio", "server", "render", "daemon",
            "blueprint", "initialization", "monkey_patch", "keep-alive",
        },
        "devops": {
            "render", "daemon", "ping", "keep-alive", "server", "health",
            "timer", "background", "thread",
        },
        "deployment": {
            "render", "server", "keep-alive", "daemon", "health", "ping",
        },

        # Data
        "database": {
            "supabase", "table", "query", "insert", "upsert", "select", "rls",
            "profiles", "client", "migration", "schema", "sql", "engine",
            "rollback", "sqlalchemy",
        },
        "persistence": {
            "supabase", "database", "table", "insert", "store", "save", "json",
        },
        "storage": {
            "supabase", "database", "json", "file", "upload", "save", "memory",
            "rag", "cache", "redis",
        },
        "schema": {
            "migration", "table", "column", "alter", "create", "drop",
            "rollback", "version_control", "up_sql", "down_sql",
        },

        # Caching / Performance
        "cache": {
            "memoize", "memoization", "ttl", "decorator", "redis", "lru",
            "caching", "in_memory", "performance", "cache_result",
        },
        "performance": {
            "cache", "memoize", "ttl", "optimize", "speed", "latency",
            "profiling", "benchmark", "async", "concurrent",
        },
        "memoization": {
            "cache", "memoize", "decorator", "ttl", "lru", "caching",
            "cache_result", "performance", "in_memory",
        },

        # UX
        "onboarding": {
            "register", "signup", "login", "profile", "dashboard", "welcome",
        },
        "experience": {
            "user", "page", "template", "render", "form", "dashboard", "explore",
        },

        # Validation
        "validation": {
            "validate", "regex", "email", "password", "check", "form",
            "username_unique", "whitelist", "extension_validation", "pattern",
            "format",
        },
        "sanitization": {
            "strip", "lower", "clean", "validate", "escape", "html_escape",
            "regex",
        },

        # API
        "api": {
            "endpoint", "route", "json", "post", "get", "request", "response",
            "api_key", "rest", "rate_limit", "throttle", "middleware",
        },
        "rest": {
            "endpoint", "route", "json", "post", "get", "request", "response",
        },
        "json": {
            "jsonify", "json", "response", "api", "parse", "data",
        },
        "response": {
            "return", "jsonify", "render_template", "redirect", "flash",
        },
        "redirect": {
            "redirect", "url_for", "flash", "agent_detail", "login_google",
        },

        # Rate limiting
        "rate_limit": {
            "throttle", "ratelimiter", "redis", "middleware", "sliding_window",
            "max_requests", "client_ip", "api",
        },
        "throttle": {
            "rate_limit", "ratelimiter", "redis", "middleware", "api",
        },
    }

    # Synonym pairs: if query contains left word, also consider right words
    SYNONYM_MAP: Dict[str, Set[str]] = {
        "forget": {"reset", "forgot", "recover", "lost"},
        "forgot": {"reset", "forget", "recover", "lost"},
        "protect": {"security", "rate_limit", "throttle", "guard", "defend"},
        "overwhelm": {"rate_limit", "throttle", "ddos", "overload", "flood"},
        "expire": {"expiry", "ttl", "refresh", "timeout", "renew"},
        "expires": {"expiry", "ttl", "refresh", "timeout", "renew"},
        "renew": {"refresh", "expiry", "extend", "token"},
        "version": {"migration", "version_control", "rollback", "schema"},
        "versioned": {"migration", "version_control", "rollback", "schema"},
        "memoize": {"cache", "caching", "memoization", "decorator", "ttl"},
        "optimize": {"performance", "cache", "speed", "efficient"},
        "manage": {"manager", "handler", "controller", "orchestrator"},
        "managed": {"manager", "handler", "controller", "orchestrator"},
        "interact": {"query", "insert", "update", "delete", "call", "invoke"},
        "change": {"update", "modify", "alter", "migration", "edit"},
        "changes": {"update", "modify", "alter", "migration", "edit"},
        "expensive": {"performance", "cache", "optimize", "slow", "heavy"},
    }

    # =====================================================================
    # Public search interface
    # =====================================================================

    def search(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        hybrid_alpha: float = 0.55,
        file_paths_filter: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform a hybrid search for the query with metadata filtering.

        Args:
            agent_id: The agent identifier.
            query: Natural-language question about the code.
            top_k: Max results to return.
            hybrid_alpha: Weight for vector score (1-alpha goes to keyword).
        """
        # 1. Metadata Filtering Extraction
        file_filter = None
        clean_query = query
        route_tokens: List[str] = []

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
            _, ext = os.path.splitext(clean_token)
            if ext.lower() in FILE_EXTENSIONS:
                file_filter = clean_token
                clean_query = query.replace(token, "").strip()
                break
            if '/' in clean_token and not ext:
                route_tokens.append(clean_token)

        if not clean_query:
            clean_query = "summary overview"

        # 2. Detect broad vs narrow intent
        query_lower_set = set(clean_query.lower().split())
        is_broad = bool(query_lower_set & _BROAD_INTENT_SIGNALS)
        effective_top_k = max(top_k, 8) if is_broad else top_k

        # 3. Search
        results = self._hybrid_search_chunks(
            agent_id, clean_query, effective_top_k, hybrid_alpha,
            file_filter=file_filter,
            route_tokens=route_tokens,
            is_broad=is_broad,
            file_paths_filter=file_paths_filter,
        )

        return {
            "status": "search_results",
            "query": clean_query,
            "filter": file_filter,
            "results": results,
            "count": len(results),
        }

    # =====================================================================
    # Keyword decomposition helpers
    # =====================================================================

    @staticmethod
    def _decompose_compound(token: str) -> Set[str]:
        """
        Decompose compound identifiers into sub-tokens.
        e.g., 'login_required' -> {'login', 'required'}
              'camelCase'      -> {'camel', 'case'}
              '@decorator'     -> {'decorator'}
        """
        parts: Set[str] = set()
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

        parts.add(clean.lower())
        return parts

    @staticmethod
    def _extract_line_numbers(query: str) -> List[int]:
        """Extract line numbers from query like 'line 447' or 'at line 100'."""
        line_nums: List[int] = []
        for m in re.finditer(r'\blines?\s*:?\s*(\d+)', query, re.IGNORECASE):
            line_nums.append(int(m.group(1)))
        for m in re.finditer(r'\b(\d{3,})\b', query):
            num = int(m.group(1))
            if num < 50000:
                line_nums.append(num)
        return line_nums

    # =====================================================================
    # Concept / synonym expansion
    # =====================================================================

    def _expand_concepts(self, keywords: Set[str]) -> Set[str]:
        """Expand abstract concept keywords into concrete searchable terms."""
        expanded = set(keywords)
        for kw in keywords:
            if kw in self.CONCEPT_MAP:
                expanded.update(self.CONCEPT_MAP[kw])
            if kw in self.SYNONYM_MAP:
                expanded.update(self.SYNONYM_MAP[kw])
        return expanded

    # =====================================================================
    # Query-intent helpers
    # =====================================================================

    @staticmethod
    def _detect_query_intent(query: str) -> str:
        """
        Detect the intent category of a query to fine-tune scoring.
        Returns one of: 'symbol', 'concept', 'usage', 'line', 'broad'.
        """
        q = query.lower()
        if re.search(r'\bline\s*\d+', q):
            return "line"
        if re.search(r'\b(all|every|list|show me|overview)\b', q):
            return "broad"
        if re.search(r'\b(how is|where is|which.*use|interact|call)\b', q):
            return "usage"
        # Check if query is mainly a single identifier
        content_words = [w for w in q.split() if w not in _STOP_WORDS]
        if len(content_words) <= 2:
            return "symbol"
        return "concept"

    # =====================================================================
    # Core hybrid search
    # =====================================================================

    def _hybrid_search_chunks(
        self,
        agent_id: str,
        query: str,
        top_k: int,
        alpha: float,
        file_filter: str = None,
        route_tokens: List[str] = None,
        is_broad: bool = False,
        file_paths_filter: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute multi-signal hybrid search."""

        intent = self._detect_query_intent(query)

        # ── A. Query representations ────────────────────────────────────

        # Vector
        query_vector: List[float] = []
        try:
            query_vector = self.embedding_client.embed(query)
            if hasattr(query_vector, 'tolist'):
                query_vector = query_vector.tolist()
        except Exception as e:
            logger.warning(f"Embedding generation failed for query '{query}': {e}")

        # Keywords
        raw_tokens = query.split()
        query_keywords: Set[str] = set()
        for w in raw_tokens:
            cleaned = w.lower().strip(string.punctuation + "'\"")
            if cleaned and cleaned not in _STOP_WORDS:
                query_keywords.add(cleaned)
                query_keywords.update(self._decompose_compound(cleaned))
        query_keywords.discard("")

        # Save original (pre-expansion) for IDF weighting
        original_query_keywords = set(query_keywords)

        # Expand
        query_keywords = self._expand_concepts(query_keywords)

        # Line numbers
        query_line_numbers = self._extract_line_numbers(query)

        # Route tokens
        route_patterns: List[str] = []
        if route_tokens:
            for rt in route_tokens:
                clean_rt = rt.strip(string.punctuation + "'\"")
                route_patterns.append(clean_rt)
                segments = [seg for seg in clean_rt.split('/') if seg and seg not in _STOP_WORDS]
                query_keywords.update(s.lower() for s in segments)

        # ── B. Build corpus-level IDF (per search, lightweight) ─────────

        contexts = self.store._load_agent_data(agent_id, "file_contexts")
        all_vectors = self.vector_store._load_vectors(agent_id) or {}

        all_chunks_flat: List[Dict[str, Any]] = []
        chunk_to_ctx: List[Tuple[Dict[str, Any], str]] = []  # (chunk, file_path)
        
        normalized_file_paths_filter = set(os.path.normpath(fp) for fp in file_paths_filter) if file_paths_filter else None

        for ctx in contexts:
            file_path = ctx.get("file_path", "")
            
            # Apply file_paths_filter if provided
            if normalized_file_paths_filter is not None:
                if os.path.normpath(file_path) not in normalized_file_paths_filter:
                    continue
                    
            if file_filter:
                normalized_filter = file_filter.replace("/", os.sep).replace("\\", os.sep).lower()
                path_lower = file_path.lower()
                if normalized_filter not in path_lower:
                    if os.path.basename(normalized_filter) not in os.path.basename(path_lower):
                        continue
            for chunk in ctx.get("chunks", []):
                all_chunks_flat.append(chunk)
                chunk_to_ctx.append((chunk, file_path))

        total_docs = max(len(all_chunks_flat), 1)

        # Build DF (document frequency) for keywords from original query
        doc_freq: Counter = Counter()
        for chunk in all_chunks_flat:
            chunk_text_bag = self._get_chunk_text_bag(chunk)
            for kw in original_query_keywords:
                if kw in chunk_text_bag:
                    doc_freq[kw] += 1

        def idf(term: str) -> float:
            df = doc_freq.get(term, 0)
            return math.log((total_docs + 1) / (df + 1)) + 1.0

        # ── C. Build parent-class index for relationship boosts ─────────

        # Map: class_name -> list of method chunk indices
        class_names: Set[str] = set()
        for chunk in all_chunks_flat:
            if chunk.get("type", "").lower() in ("class",):
                class_names.add(chunk.get("name", "").lower())

        # ── D. Score each chunk ─────────────────────────────────────────

        scored_chunks: List[Dict[str, Any]] = []

        for chunk, file_path in chunk_to_ctx:
            # 1. Vector Score
            vec_score = 0.0
            hash_id = chunk.get("hash_id") or chunk.get("embedding_id")
            chunk_vec = all_vectors.get(hash_id) if hash_id else None

            if query_vector and chunk_vec and len(chunk_vec) > 0:
                try:
                    dot = sum(a * b for a, b in zip(query_vector, chunk_vec))
                    # Normalise to [0, 1] via cosine (vectors should be pre-normalised,
                    # but guard anyway)
                    mag_q = math.sqrt(sum(a * a for a in query_vector)) or 1.0
                    mag_c = math.sqrt(sum(b * b for b in chunk_vec)) or 1.0
                    vec_score = max(0.0, dot / (mag_q * mag_c))
                except Exception:
                    vec_score = 0.0

            # 2. Keyword Score (multi-signal)
            kw_score = 0.0

            chunk_kws: Set[str] = set()
            for kw in chunk.get("keywords", []):
                chunk_kws.add(kw.lower())
                chunk_kws.update(self._decompose_compound(kw))

            chunk_name = chunk.get("name", "").lower()
            chunk_name_parts = self._decompose_compound(chunk_name)
            chunk_summary = chunk.get("summary", "").lower()
            chunk_content = chunk.get("content", "").lower()
            chunk_type = chunk.get("type", "").lower()

            summary_words = set(re.findall(r'[a-z0-9_]+', chunk_summary))
            content_words = set(re.findall(r'[a-z0-9_]+', chunk_content)) if chunk_content else set()

            # Union of all searchable text for this chunk
            chunk_text_bag = chunk_kws | summary_words | content_words | chunk_name_parts | {chunk_name}

            if query_keywords:
                # A. IDF-weighted keyword overlap
                overlap = query_keywords & chunk_text_bag
                if overlap:
                    idf_sum = sum(idf(t) for t in overlap if t in original_query_keywords)
                    idf_total = sum(idf(t) for t in original_query_keywords) or 1.0
                    # Also credit expanded-keyword matches (lower weight)
                    expanded_only = overlap - original_query_keywords
                    expanded_credit = len(expanded_only) * 0.3
                    base_kw = min(1.0, (idf_sum / idf_total) * 0.7 + expanded_credit * 0.1)
                else:
                    base_kw = 0.0

                # B. Name Match Boost
                name_match_boost = 0.0
                for qkw in query_keywords:
                    if qkw == chunk_name:
                        name_match_boost = 0.8
                        break
                    elif qkw in chunk_name_parts:
                        name_match_boost = max(name_match_boost, 0.5)
                    elif qkw in chunk_name:
                        name_match_boost = max(name_match_boost, 0.4)

                # C. Summary sentence overlap (percentage of query words in summary)
                summary_overlap = original_query_keywords & summary_words
                summary_match_boost = 0.0
                if summary_overlap:
                    summary_match_boost = (len(summary_overlap) / len(original_query_keywords)) * 0.4

                # D. Content grep: check if the *original* query tokens appear in content
                content_match_boost = 0.0
                if chunk_content:
                    content_hits = original_query_keywords & content_words
                    if content_hits:
                        content_match_boost = min(0.35, (len(content_hits) / len(original_query_keywords)) * 0.35)

                # E. Route/pattern match
                route_match_boost = 0.0
                if route_patterns:
                    for rp in route_patterns:
                        if rp in chunk_content or rp in chunk_summary:
                            route_match_boost = 0.6
                            break

                # F. Type match
                type_match_boost = 0.0
                type_query_words = {"class", "function", "module", "block", "import",
                                    "classes", "functions", "modules", "blocks", "method", "methods"}
                query_type_mentions = query_keywords & type_query_words
                if query_type_mentions and chunk_type:
                    for qt in query_type_mentions:
                        if qt.rstrip('s') == chunk_type or qt == chunk_type:
                            type_match_boost = 0.2
                            break

                # G. Parent-class relationship boost
                # If query matches a class, boost its methods too
                relationship_boost = 0.0
                if '.' in chunk_name:
                    parent_class = chunk_name.split('.')[0]
                    if parent_class in query_keywords or any(
                        qkw in parent_class for qkw in query_keywords if len(qkw) > 3
                    ):
                        relationship_boost = 0.3

                # H. Exact phrase / bigram match in summary
                bigram_boost = 0.0
                query_lower = query.lower()
                # Check 2-word subsequences from query appearing in summary
                q_words = [w for w in query_lower.split() if w not in _STOP_WORDS]
                for i in range(len(q_words) - 1):
                    bigram = q_words[i] + " " + q_words[i + 1]
                    if bigram in chunk_summary:
                        bigram_boost = 0.25
                        break

                # Combine
                kw_score = min(1.0,
                    base_kw
                    + name_match_boost
                    + summary_match_boost
                    + content_match_boost
                    + route_match_boost
                    + type_match_boost
                    + relationship_boost
                    + bigram_boost
                )

            # 3. Line Number Matching (prefer tight ranges)
            line_match_boost = 0.0
            if query_line_numbers:
                start_line = chunk.get("start_line", 0)
                end_line = chunk.get("end_line", 0)
                if start_line or end_line:
                    span = max(end_line - start_line, 1)
                    for ln in query_line_numbers:
                        if start_line <= ln <= end_line:
                            # Tighter span → higher boost (inverse span reward)
                            tightness = min(1.0, 20.0 / span)
                            score = 0.5 + 0.4 * tightness  # 0.5 – 0.9
                            line_match_boost = max(line_match_boost, score)
                        elif abs(ln - start_line) <= 10 or abs(ln - end_line) <= 10:
                            line_match_boost = max(line_match_boost, 0.3)

            # 4. Final hybrid combination
            # Adjust alpha based on intent
            effective_alpha = alpha
            if intent == "symbol":
                effective_alpha = 0.35  # lean towards keyword matching
            elif intent == "usage":
                effective_alpha = 0.40  # content/keyword heavy
            elif intent == "broad":
                effective_alpha = 0.45  # balanced but more keyword
            elif intent == "line":
                effective_alpha = 0.30  # line match dominates

            final_score = (
                vec_score * effective_alpha
                + kw_score * (1.0 - effective_alpha)
                + line_match_boost
            )

            # Minimum threshold
            if final_score > 0.01:
                summary_chunk = {
                    k: v for k, v in chunk.items()
                    if k not in ("hash_id", "embedding_id", "vector")
                }

                scored_chunks.append({
                    "file_path": file_path,
                    "chunk": summary_chunk,
                    "score": round(final_score, 6),
                    "match_type": "hybrid",
                    "vector_score": round(vec_score, 6),
                    "keyword_score": round(kw_score, 6),
                })

        # Sort and return
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    # =====================================================================
    # Internal helpers
    # =====================================================================

    @staticmethod
    def _get_chunk_text_bag(chunk: Dict[str, Any]) -> Set[str]:
        """Build a set of all lowercase tokens in a chunk (keywords + summary + content + name)."""
        bag: Set[str] = set()
        for kw in chunk.get("keywords", []):
            bag.add(kw.lower())
        bag.update(re.findall(r'[a-z0-9_]+', chunk.get("summary", "").lower()))
        bag.update(re.findall(r'[a-z0-9_]+', chunk.get("content", "").lower()))
        name = chunk.get("name", "").lower()
        if name:
            bag.add(name)
            bag.update(CodingHybridRetriever._decompose_compound(name))
        return bag
