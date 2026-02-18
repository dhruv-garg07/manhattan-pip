"""
Coding Memory Builder – v2 (Optimized for Q&A Retrieval)

Handles the ingestion of code chunks:
  1. Generates rich embedding text that captures:
     - Semantic meaning (summary)
     - Structural context (type, name, parent class)
     - Searchable patterns  (decorators, routes, SQL tables)
     - Inferred Q&A pairs   (anticipate how a developer would ask)
     - Keywords & synonyms
  2. Stores vector embeddings in CodingVectorStore (vectors.json).
  3. Persists chunks (without inline vectors) to CodingContextStore.
"""
from typing import List, Dict, Any, Optional
import os
import uuid
import re
import hashlib
from .models import CodeChunk
from ..gitmem.embedding import RemoteEmbeddingClient
from .coding_store import CodingContextStore
from .coding_vector_store import CodingVectorStore
import logging

logger = logging.getLogger(__name__)


class CodingMemoryBuilder:
    """
    Orchestrates the ingestion of code contexts.

    Responsibilities:
    1. Accepts code chunks (from file or pre-computed).
    2. Enriches keyword sets with inferred terms (table names, decorators, etc.)
    3. Ensures every chunk has a vector embedding stored in CodingVectorStore.
    4. Persists the chunks (without inline vectors) to CodingContextStore.
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
    # Public entry point
    # =====================================================================

    def process_file_chunks(
        self,
        agent_id: str,
        file_path: str,
        chunks: List[Dict[str, Any]],
        language: str = "auto",
        session_id: str = "",
    ) -> Dict[str, Any]:
        """
        Process and store chunks for a file.
        Ensures all chunks have embeddings in vectors.json before storage.
        """
        # Phase 1: Preparation & Deduplication
        chunks_to_embed: List[str] = []
        embedding_indices: List[int] = []
        enriched_chunks: List[Dict[str, Any]] = []

        for i, chunk_data in enumerate(chunks):
            chunk = chunk_data.copy()

            # ── Auto-enrich keywords before hashing ──────────────────
            chunk = self._enrich_keywords(chunk)

            # ── Ensure hash_id exists ────────────────────────────────
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

            if hash_id:
                chunk["embedding_id"] = hash_id

            chunk.pop("vector", None)
            enriched_chunks.append(chunk)

        # Phase 2: Batch Embedding
        if chunks_to_embed:
            try:
                vectors = self.embedding_client.embed_batch(chunks_to_embed)
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
            session_id=session_id,
        )

    # =====================================================================
    # Keyword enrichment
    # =====================================================================

    @staticmethod
    def _enrich_keywords(chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-infer additional keywords from chunk content and metadata.
        This makes keyword-based retrieval much more accurate.
        """
        keywords = set(chunk.get("keywords", []))
        content = chunk.get("content", "")
        name = chunk.get("name", "")
        summary = chunk.get("summary", "")
        chunk_type = chunk.get("type", "")

        # 1. Decompose the name
        if name:
            keywords.add(name)
            # Split camelCase and snake_case
            parts = re.findall(r'[a-z]+|[A-Z][a-z]*|[A-Z]+(?=[A-Z][a-z]|$)', name)
            for p in parts:
                if len(p) > 1:
                    keywords.add(p.lower())
            if '_' in name:
                for seg in name.split('_'):
                    if len(seg) > 1:
                        keywords.add(seg.lower())
            # Parent.method notation
            if '.' in name:
                for part in name.split('.'):
                    keywords.add(part.lower())
                    # Decompose parent too
                    sub_parts = re.findall(r'[a-z]+|[A-Z][a-z]*', part)
                    for sp in sub_parts:
                        if len(sp) > 1:
                            keywords.add(sp.lower())

        # 2. Extract patterns from content
        if content:
            # Decorators
            decorators = re.findall(r'@(\w+[\.\w]*)', content)
            keywords.update(d.lower() for d in decorators)

            # DB table references: .query('table'), .insert('table'), etc.
            tables = re.findall(r"\.(?:query|insert|update|delete|select|from)\s*\(\s*['\"](\w+)['\"]", content)
            keywords.update(t.lower() for t in tables)
            # Also mark these as table names for better matching
            for t in tables:
                keywords.add(f"{t.lower()}_table")  # Helps queries like "users table"

            # Flask/HTTP routes
            routes = re.findall(r"@\w+\.route\s*\(\s*['\"]([^'\"]+)['\"]", content)
            for route in routes:
                keywords.add(route.lower())
                segments = [s for s in route.split('/') if s]
                keywords.update(s.lower() for s in segments)

            # HTTP methods
            methods = re.findall(r"methods\s*=\s*\[([^\]]+)\]", content)
            for m_group in methods:
                for method in re.findall(r"'(\w+)'", m_group):
                    keywords.add(method.lower())

            # Import references
            imports = re.findall(r'(?:from|import)\s+([\w.]+)', content)
            for imp in imports:
                keywords.add(imp.split('.')[-1].lower())

            # Exception types
            exceptions = re.findall(r'(?:raise|except)\s+(\w+)', content)
            keywords.update(e.lower() for e in exceptions)

            # Library/function calls: specific patterns
            calls = re.findall(r'(\w+)\.(\w+)\s*\(', content)
            for obj, method in calls:
                if len(obj) > 1 and obj.lower() not in ('self', 'cls'):
                    keywords.add(obj.lower())
                if len(method) > 1:
                    keywords.add(method.lower())

        # 3. Extract from summary
        if summary:
            # Important nouns/verbs from summary
            # Look for capitalized terms (likely class/type refs)
            caps = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)\b', summary)
            keywords.update(c.lower() for c in caps)
            # Also catch technical terms
            tech_terms = re.findall(r'\b(?:JWT|API|SQL|HTTP|REST|OAuth|TTL|Redis|CORS)\b', summary, re.IGNORECASE)
            keywords.update(t.lower() for t in tech_terms)

        # 4. Add chunk type as keyword
        if chunk_type:
            keywords.add(chunk_type.lower())

        # Filter out very short / noise keywords
        keywords = {kw for kw in keywords if len(kw) > 1 and kw not in ('self', 'cls', 'none', 'true', 'false')}

        chunk["keywords"] = list(keywords)
        return chunk

    # =====================================================================
    # Embedding text preparation
    # =====================================================================

    def _prepare_embedding_text(self, chunk: Dict[str, Any]) -> str:
        """
        Prepare the text to be embedded for a chunk.

        Strategy (optimized for Q&A retrieval):
        1. Name & Type header          → direct name queries
        2. Summary                     → semantic meaning
        3. Content snippet             → exact code patterns
        4. Inferred Q&A pairs          → anticipate developer questions
        5. Extracted patterns           → decorators, routes, tables
        6. Keywords                    → concept matching
        7. Line range                  → positional queries
        """
        text_parts: List[str] = []

        name = chunk.get("name", "")
        type_ = chunk.get("type", "")
        summary = chunk.get("summary", "")
        content = chunk.get("content", "")
        keywords = chunk.get("keywords", [])

        # 1. Name & Type header
        if name or type_:
            header = f"{type_} {name}".strip()
            text_parts.append(header)
            # Add parent context if method
            if '.' in name:
                parent, method = name.rsplit('.', 1)
                text_parts.append(f"Method {method} of class {parent}")

        # 2. Summary (primary semantic signal)
        if summary:
            text_parts.append(summary)

        # 3. Content snippet (first 600 chars for richer pattern matching)
        if content:
            text_parts.append(content[:600])

            # 4. Extract decorators & routes
            decorators = re.findall(r'@\w+[\.\w]*(?:\([^)]*\))?', content)
            if decorators:
                text_parts.append(f"Decorators: {', '.join(decorators)}")

            methods_match = re.findall(r"methods\s*=\s*\[([^\]]+)\]", content)
            if methods_match:
                text_parts.append(f"HTTP methods: {', '.join(methods_match)}")

            # DB tables
            tables = re.findall(r"\.(?:query|insert|update|delete|select|from)\s*\(\s*['\"](\w+)['\"]", content)
            if tables:
                text_parts.append(f"Database tables: {', '.join(set(tables))}")

            # Exception types
            exceptions = re.findall(r'(?:raise|except)\s+(\w+)', content)
            if exceptions:
                text_parts.append(f"Exceptions: {', '.join(set(exceptions))}")

        # 5. Inferred Q&A context
        #    Anticipate how a developer would ask about this chunk
        qa_hints = self._generate_qa_hints(chunk)
        if qa_hints:
            text_parts.append("Related questions: " + " | ".join(qa_hints))

        # 6. Keywords
        if keywords:
            text_parts.append(f"Keywords: {', '.join(keywords)}")

        # 7. Line range
        start_line = chunk.get("start_line", 0)
        end_line = chunk.get("end_line", 0)
        if start_line or end_line:
            text_parts.append(f"Lines: {start_line}-{end_line}")

        return "\n".join(text_parts)

    # =====================================================================
    # Q&A hint generation
    # =====================================================================

    @staticmethod
    def _generate_qa_hints(chunk: Dict[str, Any]) -> List[str]:
        """
        Generate anticipated Q&A-style phrases for a chunk.
        These get embedded into the vector, improving recall for natural questions.
        """
        hints: List[str] = []
        name = chunk.get("name", "")
        type_ = chunk.get("type", "")
        summary = chunk.get("summary", "")
        content = chunk.get("content", "")

        # Type-based hints
        if type_ == "class":
            hints.append(f"What does the {name} class do?")
            hints.append(f"How does {name} work?")
        elif type_ in ("function", "method"):
            base_name = name.split('.')[-1] if '.' in name else name
            hints.append(f"What does {base_name} do?")
            hints.append(f"How does {base_name} work?")
            if '.' in name:
                parent = name.split('.')[0]
                hints.append(f"How does {parent} handle {base_name}?")

        # Content-based hints
        if content:
            # If it has DB operations
            if re.search(r'\.(?:query|insert|update|delete)\s*\(', content):
                tables = re.findall(r"\.(?:query|insert|update|delete)\s*\(\s*['\"](\w+)['\"]", content)
                for t in set(tables):
                    hints.append(f"Which functions interact with the {t} table?")
                    hints.append(f"How is the {t} table used?")

            # If it has security patterns
            if re.search(r'\b(bcrypt|jwt|token|hash|encrypt|password)\b', content, re.IGNORECASE):
                hints.append("How is security implemented?")
                hints.append("What security measures are in place?")

            # If it has caching patterns
            if re.search(r'\b(cache|ttl|memoize|lru)\b', content, re.IGNORECASE):
                hints.append("How is caching used?")
                hints.append("How to improve performance?")

            # If it has rate limiting
            if re.search(r'\b(rate|throttle|limit)\b', content, re.IGNORECASE):
                hints.append("How are API endpoints protected?")
                hints.append("What prevents too many requests?")

            # If it has email operations
            if re.search(r'\b(email|send_email|smtp)\b', content, re.IGNORECASE):
                hints.append("How are emails sent?")

        # Summary-based hints
        if summary:
            if 'reset' in summary.lower() and 'password' in summary.lower():
                hints.append("What happens when a user forgets their password?")
            if 'migration' in summary.lower():
                hints.append("How are database schema changes managed?")
            if 'refresh' in summary.lower() and 'token' in summary.lower():
                hints.append("What happens when a token expires?")

        return hints[:5]  # Cap at 5 hints to avoid bloating embedding text
