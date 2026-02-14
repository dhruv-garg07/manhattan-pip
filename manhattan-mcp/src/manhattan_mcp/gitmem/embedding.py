"""
GitMem Local - Remote Embedding Client

Calls a remote embedding API to generate vector embeddings.
Vectors are stored locally in JSON files alongside the memories.

No local model downloads required - uses remote API for embeddings.
Uses only Python standard library (no requests dependency).
"""

import os
import json
import math
import hashlib
import urllib.request
import urllib.error
import ssl
from typing import List, Optional, Union, Dict, Any
from pathlib import Path


# Check if numpy is available, use pure Python fallback if not
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[Embedding] numpy not available, using pure Python vectors")


class PythonVector:
    """Pure Python vector implementation when numpy is not available."""
    
    def __init__(self, data: List[float]):
        self.data = list(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def tolist(self) -> List[float]:
        return self.data
    
    @property
    def shape(self):
        return (len(self.data),)
    
    def __truediv__(self, scalar):
        return PythonVector([x / scalar for x in self.data])
    
    @staticmethod
    def zeros(size: int) -> 'PythonVector':
        return PythonVector([0.0] * size)
    
    @staticmethod
    def from_list(data: List[float]) -> 'PythonVector':
        return PythonVector(data)
    
    @staticmethod
    def dot(a: 'PythonVector', b: 'PythonVector') -> float:
        return sum(x * y for x, y in zip(a.data, b.data))
    
    @staticmethod
    def norm(v: 'PythonVector') -> float:
        return math.sqrt(sum(x * x for x in v.data))


def create_vector(data: List[float]):
    """Create a vector using numpy if available, otherwise pure Python."""
    if HAS_NUMPY:
        return np.array(data, dtype=np.float32)
    return PythonVector.from_list(data)


def zeros_vector(size: int):
    """Create a zero vector."""
    if HAS_NUMPY:
        return np.zeros(size, dtype=np.float32)
    return PythonVector.zeros(size)


def vector_norm(v) -> float:
    """Calculate vector norm."""
    if HAS_NUMPY:
        return float(np.linalg.norm(v))
    return PythonVector.norm(v)


def vector_dot(a, b) -> float:
    """Calculate dot product."""
    if HAS_NUMPY:
        return float(np.dot(a, b))
    return PythonVector.dot(a, b)


def stack_vectors(vectors: List):
    """Stack vectors into a 2D array."""
    if HAS_NUMPY:
        return np.stack(vectors, axis=0)
    return vectors  # Return list for pure Python


class RemoteEmbeddingClient:
    """
    Remote embedding client that calls an API to generate vector embeddings.
    Uses Python standard library (urllib) - no external dependencies.
    
    Supports multiple API formats:
    - Gradio API (HuggingFace Spaces)
    - HuggingFace Inference API
    - OpenAI-compatible API
    
    Environment Variables:
        REMOTE_EMBEDDING_URL: Custom embedding API URL
        REMOTE_EMBEDDING_DIMENSION: Embedding dimension (default: 768)
    
    Usage:
        client = RemoteEmbeddingClient()
        embedding = client.embed("Hello world")
        embeddings = client.embed_batch(["Hello", "World"])
    """
    
    # Default Gradio API endpoint (768-dim dense embeddings)
    DEFAULT_API_URL = "https://iotacluster-embedding-model.hf.space/gradio_api/call/embed_dense"
    DEFAULT_DIMENSION = 768
    
    def __init__(
        self,
        api_url: str = None,
        api_key: str = None,
        model_name: str = "dense-embedding",
        dimension: int = None,
        timeout: int = 60,
        cache_embeddings: bool = True,
        cache_path: str = None
    ):
        """
        Initialize the remote embedding client.
        
        Args:
            api_url: Remote embedding API URL. Uses REMOTE_EMBEDDING_URL env var or default.
            api_key: API key for authentication (optional for some endpoints)
            model_name: Name of the embedding model
            dimension: Expected embedding dimension. Uses REMOTE_EMBEDDING_DIMENSION env var or 768.
            timeout: Request timeout in seconds
            cache_embeddings: Whether to cache embeddings locally
            cache_path: Path for embedding cache (if caching enabled)
        """
        # Get API URL from environment or use default
        self.api_url = api_url or os.getenv("REMOTE_EMBEDDING_URL") or os.getenv("EMBEDDING_API_URL") or self.DEFAULT_API_URL
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("HF_TOKEN")
        self.model_name = model_name
        
        # Get dimension from environment or use default
        env_dim = os.getenv("REMOTE_EMBEDDING_DIMENSION")
        self.dimension = dimension or (int(env_dim) if env_dim else self.DEFAULT_DIMENSION)
        
        self.timeout = timeout
        self.cache_embeddings = cache_embeddings
        
        # Detect API type based on URL
        self._is_gradio = "gradio_api" in self.api_url or "/call/" in self.api_url
        
        # Embedding cache
        self._cache: dict = {}
        self._cache_path = Path(cache_path) if cache_path else None
        
        if self._cache_path and self._cache_path.exists():
            self._load_cache()
        
        # SSL Context for certificate issues
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

        api_type = "Gradio" if self._is_gradio else "REST"
        print(f"[Embedding] Using {api_type} API: {self.api_url}")
        print(f"[Embedding] Dimension: {self.dimension}")
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        try:
            cache_file = self._cache_path / "embedding_cache.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert lists back to vectors
                    self._cache = {k: create_vector(v) for k, v in data.items()}
                print(f"[Embedding] Loaded {len(self._cache)} cached embeddings")
        except Exception as e:
            print(f"[Embedding] Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        if not self._cache_path:
            return
        try:
            self._cache_path.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_path / "embedding_cache.json"
            # Convert vectors to lists for JSON serialization
            data = {}
            for k, v in self._cache.items():
                if hasattr(v, 'tolist'):
                    data[k] = v.tolist()
                else:
                    data[k] = list(v)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[Embedding] Failed to save cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def embed(self, text: str):
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Normalized embedding vector
        """
        # Check cache first
        if self.cache_embeddings:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Call remote API
        embedding = self._call_api(text)
        
        # Cache result
        if self.cache_embeddings:
            self._cache[cache_key] = embedding
            # Periodically save cache
            if len(self._cache) % 100 == 0:
                self._save_cache()
        
        return embedding
    
    def embed_batch(self, texts: List[str]):
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            Array/list of normalized embedding vectors
        """
        embeddings = []
        for text in texts:
            emb = self.embed(text)
            embeddings.append(emb)
        return stack_vectors(embeddings)
    
    def _call_api(self, text: str):
        """
        Call the remote embedding API using urllib (no external dependencies).
        
        Supports multiple API formats:
        1. Gradio API (HuggingFace Spaces) - two-step request
        2. HuggingFace Inference API format
        3. OpenAI-compatible format
        """
        headers = {"Content-Type": "application/json"}
        
        # Add API key if available
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            if self._is_gradio:
                # Gradio API format: two-step request
                return self._call_gradio_api(text, headers)
            else:
                # Standard REST API format
                return self._call_rest_api(text, headers)
            
        except urllib.error.HTTPError as e:
            print(f"[Embedding] API HTTP error {e.code}: {e.reason}")
            return zeros_vector(self.dimension)
        except urllib.error.URLError as e:
            print(f"[Embedding] API URL error: {e.reason}")
            return zeros_vector(self.dimension)
        except Exception as e:
            print(f"[Embedding] API request failed: {e}")
            return zeros_vector(self.dimension)
    
    def _call_gradio_api(self, text: str, headers: dict):
        """
        Call Gradio API (HuggingFace Spaces).
        
        Gradio uses a two-step process:
        1. POST to /call/endpoint with {"data": [inputs]} -> returns {"event_id": "xxx"}
        2. GET /call/endpoint/event_id -> returns {"data": [result]}
        """
        import time
        
        # Step 1: Submit the request
        payload = {"data": [text]}
        data = json.dumps(payload).encode('utf-8')
        
        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers=headers,
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl_context) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        # Check if it's the two-step Gradio format
        if isinstance(result, dict) and "event_id" in result:
            event_id = result["event_id"]
            result_url = f"{self.api_url}/{event_id}"
            
            # Step 2: Poll for the result (with streaming support)
            max_attempts = 30
            for attempt in range(max_attempts):
                req = urllib.request.Request(result_url, headers=headers, method='GET')
                
                with urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl_context) as response:
                    # Read streaming response
                    response_text = response.read().decode('utf-8')
                    
                    # Parse event stream format
                    completed = False
                    embedding_result = None
                    
                    for line in response_text.strip().split('\n'):
                        line = line.strip()
                        if line.startswith('event:'):
                            event_type = line[6:].strip()
                            if event_type == 'complete':
                                completed = True
                        elif line.startswith('data:'):
                            json_str = line[5:].strip()
                            if json_str:
                                try:
                                    event_data = json.loads(json_str)
                                    if isinstance(event_data, list) and len(event_data) > 0:
                                        embedding_list = self._parse_gradio_result(event_data)
                                        if embedding_list:
                                            embedding_result = embedding_list
                                except json.JSONDecodeError:
                                    continue
                    
                    # Return if we got a valid embedding
                    if embedding_result:
                        return self._finalize_embedding(embedding_result)
                    
                    # If completed but no embedding, break (error case)
                    if completed:
                        break
                
                time.sleep(0.1)
            
            print("[Embedding] Gradio API timeout waiting for result")
            return zeros_vector(self.dimension)
        else:
            # Direct response (not two-step)
            embedding_list = self._parse_response(result)
            return self._finalize_embedding(embedding_list)
    
    def _call_rest_api(self, text: str, headers: dict):
        """Call standard REST API (HuggingFace Inference, OpenAI-compatible)."""
        payload = {"inputs": text, "options": {"wait_for_model": True}}
        data = json.dumps(payload).encode('utf-8')
        
        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers=headers,
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl_context) as response:
            response_data = json.loads(response.read().decode('utf-8'))
        
        embedding_list = self._parse_response(response_data)
        return self._finalize_embedding(embedding_list)
    
    def _finalize_embedding(self, embedding_list: List[float]):
        """Create and normalize an embedding vector."""
        embedding = create_vector(embedding_list)
        norm = vector_norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Update dimension if different
        if len(embedding) != self.dimension:
            self.dimension = len(embedding)
        
        return embedding
    
    def _parse_gradio_result(self, data) -> List[float]:
        """Parse embedding from Gradio API response.
        
        Handles formats:
        - [embedding_vector]  (direct list of floats)
        - [[embedding_vector]]  (nested list)
        - [{"dense_embedding": [...]}]  (dict with key)
        """
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            
            # Direct embedding vector: [0.1, 0.2, ...]
            if isinstance(first, (int, float)):
                return data
            
            # Dict format: [{"dense_embedding": [...]}]
            elif isinstance(first, dict):
                # Try common embedding keys
                for key in ['dense_embedding', 'embedding', 'vector', 'embeddings', 'data']:
                    if key in first:
                        emb = first[key]
                        if isinstance(emb, list) and len(emb) > 0:
                            if isinstance(emb[0], (int, float)):
                                return emb
                # Fallback: try first list value in dict
                for value in first.values():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], (int, float)):
                            return value
            
            # Nested vector: [[0.1, 0.2, ...]]
            elif isinstance(first, list):
                if len(first) > 0 and isinstance(first[0], (int, float)):
                    return first
        
        return None
    
    def _parse_response(self, data) -> List[float]:
        """Parse embedding from various API response formats."""
        
        # HuggingFace Inference API format: direct array
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], (int, float)):
                return data
            # Nested array format
            if isinstance(data[0], list):
                return data[0]
        
        # OpenAI format: {"data": [{"embedding": [...]}]}
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                if len(data["data"]) > 0:
                    item = data["data"][0]
                    if isinstance(item, dict) and "embedding" in item:
                        return item["embedding"]
                    if isinstance(item, dict) and "dense_embedding" in item:
                        return item["dense_embedding"]
            
            # Direct embedding field
            if "embedding" in data:
                return data["embedding"]
            if "dense_embedding" in data:
                return data["dense_embedding"]
        
        raise ValueError(f"Unknown embedding response format: {type(data)}")
    
    def cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = vector_norm(vec1)
        norm2 = vector_norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return vector_dot(vec1, vec2) / (norm1 * norm2)
    
    def batch_cosine_similarity(self, query_vec, doc_vecs) -> List[float]:
        """
        Calculate cosine similarity between a query vector and multiple document vectors.
        
        Args:
            query_vec: Query embedding
            doc_vecs: Document embeddings (list or array)
        
        Returns:
            List of similarity scores
        """
        if not doc_vecs:
            return []
        
        # Normalize query vector
        query_norm = vector_norm(query_vec)
        if query_norm == 0:
            return [0.0] * len(doc_vecs)
        
        similarities = []
        for doc_vec in doc_vecs:
            sim = self.cosine_similarity(query_vec, doc_vec)
            similarities.append(sim)
        
        return similarities
    
    def save_cache(self):
        """Manually save the embedding cache."""
        self._save_cache()
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        if self._cache_path:
            cache_file = self._cache_path / "embedding_cache.json"
            if cache_file.exists():
                cache_file.unlink()


# Convenience function for getting embeddings
def get_embedding(text: str, client: RemoteEmbeddingClient = None):
    """
    Get embedding for a text using the default or provided client.
    
    Args:
        text: Text to embed
        client: Optional embedding client (creates default if None)
    
    Returns:
        Embedding vector
    """
    if client is None:
        client = get_embedding_client()
    return client.embed(text)


def get_embeddings(texts: List[str], client: RemoteEmbeddingClient = None):
    """
    Get embeddings for multiple texts.
    
    Args:
        texts: List of texts to embed
        client: Optional embedding client
    
    Returns:
        List/array of embedding vectors
    """
    if client is None:
        client = get_embedding_client()
    return client.embed_batch(texts)


# =============================================================================
# Global Singleton Pattern - Avoids re-initialization overhead
# =============================================================================

import threading

# Global singleton instances
_embedding_client_singleton: Optional[RemoteEmbeddingClient] = None
_embedding_client_lock = threading.Lock()


def get_embedding_client(cache_path: str = None) -> RemoteEmbeddingClient:
    """
    Get or create the global embedding client singleton.
    
    This function ensures only one embedding client is created,
    avoiding the overhead of repeated initialization.
    
    Args:
        cache_path: Optional cache path (only used on first creation)
    
    Returns:
        The global RemoteEmbeddingClient instance
    """
    global _embedding_client_singleton
    
    if _embedding_client_singleton is None:
        with _embedding_client_lock:
            # Double-check locking
            if _embedding_client_singleton is None:
                _embedding_client_singleton = RemoteEmbeddingClient(
                    cache_path=cache_path
                )
    
    return _embedding_client_singleton


def reset_embedding_client():
    """Reset the global embedding client (for testing or reconfiguration)."""
    global _embedding_client_singleton
    with _embedding_client_lock:
        _embedding_client_singleton = None
