"""
GitMem Local - Local File System for AI Context Storage

A GitHub-like version-controlled memory system for AI agents.
Inspired by gitmem, but running entirely locally without web APIs.

Features:
    - Local JSON-based storage
    - Vector embeddings for semantic search
    - Hybrid retrieval (keyword + semantic)
    - Git-like version control

Usage:
    from gitmem import LocalAPI
    
    api = LocalAPI()
    api.session_start("my-agent")
    api.add_memory("my-agent", [{"lossless_restatement": "User likes Python"}])
    api.search_memory("my-agent", "Python")
    api.session_end("my-agent")
"""

# Core data models
from .models import (
    MemoryEntry,
    Commit,
    Checkpoint,
    ActivityLog,
    AgentContext,
    MemoryType,
    MemoryScope,
    ObjectType as ModelObjectType
)

# Object store and version control
from .object_store import (
    ObjectStore,
    MemoryDAG,
    MemoryBlob,
    CognitiveTree,
    MemoryCommit,
    ObjectType,
    TreeEntry
)

# Virtual file system
from .file_system import (
    LocalFileSystem,
    FolderType,
    FileNode,
    FolderPermissions,
    AccessLevel,
    FOLDER_PERMISSIONS
)

# Memory storage
from .memory_store import LocalMemoryStore

# High-level context manager
from .context_manager import (
    ContextManager,
    SessionContext,
    get_context_manager
)

# Simple API interface
from .api import (
    LocalAPI,
    get_api,
    init as init_api
)

# Vector/Embedding components (optional - may not be available)
try:
    from .embedding import (
        RemoteEmbeddingClient,
        get_embedding,
        get_embeddings
    )
    from .vector_store import (
        LocalVectorStore,
        get_vector_store
    )
    from .hybrid_retriever import (
        HybridRetriever,
        RetrievalConfig,
        get_retriever
    )
    _VECTORS_AVAILABLE = True
except ImportError:
    _VECTORS_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "Manhattan AI"

__all__ = [
    # API (Primary Interface)
    "LocalAPI",
    "get_api",
    "init_api",
    
    # Data Models
    "MemoryEntry",
    "Commit",
    "Checkpoint",
    "ActivityLog",
    "AgentContext",
    "MemoryType",
    "MemoryScope",
    
    # Object Store
    "ObjectStore",
    "MemoryDAG",
    "MemoryBlob",
    "CognitiveTree",
    "MemoryCommit",
    "ObjectType",
    "TreeEntry",
    
    # File System
    "LocalFileSystem",
    "FolderType",
    "FileNode",
    "FolderPermissions",
    "AccessLevel",
    "FOLDER_PERMISSIONS",
    
    # Memory Store
    "LocalMemoryStore",
    
    # Context Manager
    "ContextManager",
    "SessionContext",
    "get_context_manager",
    
    # Vector/Embedding (if available)
    "RemoteEmbeddingClient",
    "get_embedding",
    "get_embeddings",
    "LocalVectorStore",
    "get_vector_store",
    "HybridRetriever",
    "RetrievalConfig",
    "get_retriever",
]


def quick_start():
    """Print quick start guide."""
    print("""
GitMem Local - Quick Start Guide
================================

from gitmem import LocalAPI

# Initialize
api = LocalAPI()

# Start session
api.session_start("my-agent")

# Add memories
api.add_memory("my-agent", [{
    "lossless_restatement": "User prefers dark mode",
    "keywords": ["preference", "dark mode"],
    "topic": "preferences"
}])

# Search
results = api.search_memory("my-agent", "dark mode")

# End session
api.session_end("my-agent")
    """)
