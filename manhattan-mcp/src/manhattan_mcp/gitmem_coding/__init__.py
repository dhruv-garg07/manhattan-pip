"""
GitMem Coding - Coding Context Storage for AI Agents

A cross-session file content caching system focused on TOKEN REDUCTION.
When an agent reads files in one session, the content is stored locally.
In subsequent sessions, the same content can be retrieved from cache
instead of re-reading, saving significant tokens.

Features:
    - Cross-session file content caching
    - Hash-based freshness detection (SHA-256)
    - Token savings tracking and analytics
    - Keyword search across cached file contexts
    - Automatic stale entry cleanup

Storage:
    Data is stored in .gitmem_coding/ (separate from .gitmem_data/)

Usage:
    from manhattan_mcp.gitmem_coding import CodingAPI
    
    api = CodingAPI()
    
    # Store file read by agent
    api.store_file("my-agent", "/path/to/file.py", content, "python")
    
    # Retrieve in next session (with freshness check)
    result = api.retrieve_file("my-agent", "/path/to/file.py")
    if result["freshness"] == "fresh":
        content = result["content"]  # Token savings!
    
    # Get stats
    stats = api.get_stats("my-agent")
"""

# Data models
from .models import (
    FileContext,
    CodingSession,
    FileLanguage,
    ContextStatus,
    TOKENS_PER_CHAR_RATIO
)

# Storage engine
from .coding_store import CodingContextStore

# Code Flow Generator
from .ast_skeleton import (
    ContextTreeBuilder,
    detect_language,
    BSTIndex
)

# High-level API
from .coding_api import CodingAPI
from .coding_memory_builder import CodingMemoryBuilder
from .coding_hybrid_retriever import CodingHybridRetriever

__version__ = "1.0.0"
__author__ = "Manhattan AI"

__all__ = [
    # API (Primary Interface)
    "CodingAPI",
    "CodingMemoryBuilder",
    "CodingHybridRetriever",
    
    # Storage Engine
    "CodingContextStore",
    
    # Code Flow Generator
    "ContextTreeBuilder",
    "BSTIndex",
    
    # Data Models
    "FileContext",
    "CodingSession",
    "FileLanguage",
    "ContextStatus",
    "TOKENS_PER_CHAR_RATIO",
]
