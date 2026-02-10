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

# Virtual file system
from .coding_file_system import CodingFileSystem, CodingFileNode

# High-level API
from .coding_api import (
    CodingAPI,
    get_coding_api
)

__version__ = "1.0.0"
__author__ = "Manhattan AI"

__all__ = [
    # API (Primary Interface)
    "CodingAPI",
    "get_coding_api",
    
    # Storage Engine
    "CodingContextStore",
    
    # Virtual File System
    "CodingFileSystem",
    "CodingFileNode",
    
    # Data Models
    "FileContext",
    "CodingSession",
    "FileLanguage",
    "ContextStatus",
    "TOKENS_PER_CHAR_RATIO",
]
