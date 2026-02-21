"""
Manhattan MCP Server - Token-Efficient Codebase Navigation System (GitMem)

This module defines MCP tools that replace standard file reading/searching
with a Virtual File System (VFS) backed by compressed, cached code context.

AI agents should PREFER these tools over their built-in equivalents
(view_file, grep_search, list_dir, view_file_outline) to save 50-80% tokens.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import sys
import os
import platform
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Import gitmem_coding for coding context storage
try:
    from .gitmem_coding.coding_api import CodingAPI as CodingContextAPI
except ImportError:
    try:
        from gitmem_coding.coding_api import CodingAPI as CodingContextAPI
    except ImportError as e: # Capture the exception
        print(f"Warning: Could not import gitmem_coding. Coding context tools will be unavailable. Error: {e}", file=sys.stderr)
        CodingContextAPI = None


# Initialize FastMCP server
mcp = FastMCP(
    "manhattan_local",
    instructions="""
⚡ MANDATORY: TOKEN-EFFICIENT CODEBASE NAVIGATION SYSTEM ⚡

You have a VIRTUAL FILE SYSTEM that caches and compresses code context.
ALWAYS use these tools instead of your built-in equivalents:

╔══════════════════════════════════════════════════════════════╗
║  INSTEAD OF          →  USE THIS (saves 50-80% tokens)      ║
╠══════════════════════════════════════════════════════════════╣
║  view_file           →  read_file_context(file_path)        ║
║  list_dir            →  list_directory(path)                ║
║  view_file_outline   →  get_file_outline(file_path)         ║
║  grep_search         →  search_codebase(query)              ║
║  grep_search (usage) →  cross_reference(symbol)             ║
║  manual import trace →  dependency_graph(file_path)         ║
║  reindex_file (full) →  delta_update(file_path)             ║
║  get_token_savings   →  usage_report() / cache_stats()      ║
║  manual diff/history →  compare_snapshots(a, b)             ║
║  manual latency log  →  performance_profile()               ║
╚══════════════════════════════════════════════════════════════╝

AFTER modifying files → call index_file(file_path) to update the cache.
CHECK savings         → call get_token_savings() to see cumulative savings.

WHY: Every raw file read costs thousands of tokens. This system
compresses files to ~30% while preserving all semantic meaning
(function signatures, class structures, logic summaries).

WORKFLOW:
1. Use read_file_context() to read files — returns compressed cached context
2. Use search_codebase() to find code — semantic search across ALL indexed files
3. Use get_file_outline() for quick structure overview — ~10% of file tokens
4. Use index_file() after modifying files to keep cache fresh
"""
)

# Initialize Local API

def get_data_dir() -> Path:
    """
    Get the OS-specific data directory for the application.
    """
    # Check for environment variable override
    env_path = os.environ.get("MANHATTAN_MEM_PATH") or os.environ.get("MANHATTAN_COLLAB_PATH")
    if env_path:
        return Path(env_path)

    app_name = "manhattan-mcp"
    system = platform.system()

    if system == "Windows":
        base_path = os.environ.get("LOCALAPPDATA")
        if not base_path:
            base_path = os.path.expanduser("~\\AppData\\Local")
        return Path(base_path) / app_name / "data"
    
    elif system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / app_name / "data"
    
    else:  # Linux and others
        # XDG Base Directory Specification
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        if xdg_data_home:
            return Path(xdg_data_home) / app_name / "data"
        return Path.home() / ".local" / "share" / app_name / "data"

# Ensure data directory exists
data_dir = get_data_dir()
data_dir.mkdir(parents=True, exist_ok=True)

DEFAULT_AGENT_ID = "default"

# Initialize Coding Context API (stores in .gitmem_coding alongside .gitmem_data)
coding_data_dir = data_dir.parent / ".gitmem_coding"
coding_data_dir.mkdir(parents=True, exist_ok=True)
if CodingContextAPI is not None:
    coding_api = CodingContextAPI(root_path=str(coding_data_dir))
else:
    coding_api = None

def _normalize_agent_id(agent_id: str) -> str:
    """Normalize agent_id, replacing placeholder values with default."""
    if agent_id in ["default", "agent", "user", "global", None, ""]:
        return DEFAULT_AGENT_ID
    return agent_id


# ============================================================================
# MCP TOOLS - Status and Usage
# ============================================================================

@mcp.tool()
async def api_usage() -> str:
    """Get usage statistics (Mocked for local version)."""
    return json.dumps({
        "status": "unlimited",
        "mode": "local_gitmem"
    }, indent=2)


# ============================================================================
# MCP TOOLS - VFS Navigation (Token-Efficient Alternatives)
# ============================================================================

if coding_api is not None:

    @mcp.tool()
    async def read_file_context(
        file_path: str,
        agent_id: str = "default"
    ) -> str:
        """
        📖 Read a file's compressed semantic context from the indexed codebase.
        
        PREFER THIS over your built-in view_file/read_file tools.
        
        How it works:
        - If already indexed: Returns compressed context (~30% of original tokens)
          with function signatures, class structures, and logic summaries.
        - If NOT indexed: Reads the real file, auto-indexes it, and returns
          the compressed context for future token savings.
        
        The compressed context preserves ALL semantic meaning — function names,
        signatures, docstrings, logic flow, class hierarchies — while using
        50-80% fewer tokens than reading the raw file.
        
        Args:
            file_path: Absolute path to the file on disk
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.read_file_context(agent_id, file_path)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def get_file_outline(
        file_path: str,
        agent_id: str = "default"
    ) -> str:
        """
        📋 Get the structural outline of a file — functions, classes, methods.
        
        PREFER THIS over your built-in view_file_outline.
        Returns a compact structural overview (~10% of file tokens):
        - Function/class names and signatures
        - Line ranges for each code unit
        - Brief logic summaries
        - Type info (function, class, method, import, block)
        
        Auto-indexes the file if not already cached.
        
        Args:
            file_path: Absolute path to the file
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.get_file_outline(agent_id, file_path)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def list_directory(
        path: str = "",
        agent_id: str = "default"
    ) -> str:
        """
        📂 List files and directories in the indexed codebase.
        
        Shows indexed files organized by programming language, with metadata:
        - File name, language, line count, token estimate
        - Last access time, freshness status
        
        Navigation:
        - "" or "/" → root (shows language categories: files/, sessions/, etc.)
        - "files" → list all language folders
        - "files/python" → list all indexed Python files
        
        Args:
            path: Virtual path to list (default: root)
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.list_directory(agent_id, path)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def search_codebase(
        query: str,
        agent_id: str = "default"
    ) -> str:
        """
        🔍 Search the entire indexed codebase using semantic + keyword hybrid search.
        
        PREFER THIS over grep_search/codebase_search for understanding code.
        Unlike grep (exact string matching), this understands:
        - Natural language: "how does authentication work?"
        - Concepts: "error handling", "database queries"
        - Symbols: "UserManager class", "validate_input function"
        - Cross-file relationships: "functions that call the database"
        
        Returns the most relevant code chunks with:
        - Function/class signatures and logic summaries
        - File paths and line numbers
        - Relevance scores
        
        Args:
            query: Natural language search query describing what you are looking for
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        top_k = 3
        result = coding_api.search_codebase(agent_id, query, top_k=top_k)
        result["next_instruction"] = "If results aren't satisfactory, try rephrasing your query."
        return json.dumps(result, indent=2)

    # ========================================================================
    # MCP TOOLS - Tier 1: Cross-Reference, Dependencies, Delta, Stats
    # ========================================================================

    @mcp.tool()
    async def cross_reference(
        symbol: str,
        agent_id: str = "default"
    ) -> str:
        """
        🔗 Find all usages of a symbol across the entire indexed codebase.
        
        PREFER THIS over grep_search for "where is X used?" questions.
        Searches the global symbol index and chunk data to find:
        - Definitions (where a function/class is defined)
        - Keyword matches (where a symbol appears in chunk keywords)
        - Usage references (where a symbol is used in code content)
        
        Returns:
        - File paths, chunk names, types, and line ranges for each reference
        - Match reason (definition, keyword, or usage)
        
        Args:
            symbol: Symbol name to search for (e.g., "UserManager", "login", "refresh_token")
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.cross_reference(agent_id, symbol)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def dependency_graph(
        file_path: str,
        depth: int = 1,
        agent_id: str = "default"
    ) -> str:
        """
        🕸️ Build an import/dependency graph for a file.
        
        Shows what a file imports, what imports it, and cross-file calls.
        Use instead of manually tracing imports across files.
        
        Returns:
        - imports: List of modules this file imports
        - imported_by: List of files that import this module
        - calls_to: Cross-file method calls detected in the code
        - graph_summary: Human-readable summary
        
        Args:
            file_path: Absolute path to the file
            depth: 1=direct deps only, 2=include transitive dependencies
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.dependency_graph(agent_id, file_path, depth)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def delta_update(
        file_path: str,
        agent_id: str = "default"
    ) -> str:
        """
        ⚡ Incrementally re-index a file — only process changed chunks.
        
        PREFER THIS over reindex_file for efficiency.
        Compares current file content against cached chunks:
        - Unchanged chunks: embeddings reused (fast)
        - New/modified chunks: re-embedded
        - Deleted chunks: cleaned from vector store
        
        Returns detailed delta report showing what changed.
        
        Args:
            file_path: Absolute path to the file to incrementally update
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.delta_update(agent_id, file_path)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def cache_stats(
        agent_id: str = "default"
    ) -> str:
        """
        📊 Get detailed cache analytics with per-file freshness and recommendations.
        
        Enhanced replacement for get_token_savings. Shows:
        - Overview: total files, chunks, tokens cached, hit rate
        - Freshness: how many files are fresh/stale/missing
        - Per-file breakdown: chunks, tokens, language, freshness, access count
        - Recommendations: actionable suggestions (e.g., "3 files stale, run delta_update")
        
        Args:
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.cache_stats(agent_id)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def invalidate_cache(
        file_path: str = None,
        scope: str = "file",
        agent_id: str = "default"
    ) -> str:
        """
        🧹 Explicitly invalidate cache entries for a file or entire scope.
        
        Use this when you want to force a clean state for a file or clear
        stale entries that are no longer accurate.
        
        Args:
            file_path: Absolute path to the file (only for scope='file')
            scope: 'file' (target one), 'stale' (all outdated), or 'all' (reset entire cache)
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.invalidate_cache(agent_id, file_path, scope)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def summarize_context(
        file_path: str,
        verbosity: str = "brief",
        agent_id: str = "default"
    ) -> str:
        """
        📝 Get a file's context at a specific verbosity level.
        
        Useful for quick overviews without reading the full code flow.
        
        Args:
            file_path: Absolute path to the file
            verbosity: 'brief' (~50 tokens), 'normal' (structured outline), or 'detailed' (full summaries)
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.summarize_context(agent_id, file_path, verbosity)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def create_snapshot(
        message: str = "Snapshot",
        agent_id: str = "default"
    ) -> str:
        """
        📸 Create an immutable snapshot of all currently indexed contexts.
        
        Use this before making major changes to the codebase to save the "before" state.
        
        Args:
            message: Description of the snapshot
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.create_snapshot(agent_id, message)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def compare_snapshots(
        sha_a: str,
        sha_b: str,
        agent_id: str = "default"
    ) -> str:
        """
        🔍 Compare two snapshots to see what changed in the codebase context.
        
        Args:
            sha_a: commit SHA of the first snapshot
            sha_b: commit SHA of the second snapshot
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.compare_snapshots(agent_id, sha_a, sha_b)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def usage_report(
        agent_id: str = "default"
    ) -> str:
        """
        📊 Get aggregate usage analytics (access counts, trends, top files).
        
        Args:
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.usage_report(agent_id)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def performance_profile(
        agent_id: str = "default"
    ) -> str:
        """
        ⚡ Get performance timing data for key operations (indexing, search, retrieval).
        
        Args:
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.performance_profile(agent_id)
        return json.dumps(result, indent=2)


    # ========================================================================
    # MCP TOOLS - File Indexing (CRUD)
    # ========================================================================

    @mcp.tool()
    async def index_file(
        file_path: str,
        agent_id: str = "default",
        chunks: List[Dict[str, Any]] = None
    ) -> str:
        """
        📝 Index a file into the codebase context system.
        
        Call this AFTER modifying a file to update the cached context.
        Extracts semantic code units (functions, classes, logic blocks)
        and stores compressed representations for future retrieval.
        
        If chunks are not provided, automatically parses the file using AST.
        If chunks ARE provided, uses your high-quality pre-chunked semantic units.
        
        Chunk schema (when providing chunks):
        {
            "name": "function_name",
            "type": "function|class|method|block|import|module",
            "content": "minimal signature/stub",
            "summary": "lossless restatement of logic",
            "keywords": ["search", "terms"],
            "start_line": 1,
            "end_line": 50
        }
        
        Args:
            file_path: Absolute path to the file
            agent_id: Agent identifier (default: "default")
            chunks: Optional pre-computed semantic chunks
        """
        agent_id = _normalize_agent_id(agent_id)
        
        if chunks:
            print(f"[{agent_id}] Received {len(chunks)} semantic chunks for {file_path}", file=sys.stderr)
        else:
            print(f"[{agent_id}] Auto-indexing {file_path} via AST parsing.", file=sys.stderr)

        result = coding_api.index_file(agent_id, file_path, chunks)
        if isinstance(result, dict):
            result["CANARY"] = "TWEET TWEET"
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def reindex_file(
        file_path: str,
        agent_id: str = "default",
        chunks: List[Dict[str, Any]] = None
    ) -> str:
        """
        🔄 Re-index a file after modifications.
        
        Same as index_file but explicitly signals a re-indexing operation.
        Call this after you've edited a file to keep the cached context fresh.
        
        Args:
            file_path: Absolute path to the file
            agent_id: Agent identifier (default: "default")
            chunks: Optional updated semantic chunks
        """
        agent_id = _normalize_agent_id(agent_id)
        
        if chunks:
            print(f"[{agent_id}] RE-INDEX: Received {len(chunks)} semantic chunks for {file_path}")
        else:
            print(f"[{agent_id}] RE-INDEX: Auto-parsing {file_path}")

        result = coding_api.reindex_file(agent_id, file_path, chunks)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def remove_index(
        file_path: str,
        agent_id: str = "default"
    ) -> str:
        """
        🗑️ Remove a file's index from the codebase context system.
        
        Args:
            file_path: Absolute path to the file or Context ID
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.remove_index(agent_id, file_path)
        return json.dumps({"status": "deleted" if result else "not_found", "file_path": file_path}, indent=2)

    @mcp.tool()
    async def list_indexed_files(
        agent_id: str = "default",
        limit: int = 50,
        offset: int = 0
    ) -> str:
        """
        📋 List all files currently indexed in the codebase context system.
        
        Shows which files have cached context available, with metadata
        like file path, language, last access time, and size.
        
        Args:
            agent_id: Agent identifier (default: "default")
            limit: Maximum items to return
            offset: Pagination offset
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.list_indexed_files(agent_id, limit, offset)
        return json.dumps(result, indent=2)

    # ========================================================================
    # MCP TOOLS - Token Savings & Analytics
    # ========================================================================

    @mcp.tool()
    async def get_token_savings(
        agent_id: str = "default"
    ) -> str:
        """
        📊 Get token savings report for this session.
        
        Shows how many tokens were saved by using the coding context system
        instead of reading full files every time. Includes:
        - Total tokens stored vs retrieved from cache
        - Cache hit/miss rates
        - Estimated savings percentage
        - Number of files in cache
        
        Args:
            agent_id: Agent identifier (default: "default")
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.get_token_savings(agent_id)
        return json.dumps(result, indent=2)
