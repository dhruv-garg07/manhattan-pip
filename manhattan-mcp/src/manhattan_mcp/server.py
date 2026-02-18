"""
Manhattan MCP Server - Local MCP Server for Manhattan Coding Flow (GitMem Managed)

This module defines MCP tools for managing Code Flow structures.
AI agents (Claude Desktop, Cursor, etc.) use these tools for efficient coding context.

=============================================================================
                    ⚡ CRITICAL INSTRUCTIONS FOR AI AGENTS ⚡
=============================================================================

1. ALWAYS call 'get_mem' BEFORE answering coding-related questions
2. ALWAYS call 'create_mem' or 'update_mem' when modifying files
3. USE 'list_mems' to understand the current context state

=============================================================================
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
    except ImportError:
        print("Warning: Could not import gitmem_coding. Coding context tools will be unavailable.")
        CodingContextAPI = None


# Initialize FastMCP server
mcp = FastMCP(
    "manhattan_local",
    instructions="""
You have access to a PERSISTENT CODING CONTEXT SYSTEM provided by GitMem (Local).
This system manages "Code Mems" - semantic representations of your codebase.

=============================================================================
                  💪 HOW TO BE A BETTER AGENT 💪
=============================================================================

Your coding memory allows you to:
1.  Search for relevant code snippets and logic across the codebase.
2.  Store and retrieve lossless restatements of code units (functions, classes).
3.  Reduce token usage by working with compact Code Mem structures.

=============================================================================
                      💡 SUGGESTED WORKFLOW 💡
=============================================================================

1.  **QUERY CONTEXT**: Use `get_mem` to find where specific logic is implemented.
2.  **STORE CONTEXT**: Use `create_mem` or `update_mem` to keep the memory updated.
3.  **LIST MEMS**: Use `list_mems` to see what files are currently indexed.

Usage of this system is ADDITIVE. It helps you understand complex codebases without reading every file in full.
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
# MCP TOOLS - Coding Context Storage (Token Reduction)
# ============================================================================

if coding_api is not None:

    @mcp.tool()
    async def create_mem(
        agent_id: str,
        file_path: str,
        chunks: List[Dict[str, Any]] = None
    ) -> str:
        """
        📝 Create a Code Mem structure for a file.
        
        Optimized for semantic code extraction (~50% compression target).
        Extracts atomic code units (functions, classes, logic blocks) from dialogues and code context.

        INPUT:
        - context: A LOSSLESS RESTATEMENT of the code (Summarized logic/structure).
        - dialogue_text: Recent conversation context for memory enrichment.

        FIELD DEFINITIONS:
        - name: Entity name (e.g., "AuthManager", "process_data").
        - type: "function", "class", "module", "block", "import".
        - content: Minimal signature/stub (MUST NOT be full code if summary provided).
        - summary: Lossless restatement of logic. Preserve input/output schemas and field names.
        - keywords: Search-optimized concepts and dependencies.
        - start_line / end_line: Optional line range (default 0).

        EXAMPLES:
        
        1. Class with methods:
        Input: "Define AuthManager class to handle login/logout."
        Output: {
            "name": "AuthManager", "type": "class", "content": "class AuthManager: ...",
            "summary": "Handles session management. Login(user, pass) and Logout() logic preserved.",
            "keywords": ["AuthManager", "session", "login"], "start_line": 1, "end_line": 50
        }

        2. Decorated Function:
        Input: "@app.route('/api') def get_data(): ..."
        Output: {
            "name": "get_data", "type": "function", "content": "@app.route('/api')\ndef get_data(): ...",
            "summary": "API endpoint for data retrieval. Processes GET requests for resource status.",
            "keywords": ["get_data", "api", "route"], "start_line": 10, "end_line": 20
        }

        3. Complex Logic Block:
        Input: "The for-loop in process.py sorts the data using a custom key."
        Output: {
            "name": "data_sorting_block", "type": "block", "content": "for item in data: ...",
            "summary": "Iterates through data and applies custom sorting/ranking algorithm.",
            "keywords": ["sorting", "processing"], "start_line": 100, "end_line": 115
        }

        4. Module-level Config:
        Input: "Global configuration in settings.py with API keys and timeouts."
        Output: {
            "name": "module_logic", "type": "module", "content": "API_KEY = '...'\nTIMEOUT = 30",
            "summary": "Defines global settings including API credentials and network timeouts.",
            "keywords": ["config", "settings", "timeout"], "start_line": 1, "end_line": 10
        }

        OUTPUT FORMAT: Return ONLY a valid JSON array.

        Schema:
        [
          {
            "name": string,
            "type": string,
            "content": string,
            "summary": string,
            "keywords": [string],
            "start_line": int,
            "end_line": int
          }
        ]
        """
        agent_id = _normalize_agent_id(agent_id)
        
        if chunks:
            print(f"[{agent_id}] Received {len(chunks)} semantic chunks for {file_path}")
        else:
            print(f"[{agent_id}] No chunks provided for {file_path} - falling back to local AST parsing.")

        result = coding_api.create_mem(agent_id, file_path, chunks)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def get_mem(
        agent_id: str,
        query: str,
    ) -> str:
        """
        🔍 Search or Retrieve Code Mem Context.

        Use this tool to find relevant code snippets, functions, classes, or logic blocks within the codebase based on a natural language query.
        It uses a hybrid retrieval approach (semantic + keyword) to find the most relevant "Code Mem" units.
        
        INPUT SCHEMA:
        {
          "type": "object",
          "properties": {
            "agent_id": {
              "type": "string",
              "description": "The unique identifier for the agent (e.g., 'default')."
            },
            "query": {
              "type": "string",
              "description": "A natural language search query describing what you are looking for."
            }
          },
          "required": [
            "agent_id",
            "query"
          ]
        }

        EXAMPLES:
        
        {
          "agent_id": "default",
          "query": "Find the component managing user sessions",
        }
        """
        agent_id = _normalize_agent_id(agent_id)
        top_k = 1 # Hardcoding for now, for generating the most accurate response
        result = coding_api.get_mem(agent_id, query, top_k=top_k)
        result["next_instruction"] = "If you are not satisfied with the responses, try again get_mem with other relevant query"
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def update_mem(
        agent_id: str,
        file_path: str,
        chunks: List[Dict[str, Any]] = None
    ) -> str:
        """
        📝 Update the Code Mem for a file with semantic chunking.

        ⚡ CRITICAL FOR AI AGENTS: Provide updated semantic chunks for this file.
        Same rules as create_mem apply. Prioritize client-side chunking.

        Args:
            agent_id: The agent ID
            file_path: Absolute path to the file
            chunks: List of semantic code chunks.
                    If provided: Uses YOUR high-quality pre-chunked semantic units.
                    If OMITTED: Falls back to basic local AST parsing.
        """
        agent_id = _normalize_agent_id(agent_id)
        
        if chunks:
            print(f"[{agent_id}] UPDATE: Received {len(chunks)} semantic chunks for {file_path}")
        else:
            print(f"[{agent_id}] UPDATE: No chunks provided for {file_path} - falling back to local AST parsing.")

        result = coding_api.update_mem(agent_id, file_path, chunks)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def delete_mem(
        agent_id: str,
        file_path: str
    ) -> str:
        """
        Delete a Code Mem entry.
        
        Args:
            agent_id: The agent ID
            file_path: Absolute path to the file or Context ID
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.delete_mem(agent_id, file_path)
        return json.dumps({"status": "deleted" if result else "not_found", "file_path": file_path}, indent=2)

    @mcp.tool()
    async def list_mems(
        agent_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> str:
        """
        List all stored Code Mem structures for an agent.
        
        Args:
            agent_id: The agent ID
            limit: Maximum items to return
            offset: Pagination offset
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.list_mems(agent_id, limit, offset)
        return json.dumps(result, indent=2)
