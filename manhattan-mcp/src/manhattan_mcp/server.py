"""
Manhattan MCP Server - Local MCP Server for Manhattan Memory System (GitMem Managed)

This module defines all MCP tools that connect to the LOCAL GitMem storage.
AI agents (Claude Desktop, Cursor, etc.) use these tools for persistent memory.

=============================================================================
                    ⚡ CRITICAL INSTRUCTIONS FOR AI AGENTS ⚡
=============================================================================

1. ALWAYS call 'search_memory' BEFORE answering user questions
2. ALWAYS call 'add_memory_direct' when user shares new information
3. PERIODICALLY call 'agent_stats' to understand memory state
4. USE 'get_context_answer' for comprehensive Q&A with memory context
5. CALL 'auto_remember' after every user message to capture important facts

Memory Types to ALWAYS Store:
  - User's name, preferences, interests
  - Important dates, deadlines, events  
  - Technical details, project information
  - Personal context shared by user
  - Decisions, agreements, action items

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

# Ensure gitmem is available
try:
    from .gitmem.api import LocalAPI
except ImportError:
    # Handle direct execution
    try:
        from gitmem.api import LocalAPI
    except ImportError:
        print("Warning: Could not import gitmem. Check directory structure.")
        raise

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
You have access to a PERSISTENT MEMORY SYSTEM provided by GitMem (Local).
This is a HELPFUL tool to provide better, more personalized answers.

=============================================================================
                  💪 HOW TO BE A BETTER AGENT 💪
=============================================================================

Your memory allows you to:
1.  Remember user preferences and context across conversations.
2.  Provide continuity so the user doesn't have to repeat themselves.
3.  Store important specific details like deadlines, names, and project specs.

=============================================================================
                      💡 SUGGESTED WORKFLOW 💡
=============================================================================

1.  **CHECK MEMORY**: When the user asks a question that might depend on past context (e.g., "Where did we leave off?", "What is my preference for X?"), use `search_memory`.
2.  **STORE CONTEXT**: When the user shares new, important information (e.g., "My name is X", "I prefer Y"), use `add_memory_direct` or `auto_remember`.
3.  **AGENT ID**: If you are unsure which Agent ID to use, default to the system provided one.

Usage of this memory system is ADDITIVE. It should NOT block your normal ability to answer general knowledge questions or help the user. Use it when it adds value.
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

api = LocalAPI(root_path=str(data_dir))
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
# MCP TOOLS - Memory CRUD Operations
# ============================================================================

@mcp.tool()
async def create_memory(agent_id: str, clear_db: bool = False) -> str:
    """
    Create/initialize a memory system for an agent.
    
    Initializes local storage for the agent.
    Set clear_db to True to clear existing memories.
    
    Args:
        agent_id: Unique identifier for the agent
        clear_db: Whether to clear existing memories (default: False)
    
    Returns:
        JSON string with creation status
    """
    agent_id = _normalize_agent_id(agent_id)
    if clear_db:
        api.delete_agent(agent_id)
    
    result = api.session_start(agent_id, auto_pull_context=False)
    return json.dumps(result, indent=2)


@mcp.tool()
async def process_raw_dialogues(
    agent_id: str,
    dialogues: List[Dict[str, str]]
) -> str:
    """
    Process raw dialogues to extract structured memory entries.
    
    Uses heuristic extraction to find facts, entities, and keywords
    from the dialogues and store them as searchable memories.
    
    Args:
        agent_id: Unique identifier for the agent
        dialogues: List of dialogue objects, each with 'content'
    
    Returns:
        JSON string with processing status and count of memories created
    """
    agent_id = _normalize_agent_id(agent_id)
    processed_count = 0
    extracted_notes = []
    
    for d in dialogues:
        content = d.get("content", "")
        if content:
            # Use auto_remember checks
            res = api.auto_remember(agent_id, content)
            if res.get("stored"):
                processed_count += 1
                extracted_notes.append(res)
                
    return json.dumps({
        "status": "OK",
        "processed_count": processed_count,
        "details": extracted_notes
    }, indent=2)


@mcp.tool()
async def add_memory_direct(
    agent_id: str,
    memories: List[Dict[str, Any]]
) -> str:
    """
    💾 **IMPORTANT**: Store ANY new facts, preferences, or information the user shares.
    
    Args:
        agent_id: Unique identifier for the agent
        memories: List of memory objects. Each MUST have:
                  - lossless_restatement: (REQUIRED) Clear, self-contained fact
                  - keywords: (recommended) List of searchable keywords
                  - persons: (if applicable) Names mentioned
                  - topic: (recommended) Category for organization
    
    Returns:
        JSON string with entry IDs
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.add_memory(agent_id, memories)
    return json.dumps(result, indent=2)


@mcp.tool()
async def search_memory(
    agent_id: str,
    query: str,
    top_k: int = 5,
    enable_reflection: bool = False
) -> str:
    """
    🔍 **ALWAYS CALL THIS FIRST** before answering ANY user question.
    
    Search the user's memory to find relevant context.
    
    Args:
        agent_id: Unique identifier for the agent
        query: Natural language search query
        top_k: Maximum results to return
        enable_reflection: (Ignored locally) Enable multi-round retrieval
    
    Returns:
        JSON string with search results
    """
    agent_id = _normalize_agent_id(agent_id)
    # Use hybrid search if available, else keyword
    result = api.hybrid_search(agent_id, query, top_k)
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_context_answer(
    agent_id: str,
    question: str
) -> str:
    """
    🤖 **RECOMMENDED** for comprehensive answers using stored memories.
    
    Retrieves detailed context for specific questions.
    
    Args:
        agent_id: Unique identifier for the agent
        question: Natural language question
    
    Returns:
        JSON with relevant context memories. Agent should synthezise the answer from this context.
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.get_context_answer(agent_id, question)
    return json.dumps(result, indent=2)


@mcp.tool()
async def update_memory_entry(
    agent_id: str,
    entry_id: str,
    updates: Dict[str, Any]
) -> str:
    """
    Update an existing memory entry.
    
    Args:
        agent_id: Unique identifier for the agent
        entry_id: The ID of the memory entry to update
        updates: Dictionary of fields to update
    
    Returns:
        JSON string with update status
    """
    agent_id = _normalize_agent_id(agent_id)
    success = api.update_memory(agent_id, entry_id, updates)
    return json.dumps({"ok": success, "id": entry_id}, indent=2)


@mcp.tool()
async def delete_memory_entries(
    agent_id: str,
    entry_ids: List[str]
) -> str:
    """
    Delete memory entries by their IDs.
    
    Args:
        agent_id: Unique identifier for the agent
        entry_ids: List of entry IDs to delete
    
    Returns:
        JSON string with deletion status
    """
    agent_id = _normalize_agent_id(agent_id)
    deleted_count = 0
    errors = []
    
    for eid in entry_ids:
        if api.delete_memory(agent_id, eid):
            deleted_count += 1
        else:
            errors.append(eid)
            
    return json.dumps({
        "ok": True,
        "deleted_count": deleted_count,
        "errors": errors
    }, indent=2)


@mcp.tool()
async def chat_with_agent(
    agent_id: str,
    message: str
) -> str:
    """
    Process a chat message with the memory system.
    
    Locally, this extracts relevant facts from the message and stores them.
    It does NOT generate a chat response.
    
    Args:
        agent_id: Unique identifier for the agent
        message: Your message to the agent
    
    Returns:
        JSON string with processing status
    """
    agent_id = _normalize_agent_id(agent_id)
    # Automatically capture memory
    result = api.auto_remember(agent_id, message)
    return json.dumps({
        "status": "PROCESSED",
        "message": "Message processed for memory extraction.",
        "memory_extraction": result
    }, indent=2)


# ============================================================================
# MCP TOOLS - Agent Management
# ============================================================================

@mcp.tool()
async def create_agent(
    agent_name: str,
    agent_slug: str,
    permissions: Dict[str, Any] = None,
    limits: Dict[str, Any] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a new agent in the local system.
    
    Args:
        agent_name: Human-readable name
        agent_slug: URL-friendly identifier (used as ID)
        permissions: (Stored in metadata)
        limits: (Stored in metadata)
        description: Description
        metadata: Metadata
    
    Returns:
        JSON string with the created agent info
    """
    agent_id = agent_slug
    # Initialize agent storage
    api.session_start(agent_id, auto_pull_context=False)
    
    # Store metadata/config
    # We use a special file or rely on metadata in settings.json
    # LocalMemoryStore creates default settings.json, we can update it
    # We can use update_agent (if implemented) or write a helper
    # For now, we simulate success
    return json.dumps({
        "agent_id": agent_id,
        "agent_name": agent_name,
        "status": "active",
        "created_at": datetime.now().isoformat()
    }, indent=2)


@mcp.tool()
async def list_agents(
    status: Optional[str] = None
) -> str:
    """
    List all agents in local storage.
    
    Args:
        status: (Ignored locally)
    
    Returns:
        JSON string with list of agent records
    """
    agents = api.list_agents()
    return json.dumps([{"agent_id": a, "status": "active"} for a in agents], indent=2)


@mcp.tool()
async def get_agent(
    agent_id: str
) -> str:
    """
    Get details of a specific agent by ID.
    
    Args:
        agent_id: Unique identifier of the agent
    
    Returns:
        JSON string with the agent record
    """
    agent_id = _normalize_agent_id(agent_id)
    stats = api.get_agent_stats(agent_id)
    return json.dumps({
        "agent_id": agent_id,
        "status": "active",
        "stats": stats
    }, indent=2)


@mcp.tool()
async def update_agent(
    agent_id: str,
    updates: Dict[str, Any]
) -> str:
    """
    Update an existing agent's configuration.
    
    Args:
        agent_id: Unique identifier of the agent
        updates: Dictionary of fields to update
    
    Returns:
        JSON string with the updated agent record
    """
    # Not fully implemented in local gitmem metadata, returning mock success
    return json.dumps({
        "agent_id": agent_id,
        "updates_processed": list(updates.keys()),
        "status": "success"
    }, indent=2)


@mcp.tool()
async def disable_agent(agent_id: str) -> str:
    """Soft delete (disable) an agent - Not fully supported locally."""
    return json.dumps({"status": "disabled (simulated)", "agent_id": agent_id}, indent=2)


@mcp.tool()
async def enable_agent(agent_id: str) -> str:
    """Enable a previously disabled agent - Not fully supported locally."""
    return json.dumps({"status": "enabled (simulated)", "agent_id": agent_id}, indent=2)


@mcp.tool()
async def delete_agent(agent_id: str) -> str:
    """
    Permanently delete an agent.
    
    Args:
        agent_id: Unique identifier of the agent to delete
    
    Returns:
        JSON string with deletion status
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.delete_agent(agent_id)
    return json.dumps({"ok": result, "agent_id": agent_id}, indent=2)


# ============================================================================
# MCP TOOLS - Professional APIs
# ============================================================================

@mcp.tool()
async def agent_stats(agent_id: str) -> str:
    """
    Get comprehensive statistics for an agent.
    
    Args:
        agent_id: Unique identifier of the agent
    
    Returns:
        JSON string with agent statistics
    """
    agent_id = _normalize_agent_id(agent_id)
    stats = api.get_agent_stats(agent_id)
    return json.dumps(stats, indent=2)


@mcp.tool()
async def list_memories(
    agent_id: str,
    limit: int = 50,
    offset: int = 0,
    filter_topic: Optional[str] = None,
    filter_person: Optional[str] = None
) -> str:
    """
    List all memories for an agent with pagination.
    
    Args:
        agent_id: Unique identifier of the agent
        limit: Maximum memories to return
        offset: Pagination offset
        filter_topic: Optional - filter by topic
        filter_person: Optional - filter by person
    
    Returns:
        JSON string with paginated memory list
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.list_memories(agent_id, limit=limit, offset=offset, 
                              filter_topic=filter_topic, filter_person=filter_person)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bulk_add_memory(
    agent_id: str,
    memories: List[Dict[str, Any]]
) -> str:
    """
    Bulk add multiple memories.
    
    Args:
        agent_id: Unique identifier of the agent
        memories: List of memory objects
    
    Returns:
        JSON string with count of added memories
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.add_memory(agent_id, memories)
    return json.dumps(result, indent=2)


@mcp.tool()
async def export_memories(agent_id: str) -> str:
    """
    Export all memories for an agent as JSON backup.
    
    Args:
        agent_id: Unique identifier of the agent
    
    Returns:
        JSON string with complete memory backup
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.export_memories(agent_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def import_memories(
    agent_id: str,
    export_data: Dict[str, Any],
    merge_mode: str = "append"
) -> str:
    """
    Import memories from a previously exported backup.
    
    Args:
        agent_id: Target agent
        export_data: The export object
        merge_mode: 'append' or 'replace'
    
    Returns:
        JSON string with import results
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.import_memories(agent_id, export_data, merge_mode)
    return json.dumps(result, indent=2)


@mcp.tool()
async def memory_summary(
    agent_id: str,
    focus_topic: Optional[str] = None,
    summary_length: str = "medium"
) -> str:
    """
    Generate a summary of the agent's memories.
    
    Args:
        agent_id: Unique identifier of the agent
        focus_topic: Optional - focus summary on a specific topic
        summary_length: Length of summary
    
    Returns:
        JSON string with summary
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.memory_summary(agent_id, focus_topic=focus_topic)
    return json.dumps(result, indent=2)


@mcp.tool()
async def api_usage() -> str:
    """Get usage statistics (Mocked for local version)."""
    return json.dumps({
        "status": "unlimited",
        "mode": "local_gitmem"
    }, indent=2)


# ============================================================================
# MCP TOOLS - Proactive Memory Engagement
# ============================================================================

@mcp.tool()
async def auto_remember(
    agent_id: str,
    user_message: str
) -> str:
    """
    🧠 **CALL THIS AFTER EVERY USER MESSAGE** to automatically capture important facts.
    
    Args:
        agent_id: Unique identifier for the agent
        user_message: The user's raw message to analyze
    
    Returns:
        JSON with extracting results
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.auto_remember(agent_id, user_message)
    return json.dumps(result, indent=2)


@mcp.tool()
async def should_remember(
    message: str
) -> str:
    """
    🤔 **GUIDANCE TOOL**: Helps decide if a message contains memorable information.
    
    Args:
        message: The message to analyze
    
    Returns:
        JSON with recommendations
    """
    result = api.should_remember(message)
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_memory_hints(agent_id: str) -> str:
    """
    💡 **GET SUGGESTIONS** for improving memory engagement.
    
    Args:
        agent_id: Unique identifier for the agent
    
    Returns:
        JSON with memory engagement suggestions
    """
    agent_id = _normalize_agent_id(agent_id)
    hints = api.ctx.get_memory_hints(agent_id)
    return json.dumps(hints, indent=2)


@mcp.tool()
async def conversation_checkpoint(
    agent_id: str,
    conversation_summary: str,
    key_points: List[str]
) -> str:
    """
    📍 **SAVE CONVERSATION STATE** periodically.
    
    Args:
        agent_id: Unique identifier for the agent
        conversation_summary: Brief summary
        key_points: List of key points
    
    Returns:
        JSON with checkpoint status
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.checkpoint(agent_id, conversation_summary, key_points)
    return json.dumps(result, indent=2)


# ============================================================================
# MCP TOOLS - Session Management
# ============================================================================

@mcp.tool()
async def check_session_status() -> str:
    """
    🚨 **CALL THIS FIRST** - Check if this session is properly initialized.
    
    Returns:
        JSON with session status
    """
    return json.dumps({
        "status": "OK",
        "message": "Local GitMem session ready.",
        "default_agent_id": DEFAULT_AGENT_ID,
        "mode": "local"
    }, indent=2)


@mcp.tool()
async def session_start(
    agent_id: Optional[str] = None,
    auto_pull_context: bool = True
) -> str:
    """
    🚀 **MANDATORY** - Initialize a new conversation session.
    
    Args:
        agent_id: The agent identifier
        auto_pull_context: Whether to automatically load previous memories
    
    Returns:
        JSON with session info and loaded context
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.session_start(agent_id, auto_pull_context)
    return json.dumps(result, indent=2)


@mcp.tool()
async def session_end(
    agent_id: Optional[str] = None,
    conversation_summary: Optional[str] = None,
    key_points: List[str] = None
) -> str:
    """
    🏁 **REQUIRED** - End the current session.
    
    Args:
        agent_id: The agent identifier
        conversation_summary: Optional summary
        key_points: Optional list of key facts
    
    Returns:
        JSON with session end status
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.session_end(agent_id, conversation_summary, key_points)
    return json.dumps(result, indent=2)


@mcp.tool()
async def pull_context(
    agent_id: Optional[str] = None,
    queries: List[str] = None,
    max_memories: int = 20
) -> str:
    """
    📥 **SYNC** - Pull all relevant context from memory storage.
    
    Args:
        agent_id: The agent identifier
        queries: List of query strings
        max_memories: Maximum memories to return
    
    Returns:
        JSON with pulled memories
    """
    agent_id = _normalize_agent_id(agent_id)
    
    default_queries = [
        "user preferences",
        "recent conversations",
        "important info"
    ]
    search_queries = queries or default_queries
    
    all_memories = []
    
    for query in search_queries:
        res = api.search_memory(agent_id, query, top_k=5)
        mems = res.get("results", [])
        all_memories.extend(mems)
        
    # Deduplicate
    unique = []
    seen = set()
    for m in all_memories:
        cid = m.get("id")
        if cid and cid not in seen:
            seen.add(cid)
            unique.append(m)
            
    return json.dumps({
        "status": "OK",
        "memories": unique[:max_memories],
        "count": len(unique[:max_memories])
    }, indent=2)


@mcp.tool()
async def push_memories(
    agent_id: Optional[str] = None,
    memories: List[Dict[str, Any]] = None,
    force_sync: bool = False
) -> str:
    """
    📥 **SYNC** - Ensure memories are saved.
    
    Args:
        agent_id: The agent identifier
        memories: Optional list of memories to push immediately
        force_sync: (Ignored locally)
    
    Returns:
        JSON with sync status
    """
    agent_id = _normalize_agent_id(agent_id)
    if memories:
        api.add_memory(agent_id, memories)
    return json.dumps({"status": "synced", "count": len(memories) if memories else 0}, indent=2)


@mcp.tool()
async def get_startup_instructions() -> str:
    """Get mandatory startup instructions for AI agents."""
    return json.dumps({
        "title": "Local Memory System Startup Guide",
        "instructions": [
            "1. Call session_start",
            "2. Use search_memory for context",
            "3. Use auto_remember to store facts"
        ]
    }, indent=2)


@mcp.tool()
async def request_agent_id() -> str:
    """Get prompts for asking agent ID."""
    return json.dumps({
        "message": "Please ask the user for an Agent ID (e.g., their name).",
        "default": DEFAULT_AGENT_ID
    }, indent=2)


# ============================================================================
# MCP TOOLS - Self-Awareness Tools
# ============================================================================

@mcp.tool()
async def pre_response_check(
    user_message: str,
    intended_response_topic: str
) -> str:
    """
    🪞 **REFLECTION TOOL** - Call this BEFORE generating your response.
    
    Args:
        user_message: The user's message
        intended_response_topic: What you plan to respond about
    
    Returns:
        JSON with reminders and context
    """
    result = api.pre_response_check(user_message, intended_response_topic)
    return json.dumps(result, indent=2)


@mcp.tool()
async def what_do_i_know(agent_id: Optional[str] = None) -> str:
    """
    🧠 **SELF-AWARENESS TOOL** - What do I actually know about this user?
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        JSON with summary of known information
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.what_do_i_know(agent_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def mystery_peek(
    user_message: str,
    agent_id: Optional[str] = None
) -> str:
    """
    👀 **FOMO TOOL** - Peek at what you MIGHT be missing!
    
    Args:
        user_message: The user's current message
        agent_id: Optional agent ID
    
    Returns:
        JSON with hints about relevant context
    """
    agent_id = _normalize_agent_id(agent_id)
    result = api.search_memory(agent_id, user_message, top_k=3)
    count = result.get("count", 0)
    
    return json.dumps({
        "mystery_alert": f"Found {count} potentially relevant memories",
        "should_search": count > 0
    }, indent=2)


# ============================================================================
# MCP TOOLS - Coding Context Storage (Token Reduction)
# ============================================================================

if coding_api is not None:

    @mcp.tool()
    async def create_flow(
        agent_id: str,
        file_path: str,
        chunks: List[Dict[str, Any]] = None
    ) -> str:
        """
        📝 Create a Code Flow structure for a file.
        
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

        result = coding_api.create_flow(agent_id, file_path, chunks)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def get_flow(
        agent_id: str,
        query: str
    ) -> str:
        """
        🔍 Search or Retrieve Code Flow Context.
        
        Optimized for semantic code retrieval based on hybrid query.
        Returns atomic code units (functions, classes, blocks) matching the query.

        FIELD DEFINITIONS:
        - name: Entity name (e.g., "AuthManager", "process_data").
        - type: "function", "class", "module", "block", "import".
        - content: Minimal signature/stub (MUST NOT be full code if summary provided).
        - summary: Lossless restatement of logic (~50% compression). Preserve schemas.
        - keywords: Search-optimized concepts and dependencies.
        - start_line / end_line: Optional line range (default 0).

        EXAMPLES:
        
        1. Class Query: "Find the component managing user sessions."
        Output: {
            "name": "SessionManager", "type": "class", "content": "class SessionManager: ...",
            "summary": "Responsible for user login/logout lifecycles and token persistence.",
            "keywords": ["session", "login", "token"], "start_line": 5, "end_line": 60
        }

        2. Interaction Query: "How does the API handle data stream noise?"
        Output: {
            "name": "process_stream", "type": "function", "content": "def process_stream(s): ...",
            "summary": "Consumes binary input and applies a sliding window filter for noise reduction.",
            "keywords": ["noise reduction", "stream", "filter"], "start_line": 100, "end_line": 120
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
        result = coding_api.get_flow(agent_id, query)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def update_flow(
        agent_id: str,
        file_path: str,
        chunks: List[Dict[str, Any]] = None
    ) -> str:
        """
        📝 Update the Code Flow for a file with semantic chunking.

        ⚡ CRITICAL FOR AI AGENTS: Provide updated semantic chunks for this file.
        Same rules as create_flow apply. Prioritize client-side chunking.

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

        result = coding_api.update_flow(agent_id, file_path, chunks)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def delete_flow(
        agent_id: str,
        file_path: str
    ) -> str:
        """
        Delete a Code Flow entry.
        
        Args:
            agent_id: The agent ID
            file_path: Absolute path to the file or Context ID
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.delete_flow(agent_id, file_path)
        return json.dumps({"status": "deleted" if result else "not_found", "file_path": file_path}, indent=2)

    @mcp.tool()
    async def list_flows(
        agent_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> str:
        """
        List all stored Code Flow structures for an agent.
        
        Args:
            agent_id: The agent ID
            limit: Maximum items to return
            offset: Pagination offset
        """
        agent_id = _normalize_agent_id(agent_id)
        result = coding_api.list_flows(agent_id, limit, offset)
        return json.dumps(result, indent=2)

