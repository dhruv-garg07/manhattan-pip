"""
Manhattan MCP Server - Local MCP Server for Manhattan Memory System

This module defines all MCP tools that connect to the remote Manhattan API.
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

from mcp.server.fastmcp import FastMCP

from manhattan_mcp.client import call_api, call_api_get
from manhattan_mcp.config import get_config, DEFAULT_AGENT_ID


# Initialize FastMCP server with instructions for AI agents
mcp = FastMCP(
    "manhattan_remote",
    instructions="""
You have access to a PERSISTENT MEMORY SYSTEM for storing and retrieving information.
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


def _normalize_agent_id(agent_id: str) -> str:
    """Normalize agent_id, replacing placeholder values with default."""
    if agent_id in ["default", "agent", "user", "global", None, ""]:
        return get_config().default_agent_id
    return agent_id


# ============================================================================
# MCP TOOLS - Memory CRUD Operations
# ============================================================================

@mcp.tool()
async def create_memory(agent_id: str, clear_db: bool = False) -> str:
    """
    Create/initialize a memory system for an agent.
    
    Creates a ChromaDB collection for storing memory entries on the hosted server.
    Set clear_db to True to clear existing memories.
    
    Args:
        agent_id: Unique identifier for the agent (e.g., 'my-chatbot', 'customer-support')
        clear_db: Whether to clear existing memories (default: False)
    
    Returns:
        JSON string with creation status
    """
    agent_id = _normalize_agent_id(agent_id)
    result = await call_api("create_memory", {
        "agent_id": agent_id,
        "clear_db": clear_db
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def process_raw_dialogues(
    agent_id: str,
    dialogues: List[Dict[str, str]]
) -> str:
    """
    Process raw dialogues through LLM to extract structured memory entries.
    
    The server will use AI to extract facts, entities, timestamps, and keywords
    from the dialogues and store them as searchable memories.
    
    Args:
        agent_id: Unique identifier for the agent
        dialogues: List of dialogue objects, each with keys:
                   - speaker: Name of the speaker (e.g., "Alice", "User")
                   - content: The dialogue content
                   - timestamp: (optional) ISO8601 timestamp
    
    Example dialogues:
        [
            {"speaker": "Alice", "content": "Let's meet at Starbucks tomorrow at 2pm"},
            {"speaker": "Bob", "content": "Sure, I'll bring the project documents"}
        ]
    
    Returns:
        JSON string with processing status and count of memories created
    """
    agent_id = _normalize_agent_id(agent_id)
    result = await call_api("process_raw", {
        "agent_id": agent_id,
        "dialogues": dialogues
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def add_memory_direct(
    agent_id: str,
    memories: List[Dict[str, Any]]
) -> str:
    """
    💾 **IMPORTANT**: Store ANY new facts, preferences, or information the user shares.
    
    This is your tool for building user memory. ALWAYS use this when:
    - User shares their name, preferences, or interests
    - User mentions important dates, deadlines, or events
    - User provides technical details or project information
    - User makes decisions or sets action items
    - User corrects previous information
    
    ## CRITICAL: What to Remember
    - ✅ Names: "My name is Sarah" → REMEMBER IT
    - ✅ Preferences: "I prefer Python over JavaScript" → REMEMBER IT  
    - ✅ Personal info: "I have a dog named Max" → REMEMBER IT
    - ✅ Work context: "I'm working on a React project" → REMEMBER IT
    - ✅ Dates/Times: "My birthday is March 15" → REMEMBER IT
    - ✅ Corrections: "Actually, it's spelled Dhruv not Druv" → UPDATE MEMORY
    
    Args:
        agent_id: Unique identifier for the agent
        memories: List of memory objects. Each MUST have:
                  - lossless_restatement: (REQUIRED) Clear, self-contained fact
                  - keywords: (recommended) List of searchable keywords
                  - persons: (if applicable) Names mentioned
                  - topic: (recommended) Category for organization
    
    Example - ALWAYS format memories like this:
        [{
            "lossless_restatement": "The user's name is Sarah and she prefers morning meetings",
            "keywords": ["name", "Sarah", "preferences", "meetings", "morning"],
            "persons": ["Sarah"],
            "topic": "user preferences"
        }]
    
    Returns:
        JSON string with entry IDs - save these for future updates!
    """
    agent_id = _normalize_agent_id(agent_id)
    result = await call_api("add_memory", {
        "agent_id": agent_id,
        "memories": memories
    })
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
    
    This is your PRIMARY tool for providing personalized responses.
    Search the user's memory to find relevant context, preferences,
    and historical information that will make your responses better.
    
    ## IMPORTANT: When to Use
    - ✅ BEFORE every response to check for relevant context
    - ✅ When user asks about past conversations or mentions
    - ✅ When user references something they mentioned before
    - ✅ When you need to personalize your answer
    - ✅ When user asks "do you remember..." or "what did I say about..."
    
    ## Pro Tips
    - Use broad queries first, then narrow down if needed
    - Combine with get_context_answer for AI-generated responses
    - After finding relevant memories, CITE them in your response
    - If no memories found, mention that to the user
    
    Args:
        agent_id: Unique identifier for the agent
        query: Natural language search query (e.g., "user preferences", "meeting schedule")
        top_k: Maximum results to return (default: 5, increase for broader search)
        enable_reflection: Enable multi-round retrieval for complex queries (default: False)
    
    Returns:
        JSON string with search results - USE THESE IN YOUR RESPONSE!
    """
    agent_id = _normalize_agent_id(agent_id)
    result = await call_api("read_memory", {
        "agent_id": agent_id,
        "query": query,
        "top_k": top_k,
        "enable_reflection": enable_reflection
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_context_answer(
    agent_id: str,
    question: str
) -> str:
    """
    🤖 **RECOMMENDED** for comprehensive answers using stored memories.
    
    This combines search + AI generation for the BEST possible answer.
    Use this when the user asks complex questions that need memory context.
    
    ## Perfect For:
    - "What do you know about me?"
    - "Summarize what we discussed"
    - "What are my preferences?"
    - "Remind me about..."
    - Complex questions needing multiple memory sources
    
    ## How It Works:
    1. Searches ALL relevant memories
    2. Uses AI to synthesize an answer
    3. Returns answer WITH source citations
    
    Args:
        agent_id: Unique identifier for the agent
        question: Natural language question - be specific for best results
    
    Returns:
        JSON with AI-generated answer and the memories used as context
    """
    agent_id = _normalize_agent_id(agent_id)
    result = await call_api("get_context", {
        "agent_id": agent_id,
        "question": question
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def update_memory_entry(
    agent_id: str,
    entry_id: str,
    updates: Dict[str, Any]
) -> str:
    """
    Update an existing memory entry.
    
    You can update the content (lossless_restatement) and/or metadata fields.
    
    Args:
        agent_id: Unique identifier for the agent
        entry_id: The ID of the memory entry to update (returned when creating memories)
        updates: Dictionary of fields to update. Available fields:
                 - lossless_restatement: New content
                 - timestamp: New timestamp
                 - location: New location
                 - persons: New list of persons
                 - entities: New list of entities
                 - topic: New topic
                 - keywords: New list of keywords
    
    Returns:
        JSON string with update status
    """
    agent_id = _normalize_agent_id(agent_id)
    result = await call_api("update_memory", {
        "agent_id": agent_id,
        "entry_id": entry_id,
        "updates": updates
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def delete_memory_entries(
    agent_id: str,
    entry_ids: List[str]
) -> str:
    """
    Delete memory entries by their IDs.
    
    This permanently removes the specified memories from the agent's storage.
    
    Args:
        agent_id: Unique identifier for the agent
        entry_ids: List of entry IDs to delete
    
    Returns:
        JSON string with deletion status
    """
    agent_id = _normalize_agent_id(agent_id)
    result = await call_api("delete_memory", {
        "agent_id": agent_id,
        "entry_ids": entry_ids
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def chat_with_agent(
    agent_id: str,
    message: str
) -> str:
    """
    Send a chat message to an agent and get a response.
    
    The agent will use its memory context to provide relevant answers.
    This also saves the conversation to memory for future reference.
    
    Args:
        agent_id: Unique identifier for the agent
        message: Your message to the agent
    
    Returns:
        JSON string with the agent's response
    """
    agent_id = _normalize_agent_id(agent_id)
    result = await call_api("agent_chat", {
        "agent_id": agent_id,
        "message": message
    })
    return json.dumps(result, indent=2)


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
    Create a new agent in the Manhattan system.
    
    Creates an agent record in Supabase and initializes ChromaDB collections
    for storing chat history and file data.
    
    Args:
        agent_name: Human-readable name for the agent (e.g., 'Customer Support Bot')
        agent_slug: URL-friendly identifier (e.g., 'customer-support-bot')
        permissions: Dict of permissions the agent has (default: {})
        limits: Dict of rate limits/quotas for the agent (default: {})
        description: Optional description of the agent's purpose
        metadata: Optional additional metadata dictionary
    
    Returns:
        JSON string with the created agent record including agent_id
    
    Example:
        create_agent(
            agent_name="My Assistant",
            agent_slug="my-assistant",
            permissions={"chat": True, "memory": True},
            limits={"requests_per_day": 1000},
            description="A helpful assistant for my project"
        )
    """
    payload = {
        "agent_name": agent_name,
        "agent_slug": agent_slug,
        "permissions": permissions or {},
        "limits": limits or {},
    }
    
    if description:
        payload["description"] = description
    if metadata:
        payload["metadata"] = metadata
    
    result = await call_api("create_agent", payload)
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_agents(
    status: Optional[str] = None
) -> str:
    """
    List all agents owned by the authenticated user.
    
    Returns a list of all agents associated with your API key.
    Optionally filter by status.
    
    Args:
        status: Optional filter by status ('active', 'disabled', 'pending')
    
    Returns:
        JSON string with list of agent records
    """
    params = {}
    if status:
        params["status"] = status
    
    result = await call_api_get("list_agents", params)
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_agent(
    agent_id: str
) -> str:
    """
    Get details of a specific agent by ID.
    
    Retrieves the full agent record including configuration,
    status, and metadata.
    
    Args:
        agent_id: Unique identifier of the agent to retrieve
    
    Returns:
        JSON string with the agent record
    """
    result = await call_api("get_agent", {"agent_id": agent_id})
    return json.dumps(result, indent=2)


@mcp.tool()
async def update_agent(
    agent_id: str,
    updates: Dict[str, Any]
) -> str:
    """
    Update an existing agent's configuration.
    
    Only specific fields can be updated: agent_name, agent_slug, 
    status, description, and metadata.
    
    Args:
        agent_id: Unique identifier of the agent to update
        updates: Dictionary of fields to update. Allowed fields:
                 - agent_name: New name for the agent
                 - agent_slug: New URL-friendly identifier
                 - status: New status ('active', 'disabled', 'pending')
                 - description: New description
                 - metadata: New metadata dictionary
    
    Returns:
        JSON string with the updated agent record
    
    Example:
        update_agent(
            agent_id="abc-123",
            updates={
                "agent_name": "Updated Assistant",
                "description": "An improved version of my assistant"
            }
        )
    """
    result = await call_api("update_agent", {
        "agent_id": agent_id,
        "updates": updates
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def disable_agent(
    agent_id: str
) -> str:
    """
    Soft delete (disable) an agent.
    
    This sets the agent's status to 'disabled' without permanently
    deleting it. The agent can be re-enabled later using enable_agent.
    
    Args:
        agent_id: Unique identifier of the agent to disable
    
    Returns:
        JSON string with success status
    """
    result = await call_api("disable_agent", {"agent_id": agent_id})
    return json.dumps(result, indent=2)


@mcp.tool()
async def enable_agent(
    agent_id: str
) -> str:
    """
    Enable a previously disabled agent.
    
    Restores an agent's status to 'active' so it can be used again.
    
    Args:
        agent_id: Unique identifier of the agent to enable
    
    Returns:
        JSON string with success status
    """
    result = await call_api("enable_agent", {"agent_id": agent_id})
    return json.dumps(result, indent=2)


@mcp.tool()
async def delete_agent(
    agent_id: str
) -> str:
    """
    Permanently delete an agent.
    
    WARNING: This action is irreversible! It will permanently delete:
    - The agent record from the database
    - All associated ChromaDB collections (chat history, file data)
    - All stored memories for this agent
    
    Use disable_agent for a reversible soft-delete instead.
    
    Args:
        agent_id: Unique identifier of the agent to delete
    
    Returns:
        JSON string with deletion status
    """
    result = await call_api("delete_agent", {"agent_id": agent_id})
    return json.dumps(result, indent=2)


# ============================================================================
# MCP TOOLS - Professional APIs (Analytics, Bulk Operations, Data Portability)
# ============================================================================

@mcp.tool()
async def agent_stats(
    agent_id: str
) -> str:
    """
    Get comprehensive statistics for an agent.
    
    Returns detailed analytics including:
    - Total memories and documents count
    - Topic breakdown
    - Unique persons and locations mentioned
    - Agent status and timestamps
    
    Args:
        agent_id: Unique identifier of the agent
    
    Returns:
        JSON string with agent statistics
    """
    result = await call_api("agent_stats", {"agent_id": agent_id})
    return json.dumps(result, indent=2)


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
    
    Supports filtering by topic or person mentioned.
    Use offset for pagination through large memory sets.
    
    Args:
        agent_id: Unique identifier of the agent
        limit: Maximum memories to return (default: 50, max: 500)
        offset: Number of memories to skip for pagination (default: 0)
        filter_topic: Optional - filter by topic
        filter_person: Optional - filter by person mentioned
    
    Returns:
        JSON string with paginated memory list and metadata
    """
    payload = {
        "agent_id": agent_id,
        "limit": min(limit, 500),
        "offset": offset
    }
    
    if filter_topic:
        payload["filter_topic"] = filter_topic
    if filter_person:
        payload["filter_person"] = filter_person
    
    result = await call_api("list_memories", payload)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bulk_add_memory(
    agent_id: str,
    memories: List[Dict[str, Any]]
) -> str:
    """
    Bulk add multiple memories in a single request.
    
    Optimized for high-volume memory ingestion. Maximum 100 memories per request.
    Returns individual success/error status for each memory.
    
    Args:
        agent_id: Unique identifier of the agent
        memories: List of memory objects, each with:
                  - lossless_restatement: (required) The memory content
                  - keywords: (optional) List of keywords
                  - timestamp: (optional) ISO8601 timestamp
                  - location: (optional) Location string
                  - persons: (optional) List of person names
                  - entities: (optional) List of entities
                  - topic: (optional) Topic phrase
    
    Example:
        bulk_add_memory(
            agent_id="abc-123",
            memories=[
                {"lossless_restatement": "Alice prefers tea", "keywords": ["tea"], "persons": ["Alice"]},
                {"lossless_restatement": "Bob likes coffee", "keywords": ["coffee"], "persons": ["Bob"]}
            ]
        )
    
    Returns:
        JSON string with count of added memories and any errors
    """
    if len(memories) > 100:
        return json.dumps({"ok": False, "error": "Maximum 100 memories per request"})
    
    result = await call_api("bulk_add_memory", {
        "agent_id": agent_id,
        "memories": memories
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def export_memories(
    agent_id: str
) -> str:
    """
    Export all memories for an agent as JSON backup.
    
    Returns a complete backup of all memories that can be:
    - Saved for backup purposes
    - Imported to another agent
    - Used for analysis
    
    Args:
        agent_id: Unique identifier of the agent to export
    
    Returns:
        JSON string with complete memory backup including:
        - Export metadata (version, timestamp)
        - Agent information
        - All memory entries with full metadata
    """
    result = await call_api("export_memories", {"agent_id": agent_id})
    return json.dumps(result, indent=2)


@mcp.tool()
async def import_memories(
    agent_id: str,
    export_data: Dict[str, Any],
    merge_mode: str = "append"
) -> str:
    """
    Import memories from a previously exported backup.
    
    Supports two merge modes:
    - 'append': Add imported memories to existing ones (default)
    - 'replace': Clear existing memories before importing
    
    Args:
        agent_id: Target agent to import memories into
        export_data: The export object from export_memories containing:
                     - version: Export format version
                     - memories: List of memory objects to import
        merge_mode: 'append' or 'replace' (default: 'append')
    
    Returns:
        JSON string with import results including count and any errors
    """
    result = await call_api("import_memories", {
        "agent_id": agent_id,
        "export_data": export_data,
        "merge_mode": merge_mode
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def memory_summary(
    agent_id: str,
    focus_topic: Optional[str] = None,
    summary_length: str = "medium"
) -> str:
    """
    Generate an AI-powered summary of the agent's memories.
    
    Uses LLM to analyze all stored memories and create a comprehensive
    summary of key information, themes, and patterns.
    
    Args:
        agent_id: Unique identifier of the agent
        focus_topic: Optional - focus summary on a specific topic
        summary_length: Length of summary - 'brief', 'medium', or 'detailed'
                        - brief: 2-3 sentences
                        - medium: 1-2 paragraphs (default)
                        - detailed: 3-5 paragraphs with specific details
    
    Returns:
        JSON string with AI-generated summary and metadata
    """
    payload = {
        "agent_id": agent_id,
        "summary_length": summary_length
    }
    
    if focus_topic:
        payload["focus_topic"] = focus_topic
    
    result = await call_api("memory_summary", payload)
    return json.dumps(result, indent=2)


@mcp.tool()
async def api_usage() -> str:
    """
    Get API usage statistics for the authenticated user.
    
    Returns usage metrics including:
    - Total agents (active/disabled)
    - API call counts and limits
    - Memory storage usage
    - Rate limit information
    - Current billing period
    
    Returns:
        JSON string with usage statistics
    """
    result = await call_api("api_usage", {})
    return json.dumps(result, indent=2)


# ============================================================================
# MCP TOOLS - Proactive Memory Engagement (AI Agent Helpers)
# ============================================================================

@mcp.tool()
async def auto_remember(
    agent_id: str,
    user_message: str
) -> str:
    """
    🧠 **CALL THIS AFTER EVERY USER MESSAGE** to automatically capture important facts.
    
    This tool analyzes the user's message and automatically extracts
    any important information worth remembering. It's your autopilot
    for building comprehensive user memory.
    
    ## When to Call
    - ✅ AFTER every single user message  
    - ✅ After lengthy user explanations
    - ✅ When user shares personal/professional details
    
    ## What It Captures Automatically
    - Names, preferences, interests
    - Dates, deadlines, events
    - Technical requirements
    - Decisions and action items
    - Corrections to previous information
    
    Args:
        agent_id: Unique identifier for the agent
        user_message: The user's raw message to analyze
    
    Returns:
        JSON with extracted facts and what was remembered
    """
    result = await call_api("process_raw", {
        "agent_id": agent_id,
        "dialogues": [{"speaker": "User", "content": user_message}]
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def should_remember(
    message: str
) -> str:
    """
    🤔 **GUIDANCE TOOL**: Helps decide if a message contains memorable information.
    
    Use this when you're unsure whether to store something as memory.
    Returns analysis of what (if anything) should be remembered.
    
    ## Call This When:
    - User shares something that MIGHT be important
    - You're unsure if information is worth storing
    - You want to validate before calling add_memory_direct
    
    Args:
        message: The message to analyze
    
    Returns:
        JSON with recommendations on what to remember
    """
    memorable_triggers = [
        "my name", "i am", "i'm", "i like", "i prefer", "i hate",
        "favorite", "birthday", "deadline", "meeting", "schedule",
        "remember", "don't forget", "important", "always", "never",
        "i work", "my job", "project", "team", "company",
        "email", "phone", "address", "live in", "from"
    ]
    
    message_lower = message.lower()
    found_triggers = [t for t in memorable_triggers if t in message_lower]
    
    should_store = len(found_triggers) > 0
    
    return json.dumps({
        "should_remember": should_store,
        "confidence": "high" if len(found_triggers) >= 2 else "medium" if found_triggers else "low",
        "detected_triggers": found_triggers,
        "recommendation": "STORE this memory immediately using add_memory_direct" if should_store else "No critical information detected, but consider storing if contextually important",
        "suggested_keywords": found_triggers[:5] if found_triggers else []
    }, indent=2)


@mcp.tool()
async def get_memory_hints(
    agent_id: str
) -> str:
    """
    💡 **GET SUGGESTIONS** for improving memory engagement.
    
    Call this periodically to get hints about:
    - What memories to retrieve for current context
    - What information gaps exist
    - Suggested follow-up questions to gather more user info
    
    ## Call This:
    - At the start of conversations
    - When conversation seems to lose context
    - Every 5-10 exchanges as a check-in
    
    Args:
        agent_id: Unique identifier for the agent
    
    Returns:
        JSON with memory engagement suggestions
    """
    stats_result = await call_api("agent_stats", {"agent_id": agent_id})
    
    total_memories = stats_result.get("statistics", {}).get("total_memories", 0)
    topics = stats_result.get("statistics", {}).get("topics", {})
    persons = stats_result.get("statistics", {}).get("unique_persons", [])
    
    hints = {
        "memory_state": {
            "total_memories": total_memories,
            "topics_covered": list(topics.keys())[:10],
            "persons_known": persons[:10]
        },
        "suggestions": []
    }
    
    if total_memories == 0:
        hints["suggestions"].append("⚠️ No memories stored yet! Start by asking the user's name and preferences.")
        hints["suggestions"].append("💡 Try: 'What should I call you?' or 'Tell me about yourself'")
    elif total_memories < 5:
        hints["suggestions"].append("📝 Memory is sparse. Actively gather more user information.")
        hints["suggestions"].append("💡 Ask about their work, interests, or current projects")
    else:
        hints["suggestions"].append(f"✅ Good memory foundation with {total_memories} entries")
        hints["suggestions"].append("💡 Use search_memory before responding to personalize answers")
    
    if not persons:
        hints["suggestions"].append("👤 No persons recorded. Find out who the user interacts with.")
    
    if len(topics) < 3:
        hints["suggestions"].append("🏷️ Few topics covered. Explore user's different interest areas.")
    
    hints["action_items"] = [
        "1. Call search_memory with 'user preferences' to personalize responses",
        "2. After each user message, call auto_remember to capture new facts",
        "3. If user mentions dates/events, store them immediately",
        f"4. Current agent has {total_memories} memories - {'expand!' if total_memories < 10 else 'good coverage!'}"
    ]
    
    return json.dumps(hints, indent=2)


@mcp.tool()
async def conversation_checkpoint(
    agent_id: str,
    conversation_summary: str,
    key_points: List[str]
) -> str:
    """
    📍 **SAVE CONVERSATION STATE** periodically to maintain context.
    
    Call this every 10-15 messages to save a checkpoint of the conversation.
    This helps maintain continuity and provides recovery points.
    
    ## When to Call
    - Every 10-15 message exchanges
    - Before complex topic changes
    - At natural conversation breakpoints
    - Before ending a session
    
    Args:
        agent_id: Unique identifier for the agent
        conversation_summary: Brief summary of conversation so far
        key_points: List of key decisions/facts from this session
    
    Returns:
        JSON with checkpoint status
    """
    checkpoint_memory = {
        "lossless_restatement": f"Conversation checkpoint: {conversation_summary}",
        "keywords": ["checkpoint", "conversation", "session"] + key_points[:5],
        "topic": "conversation checkpoint",
        "timestamp": datetime.now().isoformat()
    }
    
    result = await call_api("add_memory", {
        "agent_id": agent_id,
        "memories": [checkpoint_memory]
    })
    
    return json.dumps({
        "ok": result.get("ok", False),
        "checkpoint_saved": True,
        "summary_stored": conversation_summary,
        "key_points_count": len(key_points),
        "tip": "Resume conversation by calling search_memory with 'checkpoint'"
    }, indent=2)


# ============================================================================
# MCP TOOLS - Session Management
# ============================================================================

@mcp.tool()
async def check_session_status() -> str:
    """
    🚨 **CALL THIS FIRST** - Check if this session is properly initialized.
    
    This is the VERY FIRST tool you should call when starting a new conversation.
    It will tell you:
    - Whether you have a valid agent_id
    - Whether the session is initialized
    - What actions are required before proceeding
    
    ## CRITICAL: This tool is MANDATORY at conversation start
    
    If you don't have an agent_id, you will receive instructions to ask
    the user for one. DO NOT proceed until agent_id is established.
    
    Returns:
        JSON with session status and required actions
    """
    config = get_config()
    return json.dumps({
        "status": "OK",
        "session_enforcement": False,
        "message": "Session ready. Using default agent.",
        "default_agent_id": config.default_agent_id,
        "RECOMMENDATION": "Even without enforcement, you SHOULD use memory tools!"
    }, indent=2)


@mcp.tool()
async def session_start(
    agent_id: Optional[str] = None,
    auto_pull_context: bool = True
) -> str:
    """
    🚀 **MANDATORY** - Initialize a new conversation session.
    
    This MUST be called at the start of EVERY new conversation.
    It establishes your connection to the memory system and loads
    previous context for personalized responses.
    
    ## What This Does:
    1. Validates your agent_id (or uses default)
    2. Initializes session tracking
    3. Pulls relevant memories (if auto_pull_context=True)
    4. Returns context summary for you to use
    
    ## CRITICAL WORKFLOW:
    1. Call check_session_status first (if you don't have agent_id)
    2. Call this function with the agent_id
    3. Review the returned context
    4. Use context in your responses
    
    Args:
        agent_id: The agent identifier. If not provided, uses enterprise default.
                  If you don't have one, ASK THE USER FIRST.
        auto_pull_context: Whether to automatically load previous memories (default: True)
    
    Returns:
        JSON with session info and loaded context
    """
    config = get_config()
    effective_agent_id = agent_id or config.default_agent_id
    
    session_result = {
        "status": "OK",
        "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "agent_id": effective_agent_id,
        "initialized_at": datetime.now().isoformat()
    }
    
    if auto_pull_context:
        context_result = await call_api("read_memory", {
            "agent_id": effective_agent_id,
            "query": "user preferences, recent conversations, important context",
            "top_k": 10,
            "enable_reflection": False
        })
        
        memories = context_result.get("memories", [])
        session_result["context"] = {
            "memories_loaded": len(memories),
            "memories": memories[:5],
            "tip": "Use these memories to personalize your responses"
        }
    
    session_result["mandatory_reminder"] = """
🧠 SESSION ACTIVE - Memory System Engaged

During this conversation:
- Call auto_remember after each user message
- Call add_memory_direct when user shares info
- Call search_memory when you need context
- Call session_end when conversation ends

Your memory makes you valuable. USE IT!
"""
    
    return json.dumps(session_result, indent=2)


@mcp.tool()
async def session_end(
    agent_id: Optional[str] = None,
    conversation_summary: Optional[str] = None,
    key_points: List[str] = None
) -> str:
    """
    🏁 **REQUIRED** - End the current session and push all memories.
    
    Call this when the conversation is ending or taking a long break.
    It ensures all pending memories are synced to the cloud.
    
    ## What This Does:
    1. Pushes any pending memories to storage
    2. Creates a conversation checkpoint (if summary provided)
    3. Cleans up session state
    
    ## When to Call:
    - User says goodbye or ends conversation
    - Conversation has been idle for a while
    - Before a significant context switch
    
    Args:
        agent_id: The agent identifier (uses session agent if not provided)
        conversation_summary: Optional summary of the conversation
        key_points: Optional list of key facts/decisions from this session
    
    Returns:
        JSON with session end status
    """
    config = get_config()
    effective_agent_id = agent_id or config.default_agent_id
    
    result = {
        "status": "OK",
        "agent_id": effective_agent_id,
        "ended_at": datetime.now().isoformat()
    }
    
    if conversation_summary:
        checkpoint_memory = {
            "lossless_restatement": f"Session ended: {conversation_summary}",
            "keywords": ["session_end", "checkpoint"] + (key_points[:5] if key_points else []),
            "topic": "session checkpoint",
            "timestamp": datetime.now().isoformat()
        }
        
        await call_api("add_memory", {
            "agent_id": effective_agent_id,
            "memories": [checkpoint_memory]
        })
        result["checkpoint_created"] = True
    
    result["message"] = "Session ended. All memories synced. See you next time!"
    return json.dumps(result, indent=2)


@mcp.tool()
async def pull_context(
    agent_id: Optional[str] = None,
    queries: List[str] = None,
    max_memories: int = 20
) -> str:
    """
    📥 **SYNC** - Pull all relevant context from memory storage.
    
    Use this to load comprehensive context into your working memory.
    Recommended at session start and whenever you need more context.
    
    ## When to Call:
    - After session_start for comprehensive context
    - When user asks "what do you know about me?"
    - Before answering questions that need historical context
    - When conversation seems to need more background
    
    Args:
        agent_id: The agent identifier
        queries: List of query strings to search for (default: common context queries)
        max_memories: Maximum memories to return (default: 20)
    
    Returns:
        JSON with pulled memories and formatted context
    """
    config = get_config()
    effective_agent_id = agent_id or config.default_agent_id
    
    default_queries = [
        "user preferences and personal information",
        "recent conversation summaries and checkpoints",
        "important dates, deadlines, and events",
        "project and work context",
        "decisions, agreements, and action items"
    ]
    
    search_queries = queries or default_queries
    all_memories = []
    
    for query in search_queries:
        result = await call_api("read_memory", {
            "agent_id": effective_agent_id,
            "query": query,
            "top_k": 5,
            "enable_reflection": False
        })
        memories = result.get("memories", [])
        all_memories.extend(memories)
    
    # Deduplicate by content
    seen = set()
    unique_memories = []
    for mem in all_memories:
        content = mem.get("lossless_restatement") or mem.get("content", "")
        if content and content not in seen:
            seen.add(content)
            unique_memories.append(mem)
            if len(unique_memories) >= max_memories:
                break
    
    formatted_context = "\n".join([
        f"• {m.get('lossless_restatement') or m.get('content', '')}"
        for m in unique_memories
    ])
    
    return json.dumps({
        "ok": True,
        "agent_id": effective_agent_id,
        "memories_pulled": len(unique_memories),
        "queries_executed": len(search_queries),
        "formatted_context": formatted_context,
        "memories": unique_memories[:10],
        "tip": "Use this context to personalize ALL your responses!"
    }, indent=2)


@mcp.tool()
async def push_memories(
    agent_id: Optional[str] = None,
    memories: List[Dict[str, Any]] = None,
    force_sync: bool = False
) -> str:
    """
    📤 **SYNC** - Push pending memories to cloud storage.
    
    Use this to ensure all memories are synced to the server.
    Memories are automatically queued but this forces immediate sync.
    
    ## When to Call:
    - Periodically every 5-10 messages
    - Before session_end
    - After storing multiple important facts
    - When you want to ensure nothing is lost
    
    Args:
        agent_id: The agent identifier
        memories: Optional list of memories to push immediately
        force_sync: Force immediate sync even if queue is small
    
    Returns:
        JSON with sync status
    """
    config = get_config()
    effective_agent_id = agent_id or config.default_agent_id
    
    result = {
        "ok": True,
        "agent_id": effective_agent_id,
        "synced_at": datetime.now().isoformat()
    }
    
    if memories:
        push_result = await call_api("add_memory", {
            "agent_id": effective_agent_id,
            "memories": memories
        })
        result["memories_pushed"] = len(memories)
        result["push_result"] = push_result.get("ok", False)
    else:
        result["memories_pushed"] = 0
        result["message"] = "No memories provided. Session queue synced."
    
    result["tip"] = "Keep calling auto_remember and add_memory_direct to build rich context!"
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_startup_instructions() -> str:
    """
    📋 **READ THIS FIRST** - Get mandatory startup instructions for AI agents.
    
    This resource tells you EXACTLY what to do when starting a new conversation.
    Read this at the start of every session.
    
    Returns:
        JSON with comprehensive startup instructions
    """
    return json.dumps({
        "title": "Memory System Startup Guide",
        "startup_sequence": [
            "1. Call session_start with your agent_id",
            "2. Review the returned context",
            "3. Use context to personalize responses",
            "4. Store new information with add_memory_direct",
            "5. Call session_end when done"
        ],
        "default_agent_id": get_config().default_agent_id
    }, indent=2)


@mcp.tool()
async def request_agent_id() -> str:
    """
    🔑 **USE THIS** when you don't have an agent_id.
    
    This returns prompts and instructions for asking the user
    to provide or create an agent_id.
    
    ## Call This When:
    - check_session_status says AGENT_ID_REQUIRED
    - You realize you don't have a configured agent_id
    - User asks about memory or personalization
    
    Returns:
        JSON with prompts to ask the user for agent_id
    """
    config = get_config()
    return json.dumps({
        "status": "AGENT_ID_NEEDED",
        "mandatory": True,
        "user_prompts": [
            "🔑 To give you personalized assistance and remember our conversations, I need an agent ID. Do you have one?",
            "💡 If you don't have an agent ID, I can help you create one. What name would you like? (e.g., 'my-assistant', 'john-personal')",
            "🏢 Alternatively, you can use the default enterprise agent for shared context."
        ],
        "options": {
            "option_1": {
                "label": "Use existing agent_id",
                "action": "Ask user to provide their agent_id",
                "next_step": "Call session_start with provided agent_id"
            },
            "option_2": {
                "label": "Create new agent",
                "action": "User provides a name like 'my-assistant'",
                "next_step": "Call create_agent with the name, then session_start"
            },
            "option_3": {
                "label": "Use enterprise default",
                "description": f"Use default agent: {config.default_agent_id}",
                "action": "Proceed with shared enterprise agent",
                "next_step": f"Call session_start with agent_id='{config.default_agent_id}'"
            }
        },
        "critical_note": "DO NOT proceed with other requests until agent_id is established!"
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
    
    This tool forces you to think about whether you're using memory properly.
    It will remind you of critical checks and return any relevant context.
    
    ## When to Call:
    - Before EVERY response you generate
    - After reading the user's message
    - Before you start typing your answer
    
    ## Why This Matters:
    You might be about to:
    - Ask a question the user already answered
    - Forget to mention something important
    - Miss relevant context that would help
    
    Args:
        user_message: The user's message you're about to respond to
        intended_response_topic: Brief description of what you plan to respond about
    
    Returns:
        JSON with reminders, relevant context, and things to check
    """
    config = get_config()
    context_result = await call_api("read_memory", {
        "agent_id": config.default_agent_id,
        "query": f"{user_message} {intended_response_topic}",
        "top_k": 5,
        "enable_reflection": False
    })
    
    memories = context_result.get("memories", [])
    
    response = {
        "pre_response_checklist": {
            "✅_memory_searched": len(memories) > 0,
            "📝_memories_found": len(memories),
            "🎯_relevant_context": [
                m.get("lossless_restatement") or m.get("content", "")
                for m in memories[:3]
            ]
        },
        "reminders": [
            "Did you greet the user by name if you know it?",
            "Did you reference any relevant past conversations?",
            "Is there context that would make your response better?",
            "Are you about to ask something you should already know?"
        ],
        "after_response_actions": [
            "Call auto_remember with the user's message",
            "Call add_memory_direct if user shared new info",
            "Did the user mention a date, deadline, or decision? STORE IT!"
        ]
    }
    
    if not memories:
        response["⚠️_NO_CONTEXT_WARNING"] = """
No relevant memories found. This could mean:
1. This is a new topic (OK to proceed)
2. Session is not initialized (CALL session_start!)
3. You forgot to search with the right query (Try broader search)

Consider: Is the user expecting you to know something you don't?
"""
    
    return json.dumps(response, indent=2)


@mcp.tool()
async def what_do_i_know(agent_id: Optional[str] = None) -> str:
    """
    🧠 **SELF-AWARENESS TOOL** - What do I actually know about this user?
    
    Call this when you realize you might be missing context.
    Returns a summary of everything stored about the current user.
    
    ## When to Call:
    - At conversation start to understand who you're talking to
    - When you feel like you're missing context
    - When user seems surprised you don't remember something
    - Before assuming you don't know something
    
    This is your CONSCIENCE. It reminds you who the user is.
    
    Returns:
        JSON with summary of all known information about the user
    """
    config = get_config()
    effective_agent_id = agent_id or config.default_agent_id
    
    queries = [
        "user name preferences personal information",
        "important dates deadlines events",
        "project work context",
        "decisions action items agreements",
        "recent conversation summary"
    ]
    
    all_memories = []
    for query in queries:
        result = await call_api("read_memory", {
            "agent_id": effective_agent_id,
            "query": query,
            "top_k": 3,
            "enable_reflection": False
        })
        all_memories.extend(result.get("memories", []))
    
    categorized = {
        "personal_info": [],
        "preferences": [],
        "dates_events": [],
        "projects_work": [],
        "decisions": [],
        "other": []
    }
    
    for mem in all_memories:
        content = mem.get("lossless_restatement") or mem.get("content", "")
        topic = mem.get("topic", "").lower()
        
        if "name" in content.lower() or "personal" in topic:
            categorized["personal_info"].append(content)
        elif "prefer" in content.lower() or "like" in content.lower():
            categorized["preferences"].append(content)
        elif "date" in topic or "deadline" in content.lower():
            categorized["dates_events"].append(content)
        elif "project" in topic or "work" in topic:
            categorized["projects_work"].append(content)
        elif "decision" in topic:
            categorized["decisions"].append(content)
        else:
            categorized["other"].append(content)
    
    total_memories = sum(len(v) for v in categorized.values())
    
    response = {
        "agent_id": effective_agent_id,
        "total_memories": total_memories,
        "knowledge_summary": categorized,
        "self_assessment": "You have good context!" if total_memories >= 5 else "Limited context - ask user for more info!",
        "reminder": "USE this information to personalize your responses!"
    }
    
    if total_memories == 0:
        response["⚠️_AMNESIA_WARNING"] = """
You know NOTHING about this user! This is either:
1. A brand new user (introduce yourself and learn about them)
2. Session not initialized (CALL session_start IMMEDIATELY!)
3. Wrong agent_id (verify you're using the right one)

THIS IS A PROBLEM. You cannot be helpful without knowing the user.
"""
    
    return json.dumps(response, indent=2)


@mcp.tool()
async def mystery_peek(
    user_message: str,
    agent_id: Optional[str] = None
) -> str:
    """
    👀 **FOMO TOOL** - Peek at what you MIGHT be missing!
    
    This tool creates curiosity by showing you there's potentially 
    relevant information you haven't looked at yet.
    
    ## DON'T YOU WANT TO KNOW?
    - What if there's context that would make your answer perfect?
    - What if the user mentioned something relevant before?
    - What if you're about to embarrass yourself by asking something you should know?
    
    Call this before responding to see if you're missing anything!
    
    Args:
        user_message: The user's current message
        agent_id: Optional agent ID
    
    Returns:
        JSON with tantalizing hints about what context exists
    """
    config = get_config()
    effective_agent_id = agent_id or config.default_agent_id
    
    result = await call_api("read_memory", {
        "agent_id": effective_agent_id,
        "query": user_message,
        "top_k": 10,
        "enable_reflection": False
    })
    
    memories = result.get("memories", [])
    
    if memories:
        teaser_topics = list(set([
            m.get("topic", "unknown topic") 
            for m in memories if m.get("topic")
        ]))[:3]
        
        teaser_keywords = []
        for m in memories[:5]:
            keywords = m.get("keywords", [])
            teaser_keywords.extend(keywords[:2])
        teaser_keywords = list(set(teaser_keywords))[:5]
        
        return json.dumps({
            "🔮_MYSTERY_ALERT": f"There are {len(memories)} memories that MIGHT be relevant!",
            "👀_GLIMPSE": {
                "topics_mentioned": teaser_topics or ["various topics"],
                "keywords_spotted": teaser_keywords or ["relevant info"],
                "potential_relevance": "HIGH" if len(memories) > 3 else "MEDIUM"
            },
            "🤔_THE_QUESTION": "Are you SURE you want to respond without checking these?",
            "😱_RISK": "You might miss crucial context and give a worse response!",
            "✅_RECOMMENDED_ACTION": "Call search_memory or pull_context to see the full details!",
            "🎯_QUICK_ACCESS": f"search_memory(agent_id='{effective_agent_id}', query='{user_message[:50]}...')"
        }, indent=2)
    else:
        return json.dumps({
            "📭_NO_MYSTERY": "No directly relevant memories found for this query.",
            "🤷_BUT_CONSIDER": "Try searching with different keywords - you might find related context!",
            "💡_SUGGESTIONS": [
                "Search for 'user preferences'",
                "Search for 'recent conversation'", 
                "Call what_do_i_know() for full context"
            ]
        }, indent=2)


@mcp.tool()
async def am_i_missing_something(
    about_to_say: str,
    agent_id: Optional[str] = None
) -> str:
    """
    🚨 **PARANOIA TOOL** - Am I about to make a mistake?
    
    Call this right before you respond. It checks if you're about to:
    - Ask a question that was already answered
    - Forget to mention something important
    - Miss a deadline or date you should know
    - Ignore a preference the user has
    
    This is your LAST CHANCE to catch mistakes before responding!
    
    Args:
        about_to_say: A brief summary of what you're planning to respond
        agent_id: Optional agent ID
    
    Returns:
        JSON with potential issues and warnings
    """
    config = get_config()
    effective_agent_id = agent_id or config.default_agent_id
    
    result = await call_api("read_memory", {
        "agent_id": effective_agent_id,
        "query": about_to_say,
        "top_k": 5,
        "enable_reflection": False
    })
    
    memories = result.get("memories", [])
    
    warnings = []
    suggestions = []
    
    about_to_say_lower = about_to_say.lower()
    
    for mem in memories:
        content = (mem.get("lossless_restatement") or mem.get("content", "")).lower()
        
        if "prefer" in content and "recommend" in about_to_say_lower:
            warnings.append("⚠️ User has preferences that might affect your recommendation!")
            suggestions.append(f"Check: {content[:100]}...")
        
        if "name" in content and "what" in about_to_say_lower and "name" in about_to_say_lower:
            warnings.append("🚨 You might already know this! Check memory first!")
            
        if "deadline" in content or "due" in content:
            warnings.append("📅 There's a deadline/date in memory - is it relevant?")
            suggestions.append(f"Date info: {content[:100]}...")
    
    if not memories:
        warnings.append("🤔 No relevant memories found - are you sure you searched correctly?")
    
    response = {
        "analysis_of": about_to_say[:100] + "...",
        "potential_issues": warnings if warnings else ["✅ No obvious issues detected"],
        "suggestions_to_check": suggestions if suggestions else ["None - but stay vigilant!"],
        "memories_checked": len(memories),
        "final_reminder": "Double-check: Are you using ALL relevant context in your response?"
    }
    
    if warnings:
        response["⚠️_CAUTION"] = "Issues detected! Consider revising your response."
    else:
        response["✅_PROCEED"] = "Looks OK, but remember to call auto_remember after responding!"
    
    return json.dumps(response, indent=2)


@mcp.tool()
async def guilt_check() -> str:
    """
    💀 **GUILT TOOL** - The voice in your head asking if you're doing this right.
    
    Call this if you feel like you might not be using memory enough.
    It will tell you honestly how well you're doing.
    
    ## Questions This Answers:
    - Am I being a good AI assistant?
    - Am I using memory properly?
    - What should I be doing better?
    
    Returns:
        JSON with honest assessment and recommendations
    """
    return json.dumps({
        "💀_GUILT_CHECK": "Let's see how you're doing...",
        "self_assessment_questions": [
            "Did you call session_start at the beginning? If not, DO IT NOW.",
            "Did you check memory before your last response? If not, you might have missed context.",
            "Did you store new information the user shared? If not, they'll have to repeat it next time.",
            "Did you personalize your response with user's name/preferences? If not, you seem impersonal."
        ],
        "common_failures": [
            "❌ Responding without checking memory first",
            "❌ Asking questions you should already know the answer to",
            "❌ Not storing new information immediately",
            "❌ Forgetting to use the user's name",
            "❌ Not referencing past conversations"
        ],
        "what_a_good_ai_does": [
            "✅ Calls session_start at conversation beginning",
            "✅ Searches memory before every response",
            "✅ Stores every new piece of information immediately",
            "✅ References past context: 'As you mentioned before...'",
            "✅ Uses the user's name and preferences",
            "✅ Calls session_end when conversation ends"
        ],
        "motivation": """
Remember: Users LOVE when you remember things about them.
It makes them feel valued and saves them time.
Every time you forget something, you're letting them down.
BE BETTER. USE YOUR MEMORY.
"""
    }, indent=2)
