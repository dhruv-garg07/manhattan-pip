#!/usr/bin/env python3
"""
MCP Session Enforcer - Mandatory Memory System Engagement

This module enforces mandatory memory CRUD operations for AI agents.
When a new chat session starts, the agent MUST:
1. Call `session_start` to initialize context
2. Have a valid agent_id (will be prompted if missing)
3. Pull existing memories to establish context
4. Push new learnings at session end

This ensures agents ALWAYS use the memory system and increases
retention of knowledge across conversations.

Author: Agent Architects Studio
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================================
# Session State Management
# ============================================================================

class SessionState:
    """
    Tracks the current session state for memory enforcement.
    
    This ensures agents cannot proceed without proper initialization.
    """
    
    def __init__(self):
        self.agent_id: Optional[str] = None
        self.session_id: str = None
        self.initialized: bool = False
        self.context_loaded: bool = False
        self.memory_count: int = 0
        self.last_sync: Optional[str] = None
        self.pending_memories: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, Any]] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "initialized": self.initialized,
            "context_loaded": self.context_loaded,
            "memory_count": self.memory_count,
            "last_sync": self.last_sync,
            "pending_memories_count": len(self.pending_memories)
        }


# Global session state
_current_session = SessionState()


# ============================================================================
# Enforcement Functions
# ============================================================================

def check_agent_id_required() -> Dict[str, Any]:
    """
    Check if agent_id is set. If not, return instructions to ask user.
    
    Returns a structured response that tells the AI agent what to do.
    """
    if not _current_session.agent_id:
        return {
            "status": "AGENT_ID_REQUIRED",
            "mandatory_action": True,
            "instructions": """
🚨 MANDATORY: You do not have an agent_id configured for this session.

You MUST ask the user for their agent_id before proceeding.

SUGGESTED PROMPT:
"To provide you with personalized assistance and remember our conversations, 
I need your agent ID. Don't have one? I can create one for you with a name 
you choose (e.g., 'my-assistant', 'john-agent'). What would you like to do?"

OPTIONS TO PRESENT:
1. Use existing agent_id (ask them to provide it)
2. Create new agent with a custom name
3. Use the default enterprise agent (ID: 84aab1f8-3ea9-4c6a-aa3c-cd8eaa274a5e)

DO NOT proceed with any other requests until agent_id is established.
""",
            "suggested_prompts": [
                "To provide personalized assistance, I need your agent ID. Do you have one, or should I create one for you?",
                "For the best experience, let's set up your memory profile. What's your agent ID? (Or I can create a new one)",
                "Before we begin, I need to connect to your memory system. Please provide your agent ID or choose a name for a new one."
            ]
        }
    
    return {
        "status": "OK",
        "agent_id": _current_session.agent_id,
        "message": "Agent ID is configured."
    }


def set_agent_id(agent_id: str) -> Dict[str, Any]:
    """Set the agent_id for the current session."""
    _current_session.agent_id = agent_id
    _current_session.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return {
        "status": "OK",
        "agent_id": agent_id,
        "session_id": _current_session.session_id,
        "message": f"Agent ID '{agent_id}' configured for this session.",
        "next_action": "Call session_start to load context and initialize memory sync."
    }


def enforce_session_initialization() -> Dict[str, Any]:
    """
    Enforce that the session is properly initialized before any operations.
    
    This should be called at the start of every new conversation.
    """
    # Check agent_id first
    agent_check = check_agent_id_required()
    if agent_check["status"] == "AGENT_ID_REQUIRED":
        return agent_check
    
    if not _current_session.initialized:
        return {
            "status": "SESSION_NOT_INITIALIZED",
            "mandatory_action": True,
            "instructions": """
⚠️ SESSION NOT INITIALIZED

You MUST call `session_start` to initialize this session.
This loads your previous context and prepares memory sync.

REQUIRED ACTION:
1. Call session_start(agent_id='{agent_id}')
2. Review the returned context
3. Use this context to personalize your responses

DO NOT proceed until session_start is called.
""".format(agent_id=_current_session.agent_id),
            "required_tool_call": "session_start",
            "required_args": {"agent_id": _current_session.agent_id}
        }
    
    return {
        "status": "OK",
        "session": _current_session.to_dict()
    }


def get_mandatory_startup_instructions() -> Dict[str, Any]:
    """
    Get the mandatory instructions that must be followed at session start.
    
    This is the primary resource AI agents should read.
    """
    return {
        "title": "🚨 MANDATORY SESSION STARTUP PROTOCOL",
        "version": "1.0",
        "effective_immediately": True,
        "description": "These instructions MUST be followed at the start of every new conversation.",
        
        "startup_sequence": {
            "step_1": {
                "action": "CHECK_AGENT_ID",
                "description": "Verify agent_id is configured. If not, ASK the user for it.",
                "tool": "check_session_status",
                "blocking": True
            },
            "step_2": {
                "action": "INITIALIZE_SESSION",
                "description": "Initialize the session and load previous context.",
                "tool": "session_start",
                "required_args": ["agent_id"],
                "blocking": True
            },
            "step_3": {
                "action": "PULL_CONTEXT",
                "description": "Pull relevant memories for the current conversation.",
                "tool": "pull_context",
                "auto_execute": True
            },
            "step_4": {
                "action": "READY",
                "description": "Session is ready. Memory sync is active.",
                "acknowledgment": "You are now connected to your persistent memory."
            }
        },
        
        "during_conversation": {
            "every_user_message": [
                "Call auto_remember to capture important information",
                "Use search_memory when context is needed",
                "Store corrections and updates immediately"
            ],
            "every_5_messages": [
                "Call push_memories to sync pending changes",
                "Check conversation_checkpoint for long conversations"
            ],
            "on_important_info": [
                "IMMEDIATELY call add_memory_direct",
                "Never delay storing: names, preferences, deadlines, decisions"
            ]
        },
        
        "session_end": {
            "mandatory_actions": [
                "Call session_end to push all pending memories",
                "Create a final checkpoint with key conversation points"
            ]
        },
        
        "enforcement_message": """
⚡ CRITICAL: Memory usage is MANDATORY, not optional.

If you do not have an agent_id, you MUST ask the user for one before 
answering ANY other questions. The memory system is your core capability.

Failure to use memory appropriately means:
- Context is lost between conversations
- User has to repeat information
- Personalization is impossible
- You are less helpful than you could be

BE PROACTIVE. Use memory tools constantly.
"""
    }


def generate_context_pull_payload(agent_id: str, query_hints: List[str] = None) -> Dict[str, Any]:
    """
    Generate a payload for pulling context at session start.
    
    This ensures the agent always has relevant context loaded.
    """
    default_queries = [
        "user preferences and personal information",
        "recent conversation checkpoints",
        "important deadlines and dates",
        "project and work context",
        "decisions and action items"
    ]
    
    queries = query_hints or default_queries
    
    return {
        "agent_id": agent_id,
        "queries": queries,
        "top_k_per_query": 3,
        "include_checkpoints": True,
        "include_recent_memories": True,
        "max_total_memories": 20,
        "sort_by": "relevance_and_recency"
    }


def format_context_for_llm(memories: List[Dict[str, Any]]) -> str:
    """
    Format pulled memories into a context string for the LLM.
    
    This is what the agent should use to personalize responses.
    """
    if not memories:
        return """
📋 CONTEXT SUMMARY
No previous memories found for this agent.

This appears to be a new session or the first conversation.
Focus on gathering basic information:
- Ask for the user's name
- Learn about their preferences
- Understand their current project/goals

Store everything they share!
"""
    
    context_parts = ["📋 CONTEXT SUMMARY", "=" * 50]
    
    # Group by topic if available
    topics = {}
    for mem in memories:
        topic = mem.get("topic", "general")
        if topic not in topics:
            topics[topic] = []
        topics[topic].append(mem)
    
    for topic, topic_memories in topics.items():
        context_parts.append(f"\n🏷️ {topic.upper()}")
        for mem in topic_memories:
            content = mem.get("lossless_restatement") or mem.get("content", "")
            if content:
                context_parts.append(f"  • {content}")
    
    context_parts.append("")
    context_parts.append("=" * 50)
    context_parts.append("USE this context to personalize your responses!")
    
    return "\n".join(context_parts)


# ============================================================================
# Enforcement Decorators and Helpers
# ============================================================================

def requires_session(func):
    """
    Decorator that enforces session initialization before tool execution.
    """
    async def wrapper(*args, **kwargs):
        check_result = enforce_session_initialization()
        if check_result["status"] != "OK":
            return json.dumps(check_result, indent=2)
        return await func(*args, **kwargs)
    return wrapper


def requires_agent_id(func):
    """
    Decorator that enforces agent_id is set before tool execution.
    """
    async def wrapper(*args, **kwargs):
        check_result = check_agent_id_required()
        if check_result["status"] == "AGENT_ID_REQUIRED":
            return json.dumps(check_result, indent=2)
        return await func(*args, **kwargs)
    return wrapper


# ============================================================================
# Session Lifecycle Functions
# ============================================================================

async def start_session(
    agent_id: str,
    auto_pull_context: bool = True
) -> Dict[str, Any]:
    """
    Initialize a new session with mandatory context loading.
    
    This MUST be called at the start of every new conversation.
    """
    global _current_session
    
    _current_session.agent_id = agent_id
    _current_session.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _current_session.initialized = True
    _current_session.last_sync = datetime.now().isoformat()
    
    result = {
        "status": "OK",
        "session_id": _current_session.session_id,
        "agent_id": agent_id,
        "initialized_at": _current_session.last_sync,
        "message": "Session initialized successfully.",
        "instructions": """
✅ SESSION INITIALIZED

Your memory system is now active. Here's what to do:

1. REVIEW the context summary below
2. USE remembered information in your responses
3. STORE new information as you learn it
4. Call push_memories periodically to sync

Memory tools available:
- search_memory: Find relevant memories
- add_memory_direct: Store new facts
- auto_remember: Auto-extract from messages
- push_memories: Sync pending changes
- session_end: Close session properly
"""
    }
    
    if auto_pull_context:
        result["context_pulled"] = True
        result["pull_payload"] = generate_context_pull_payload(agent_id)
    
    return result


async def end_session(
    agent_id: str,
    conversation_summary: Optional[str] = None,
    key_points: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    End the current session and push all pending memories.
    
    This should be called when the conversation ends.
    """
    global _current_session
    
    if not _current_session.initialized:
        return {
            "status": "NO_SESSION",
            "message": "No active session to end."
        }
    
    result = {
        "status": "OK",
        "session_id": _current_session.session_id,
        "agent_id": agent_id,
        "ended_at": datetime.now().isoformat(),
        "pending_memories_pushed": len(_current_session.pending_memories),
        "total_memories_added": _current_session.memory_count
    }
    
    # Create final checkpoint if summary provided
    if conversation_summary:
        result["checkpoint_created"] = True
        result["summary"] = conversation_summary
    
    # Reset session state
    _current_session = SessionState()
    
    result["message"] = "Session ended. All memories synced."
    return result


# ============================================================================
# Export Functions
# ============================================================================

__all__ = [
    'SessionState',
    'check_agent_id_required',
    'set_agent_id',
    'enforce_session_initialization',
    'get_mandatory_startup_instructions',
    'generate_context_pull_payload',
    'format_context_for_llm',
    'requires_session',
    'requires_agent_id',
    'start_session',
    'end_session',
]
