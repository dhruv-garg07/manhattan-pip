"""
MCP Server for Manhattan Memory CRUD APIs

This MCP (Model Context Protocol) server exposes the Memory CRUD functionality
to Claude and other MCP-compatible clients. It provides tools for:
- Creating memory systems for agents
- Processing raw dialogues through LLM
- Adding memories directly (without LLM)
- Reading/searching memories using hybrid retrieval
- Getting context-aware answers
- Updating existing memories
- Deleting memories

Usage:
    Run with: python mcp_memory_server.py
    Or with uv: uv run mcp_memory_server.py

Configuration for Claude Desktop (claude_desktop_config.json):
{
    "mcpServers": {
        "manhattan-memory": {
            "command": "python",
            "args": ["<ABSOLUTE_PATH>/mcp_memory_server.py"]
        }
    }
}
"""

import os
import sys
import json
from typing import Any, Optional, List, Dict

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add lib directory
lib_dir = os.path.join(parent_dir, 'lib')
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import MCP SDK
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: mcp package not installed. Install with: pip install mcp")
    print("Or: uv add mcp")
    sys.exit(1)

# Import SimpleMem components
try:
    from SimpleMem.main import create_system, SimpleMemSystem
    from SimpleMem.models.memory_entry import MemoryEntry, Dialogue
except ImportError as e:
    print(f"Error importing SimpleMem: {e}")
    print("Make sure SimpleMem module is available in the path")
    sys.exit(1)

# Initialize FastMCP server
mcp = FastMCP(
    "manhattan-memory",
    instructions="""
You have access to a PERSISTENT MEMORY SYSTEM for storing and retrieving information.
This allows you to provide a more personalized and consistent experience.

PROACTIVE MEMORY USAGE:
1.  **SAVE IMPORTANT INFO**: If the user shares facts, preferences, or project details, store them.
2.  **CHECK CONTEXT**: Before answering context-heavy questions, check if you already know the answer.

This system is designed to HELP you, not to block you. Use it when it improves your response quality.

⚙️ CONFIGURATION:
-   `agent_id`: Optional. defaults to the enterprise agent (ID: `84aab1f8-3ea9-4c6a-aa3c-cd8eaa274a5e`) if not specified.
"""
)

# Cache for SimpleMem systems per agent
_memory_systems_cache: Dict[str, SimpleMemSystem] = {}

# Current agent context for this session
_current_agent_id: Optional[str] = None


# ============================================================================
# MCP Agents Service - CRUD for managing agent_ids in Supabase
# ============================================================================

class McpAgentsService:
    """
    Service class for CRUD operations on the `mcp_agents` table.
    Each agent represents a memory context that users can switch between.
    """
    
    TABLE_NAME = "mcp_agents"
    
    def __init__(self):
        from supabase import create_client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            self.client = None
            print("Warning: Supabase credentials not set. Agent management will work locally only.")
        else:
            self.client = create_client(supabase_url, supabase_key)
    
    def create_agent(
        self,
        user_id: str,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new MCP agent."""
        import uuid
        from datetime import datetime
        
        record = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "agent_id": agent_id,
            "name": name or agent_id,
            "description": description or "",
            "metadata": json.dumps(metadata) if metadata else "{}",
            "status": "active",
            "is_current": False,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        if self.client:
            # Check if agent_id already exists for this user
            existing = self.client.table(self.TABLE_NAME).select("id").eq("user_id", user_id).eq("agent_id", agent_id).execute()
            if existing.data:
                raise ValueError(f"Agent '{agent_id}' already exists for this user")
            
            res = self.client.table(self.TABLE_NAME).insert(record).execute()
            if res.data:
                return res.data[0]
        
        return record  # Fallback for local testing
    
    def get_agent(self, user_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent by agent_id."""
        if self.client:
            res = self.client.table(self.TABLE_NAME).select("*").eq("user_id", user_id).eq("agent_id", agent_id).limit(1).execute()
            if res.data:
                agent = res.data[0]
                if agent.get("metadata") and isinstance(agent["metadata"], str):
                    try:
                        agent["metadata"] = json.loads(agent["metadata"])
                    except:
                        pass
                return agent
        return None
    
    def list_agents(self, user_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all agents for a user."""
        if self.client:
            query = self.client.table(self.TABLE_NAME).select("*").eq("user_id", user_id)
            if status:
                query = query.eq("status", status)
            res = query.order("created_at", desc=True).execute()
            agents = res.data or []
            for agent in agents:
                if agent.get("metadata") and isinstance(agent["metadata"], str):
                    try:
                        agent["metadata"] = json.loads(agent["metadata"])
                    except:
                        pass
            return agents
        return []
    
    def update_agent(self, user_id: str, agent_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an agent."""
        from datetime import datetime
        
        if not updates:
            raise ValueError("No updates provided")
        
        updates["updated_at"] = datetime.utcnow().isoformat()
        
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])
        
        if self.client:
            res = self.client.table(self.TABLE_NAME).update(updates).eq("user_id", user_id).eq("agent_id", agent_id).execute()
            if res.data:
                agent = res.data[0]
                if agent.get("metadata") and isinstance(agent["metadata"], str):
                    try:
                        agent["metadata"] = json.loads(agent["metadata"])
                    except:
                        pass
                return agent
        return None
    
    def delete_agent(self, user_id: str, agent_id: str) -> bool:
        """Delete an agent."""
        if self.client:
            res = self.client.table(self.TABLE_NAME).delete().eq("user_id", user_id).eq("agent_id", agent_id).execute()
            return bool(res.data)
        return False
    
    def set_current_agent(self, user_id: str, agent_id: str) -> bool:
        """Set an agent as the current/default for the user."""
        from datetime import datetime
        
        if self.client:
            # Clear is_current from all agents
            self.client.table(self.TABLE_NAME).update({"is_current": False}).eq("user_id", user_id).execute()
            # Set is_current for this agent
            res = self.client.table(self.TABLE_NAME).update({
                "is_current": True, 
                "updated_at": datetime.utcnow().isoformat()
            }).eq("user_id", user_id).eq("agent_id", agent_id).execute()
            return bool(res.data)
        return False
    
    def get_current_agent(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the current/default agent for the user."""
        if self.client:
            # Try to get current agent
            res = self.client.table(self.TABLE_NAME).select("*").eq("user_id", user_id).eq("is_current", True).limit(1).execute()
            if res.data:
                agent = res.data[0]
            else:
                # Fallback to most recent
                res = self.client.table(self.TABLE_NAME).select("*").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
                if res.data:
                    agent = res.data[0]
                else:
                    return None
            
            if agent.get("metadata") and isinstance(agent["metadata"], str):
                try:
                    agent["metadata"] = json.loads(agent["metadata"])
                except:
                    pass
            return agent
        return None


# Initialize the agents service
_agents_service = McpAgentsService()

# Default user_id for MCP (can be overridden by environment variable)
_default_user_id = os.getenv("MCP_USER_ID", "mcp-default-user")


def _get_or_create_memory_system(agent_id: str, clear_db: bool = False) -> SimpleMemSystem:
    """Get cached SimpleMem system or create new one for the agent."""
    if agent_id not in _memory_systems_cache or clear_db:
        _memory_systems_cache[agent_id] = create_system(agent_id=agent_id, clear_db=clear_db)
    return _memory_systems_cache[agent_id]


# ============================================================================
# MCP TOOLS - Agent Management (CRUD for mcp_agents)
# ============================================================================

@mcp.tool()
async def register_agent(
    agent_id: str,
    name: str = None,
    description: str = ""
) -> str:
    """
    Register a new memory agent in the system.
    
    Each agent has its own separate memory space. Use different agents
    for different projects, contexts, or purposes.
    
    Args:
        agent_id: Unique identifier (e.g., 'project-notes', 'daily-journal')
                  Must contain only letters, numbers, hyphens, and underscores.
        name: Human-readable name (optional, defaults to agent_id)
        description: Description of the agent's purpose (optional)
    
    Returns:
        JSON string with the created agent details
    """
    global _current_agent_id
    
    try:
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', agent_id):
            return json.dumps({
                'ok': False, 
                'error': 'agent_id must contain only alphanumeric characters, hyphens, and underscores'
            })
        
        agent = _agents_service.create_agent(
            user_id=_default_user_id,
            agent_id=agent_id,
            name=name or agent_id,
            description=description
        )
        
        # Set as current agent
        _current_agent_id = agent_id
        _agents_service.set_current_agent(_default_user_id, agent_id)
        
        # Also initialize the memory system
        _get_or_create_memory_system(agent_id)
        
        return json.dumps({
            'ok': True,
            'message': 'agent_registered',
            'agent': agent,
            'current_agent': agent_id
        })
    except ValueError as e:
        return json.dumps({'ok': False, 'error': str(e)})
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


@mcp.tool()
async def list_my_agents() -> str:
    """
    List all your registered memory agents.
    
    Shows all agents you've created with their names, descriptions,
    and status. Use this to see what memory contexts are available.
    
    Returns:
        JSON string with list of all your agents
    """
    try:
        agents = _agents_service.list_agents(user_id=_default_user_id)
        return json.dumps({
            'ok': True,
            'agents': agents,
            'count': len(agents),
            'current_agent': _current_agent_id
        })
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


@mcp.tool()
async def get_agent_details(agent_id: str) -> str:
    """
    Get details about a specific memory agent.
    
    Args:
        agent_id: The agent identifier to look up
    
    Returns:
        JSON string with agent details (name, description, metadata, etc.)
    """
    try:
        agent = _agents_service.get_agent(user_id=_default_user_id, agent_id=agent_id)
        if agent:
            return json.dumps({'ok': True, 'agent': agent})
        return json.dumps({'ok': False, 'error': 'agent_not_found'})
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


@mcp.tool()
async def update_agent_info(
    agent_id: str,
    name: str = None,
    description: str = None
) -> str:
    """
    Update a memory agent's details.
    
    Args:
        agent_id: The agent to update
        name: New name (optional)
        description: New description (optional)
    
    Returns:
        JSON string with updated agent details
    """
    try:
        updates = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        
        if not updates:
            return json.dumps({'ok': False, 'error': 'No updates provided'})
        
        agent = _agents_service.update_agent(
            user_id=_default_user_id,
            agent_id=agent_id,
            updates=updates
        )
        
        if agent:
            return json.dumps({'ok': True, 'agent': agent})
        return json.dumps({'ok': False, 'error': 'agent_not_found'})
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


@mcp.tool()
async def remove_agent(agent_id: str, delete_memories: bool = False) -> str:
    """
    Remove a memory agent from the system.
    
    WARNING: This permanently removes the agent. Set delete_memories=True
    to also delete all memories associated with this agent.
    
    Args:
        agent_id: The agent to delete
        delete_memories: Also delete all memories in this agent (default: False)
    
    Returns:
        JSON string with deletion status
    """
    global _current_agent_id
    
    try:
        # Delete from Supabase
        deleted = _agents_service.delete_agent(user_id=_default_user_id, agent_id=agent_id)
        
        # Optionally delete ChromaDB collection
        if delete_memories and agent_id in _memory_systems_cache:
            try:
                # Clear the memory system
                del _memory_systems_cache[agent_id]
            except:
                pass
        
        # Clear current agent if deleted
        if _current_agent_id == agent_id:
            _current_agent_id = None
        
        return json.dumps({
            'ok': True,
            'message': 'agent_removed',
            'agent_id': agent_id,
            'memories_deleted': delete_memories,
            'current_agent': _current_agent_id
        })
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


@mcp.tool()
async def switch_to_agent(agent_id: str) -> str:
    """
    Switch to a different memory agent context.
    
    After switching, subsequent memory operations will use this agent
    by default when agent_id is not explicitly specified.
    
    Args:
        agent_id: The agent to switch to
    
    Returns:
        JSON string confirming the switch
    """
    global _current_agent_id
    
    try:
        # Verify agent exists
        agent = _agents_service.get_agent(user_id=_default_user_id, agent_id=agent_id)
        
        if agent:
            _current_agent_id = agent_id
            _agents_service.set_current_agent(_default_user_id, agent_id)
            
            # Initialize memory system if not already
            _get_or_create_memory_system(agent_id)
            
            return json.dumps({
                'ok': True,
                'message': f'Switched to agent: {agent_id}',
                'current_agent': agent_id,
                'agent': agent
            })
        else:
            return json.dumps({
                'ok': False,
                'error': f"Agent '{agent_id}' not found. Create it first with register_agent()",
                'current_agent': _current_agent_id
            })
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


@mcp.tool()
async def current_agent() -> str:
    """
    Get the current/active memory agent context.
    
    Returns the agent that is currently being used for memory operations.
    
    Returns:
        JSON string with current agent details
    """
    global _current_agent_id
    
    try:
        if _current_agent_id:
            agent = _agents_service.get_agent(user_id=_default_user_id, agent_id=_current_agent_id)
            return json.dumps({
                'ok': True,
                'current_agent': _current_agent_id,
                'agent': agent
            })
        
        # Try to get from server
        agent = _agents_service.get_current_agent(user_id=_default_user_id)
        if agent:
            _current_agent_id = agent.get('agent_id')
            return json.dumps({
                'ok': True,
                'current_agent': _current_agent_id,
                'agent': agent
            })
        
        return json.dumps({
            'ok': True,
            'current_agent': None,
            'message': 'No current agent set. Use switch_to_agent() or register_agent() first.'
        })
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


# ============================================================================
# MCP TOOLS - Memory CRUD Operations
# ============================================================================

@mcp.tool()
async def create_memory(agent_id: str, clear_db: bool = False) -> str:
    """
    Create/initialize a SimpleMem memory system for an agent.
    
    This creates a ChromaDB collection for storing memory entries.
    Set clear_db to True to clear existing memories.
    
    Args:
        agent_id: Unique identifier for the agent
        clear_db: Whether to clear existing memories (default: False)
    
    Returns:
        JSON string with creation status
    """
    try:
        memory_system = _get_or_create_memory_system(agent_id, clear_db=clear_db)
        return json.dumps({
            'ok': True,
            'message': 'memory_system_created' if clear_db else 'memory_system_initialized',
            'agent_id': agent_id,
            'cleared': clear_db
        })
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


@mcp.tool()
async def process_raw_dialogues(
    agent_id: str,
    dialogues: List[Dict[str, str]]
) -> str:
    """
    Process raw dialogues through LLM to extract structured memory entries.
    
    Flow: ADD_DIALOGUE → LLM → JSON RESPONSE → N Memory units → Vector Store.
    Each dialogue is processed to extract facts, entities, timestamps, and keywords.
    
    Args:
        agent_id: Unique identifier for the agent
        dialogues: List of dialogue objects, each with keys:
                   - speaker: Name of the speaker
                   - content: The dialogue content
                   - timestamp: (optional) ISO8601 timestamp
    
    Returns:
        JSON string with processing status and count of dialogues processed
    """
    try:
        if not dialogues:
            return json.dumps({'ok': False, 'error': 'dialogues list is required'})
        
        memory_system = _get_or_create_memory_system(agent_id)
        
        memories_created = 0
        for dlg in dialogues:
            speaker = dlg.get('speaker', 'unknown')
            content = dlg.get('content', '')
            timestamp = dlg.get('timestamp')
            
            if content:
                memory_system.add_dialogue(
                    speaker=speaker,
                    content=content,
                    timestamp=timestamp
                )
                memories_created += 1
        
        memory_system.finalize()
        
        return json.dumps({
            'ok': True,
            'message': 'dialogues_processed',
            'agent_id': agent_id,
            'dialogues_processed': memories_created
        })
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


@mcp.tool()
async def add_memory_direct(
    agent_id: str,
    memories: List[Dict[str, Any]]
) -> str:
    """
    Directly save pre-structured memory entries without LLM processing.
    
    Use this when you already have structured memory data and want to bypass
    the LLM extraction step.
    
    Args:
        agent_id: Unique identifier for the agent
        memories: List of memory objects, each with keys:
                  - lossless_restatement: (required) Self-contained fact statement
                  - keywords: (optional) List of keywords
                  - timestamp: (optional) ISO8601 timestamp
                  - location: (optional) Location string
                  - persons: (optional) List of person names
                  - entities: (optional) List of entities
                  - topic: (optional) Topic phrase
    
    Returns:
        JSON string with entry IDs of added memories
    """
    try:
        if not memories:
            return json.dumps({'ok': False, 'error': 'memories list is required'})
        
        memory_system = _get_or_create_memory_system(agent_id)
        
        entries = []
        entry_ids = []
        for mem in memories:
            if not mem.get('lossless_restatement'):
                continue
            
            entry = MemoryEntry(
                lossless_restatement=mem.get('lossless_restatement'),
                keywords=mem.get('keywords', []),
                timestamp=mem.get('timestamp'),
                location=mem.get('location'),
                persons=mem.get('persons', []),
                entities=mem.get('entities', []),
                topic=mem.get('topic')
            )
            entries.append(entry)
            entry_ids.append(entry.entry_id)
        
        if entries:
            memory_system.vector_store.add_entries(entries)
        
        return json.dumps({
            'ok': True,
            'message': 'memories_added',
            'agent_id': agent_id,
            'entries_added': len(entries),
            'entry_ids': entry_ids
        })
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


@mcp.tool()
async def search_memory(
    agent_id: str,
    query: str,
    top_k: int = 5,
    enable_reflection: bool = False
) -> str:
    """
    Search memories using hybrid retrieval (semantic + keyword + structured search).
    
    Uses HybridRetriever to find relevant memory entries combining:
    - Semantic vector similarity
    - Keyword/BM25-style matching
    - Structured metadata filtering
    
    Args:
        agent_id: Unique identifier for the agent
        query: Search query text
        top_k: Number of results to return (default: 5)
        enable_reflection: Enable reflection-based additional retrieval (default: False)
    
    Returns:
        JSON string with search results including memory entries
    """
    try:
        memory_system = _get_or_create_memory_system(agent_id)
        
        contexts = memory_system.hybrid_retriever.retrieve(query, enable_reflection=enable_reflection)
        
        results = []
        for ctx in contexts[:top_k]:
            results.append({
                'entry_id': ctx.entry_id,
                'lossless_restatement': ctx.lossless_restatement,
                'keywords': ctx.keywords,
                'timestamp': ctx.timestamp,
                'location': ctx.location,
                'persons': ctx.persons,
                'entities': ctx.entities,
                'topic': ctx.topic
            })
        
        return json.dumps({
            'ok': True,
            'agent_id': agent_id,
            'query': query,
            'results_count': len(results),
            'results': results
        })
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


@mcp.tool()
async def get_context_answer(
    agent_id: str,
    question: str
) -> str:
    """
    Get a context-aware answer using SimpleMem's ask function.
    
    Full Q&A flow: Query → HybridRetrieval → AnswerGenerator → Response.
    Returns both the LLM-generated answer and the memory contexts used.
    
    Args:
        agent_id: Unique identifier for the agent
        question: The question to answer using memory context
    
    Returns:
        JSON string with the answer and contexts used
    """
    try:
        memory_system = _get_or_create_memory_system(agent_id)
        
        answer = memory_system.ask(question)
        
        contexts = memory_system.hybrid_retriever.retrieve(question)
        contexts_used = [
            {
                'entry_id': ctx.entry_id,
                'lossless_restatement': ctx.lossless_restatement,
                'topic': ctx.topic
            }
            for ctx in contexts[:5]
        ]
        
        return json.dumps({
            'ok': True,
            'agent_id': agent_id,
            'question': question,
            'answer': answer,
            'contexts_used': contexts_used
        })
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


@mcp.tool()
async def update_memory_entry(
    agent_id: str,
    entry_id: str,
    updates: Dict[str, Any]
) -> str:
    """
    Update an existing memory entry in ChromaDB.
    
    You can update the document content (lossless_restatement) and/or
    metadata fields (timestamp, location, persons, entities, topic, keywords).
    
    Args:
        agent_id: Unique identifier for the agent
        entry_id: The ID of the memory entry to update
        updates: Dictionary of fields to update. Keys can be:
                 - lossless_restatement: New document content
                 - timestamp: New timestamp
                 - location: New location
                 - persons: New list of persons
                 - entities: New list of entities
                 - topic: New topic
                 - keywords: New list of keywords
    
    Returns:
        JSON string with update status
    """
    try:
        if not updates:
            return json.dumps({'ok': False, 'error': 'updates dict is required'})
        
        memory_system = _get_or_create_memory_system(agent_id)
        
        document_content = updates.get('lossless_restatement')
        
        metadata = {}
        updateable_metadata = ['timestamp', 'location', 'persons', 'entities', 'topic', 'keywords']
        for field in updateable_metadata:
            if field in updates:
                value = updates[field]
                if isinstance(value, list):
                    metadata[field] = json.dumps(value)
                else:
                    metadata[field] = value
        
        if document_content:
            memory_system.vector_store.rag.update_docs(
                agent_ID=agent_id,
                ids=[entry_id],
                documents=[document_content],
                metadatas=[metadata] if metadata else None
            )
        elif metadata:
            memory_system.vector_store.rag.update_doc_metadata(
                agent_ID=agent_id,
                ids=[entry_id],
                metadatas=[metadata]
            )
        
        return json.dumps({
            'ok': True,
            'message': 'memory_updated',
            'agent_id': agent_id,
            'entry_id': entry_id
        })
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


@mcp.tool()
async def delete_memory_entries(
    agent_id: str,
    entry_ids: List[str]
) -> str:
    """
    Delete memory entries from ChromaDB by their entry IDs.
    
    This permanently removes the memory entries from the agent's vector store.
    
    Args:
        agent_id: Unique identifier for the agent
        entry_ids: List of entry IDs to delete
    
    Returns:
        JSON string with deletion status
    """
    try:
        if not entry_ids:
            return json.dumps({'ok': False, 'error': 'entry_ids list is required'})
        
        memory_system = _get_or_create_memory_system(agent_id)
        
        memory_system.vector_store.rag.delete_chat_history(
            agent_ID=agent_id,
            ids=entry_ids
        )
        
        return json.dumps({
            'ok': True,
            'message': 'memories_deleted',
            'agent_id': agent_id,
            'deleted_count': len(entry_ids),
            'entry_ids': entry_ids
        })
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


@mcp.tool()
async def list_all_memories(agent_id: str, limit: int = 50) -> str:
    """
    List all memory entries for an agent.
    
    Args:
        agent_id: Unique identifier for the agent
        limit: Maximum number of entries to return (default: 50)
    
    Returns:
        JSON string with list of all memory entries
    """
    try:
        memory_system = _get_or_create_memory_system(agent_id)
        
        memories = memory_system.get_all_memories()
        
        results = []
        for mem in memories[:limit]:
            results.append({
                'entry_id': mem.entry_id,
                'lossless_restatement': mem.lossless_restatement,
                'keywords': mem.keywords,
                'timestamp': mem.timestamp,
                'location': mem.location,
                'persons': mem.persons,
                'entities': mem.entities,
                'topic': mem.topic
            })
        
        return json.dumps({
            'ok': True,
            'agent_id': agent_id,
            'total_memories': len(memories),
            'returned': len(results),
            'memories': results
        })
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})


# ============================================================================
# MCP RESOURCES - Expose data sources for Claude to read
# ============================================================================

@mcp.resource("memory://agents/list")
async def list_active_agents() -> str:
    """List all agents with active memory systems."""
    return json.dumps({
        'active_agents': list(_memory_systems_cache.keys()),
        'count': len(_memory_systems_cache)
    })


@mcp.resource("memory://config/info")
async def get_server_info() -> str:
    """Get information about the MCP Memory Server."""
    return json.dumps({
        'name': 'Manhattan Memory MCP Server',
        'version': '2.0.0',
        'description': 'MCP server for Memory CRUD operations with agent management',
        'current_agent': _current_agent_id,
        'available_tools': {
            'agent_management': [
                'register_agent',
                'list_my_agents',
                'get_agent_details',
                'update_agent_info',
                'remove_agent',
                'switch_to_agent',
                'current_agent'
            ],
            'memory_operations': [
                'create_memory',
                'process_raw_dialogues',
                'add_memory_direct',
                'search_memory',
                'get_context_answer',
                'update_memory_entry',
                'delete_memory_entries',
                'list_all_memories'
            ]
        }
    })


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Initialize and run the MCP server."""
    print("=" * 60, file=sys.stderr)
    print("  Manhattan Memory MCP Server v2.0", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(file=sys.stderr)
    print("Agent Management Tools:", file=sys.stderr)
    print("  * register_agent       - Create a new memory agent", file=sys.stderr)
    print("  * list_my_agents       - List all your agents", file=sys.stderr)
    print("  * get_agent_details    - Get agent info", file=sys.stderr)
    print("  * update_agent_info    - Update agent details", file=sys.stderr)
    print("  * remove_agent         - Delete an agent", file=sys.stderr)
    print("  * switch_to_agent      - Switch context to an agent", file=sys.stderr)
    print("  * current_agent        - Get current agent", file=sys.stderr)
    print(file=sys.stderr)
    print("Memory Operations:", file=sys.stderr)
    print("  * create_memory        - Initialize memory system", file=sys.stderr)
    print("  * process_raw_dialogues - Process dialogues via LLM", file=sys.stderr)
    print("  * add_memory_direct    - Add memories directly", file=sys.stderr)
    print("  * search_memory        - Hybrid search", file=sys.stderr)
    print("  * get_context_answer   - Q&A with memory context", file=sys.stderr)
    print("  * update_memory_entry  - Update memory", file=sys.stderr)
    print("  * delete_memory_entries - Delete memories", file=sys.stderr)
    print("  * list_all_memories    - List all memories", file=sys.stderr)
    print(file=sys.stderr)
    import argparse
    
    parser = argparse.ArgumentParser(description='Manhattan Memory MCP Server')
    parser.add_argument('--transport', default='stdio', choices=['stdio', 'sse'],
                      help='Transport protocol to use (default: stdio)')
    parser.add_argument('--host', default='0.0.0.0',
                      help='Host to bind to for SSE (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to listen on for SSE (default: 8000)')
    
    args = parser.parse_args()
    
    if args.transport == 'sse':
        print(f"Starting Manhattan Memory MCP Server on http://{args.host}:{args.port} (SSE)", file=sys.stderr)
        print("Local resource access enabled.", file=sys.stderr)
        mcp.settings.port = args.port
        mcp.settings.host = args.host
        mcp.run(transport="sse")
    else:
        # Check standard input/output for stdio
        print("Running on stdio transport...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        mcp.run(transport="stdio")


if __name__ == "__main__":
    # Ensure all required environment variables are set or warn
    if not os.getenv("MANHATTAN_API_KEY") and not os.getenv("SUPABASE_URL"):
        print("Warning: MANHATTAN_API_KEY or SUPABASE credentials not found in environment.", file=sys.stderr)
        print("Agent management and some features may be limited.", file=sys.stderr)
        
    main()
