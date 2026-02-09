"""
GitMem Local - Local API

Local Python API for AI context storage.
No web servers or external APIs required.

Usage:
    from gitmem.api import LocalAPI
    
    api = LocalAPI()
    
    # Start session
    api.session_start("my-agent")
    
    # Add memory
    api.add_memory("my-agent", [{"lossless_restatement": "User likes Python"}])
    
    # Search
    results = api.search_memory("my-agent", "Python")
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from .context_manager import ContextManager, get_context_manager
from .memory_store import LocalMemoryStore
from .file_system import LocalFileSystem, FolderType
from .object_store import MemoryDAG


class LocalAPI:
    """
    Local API for AI context storage.
    
    All operations are local - no network calls.
    Data is stored in JSON files with Git-like version control.
    """
    
    def __init__(self, root_path: str = "./.gitmem_data"):
        """
        Initialize the Local API.
        
        Args:
            root_path: Root directory for data storage
        """
        self.ctx = ContextManager(root_path)
        self.store = self.ctx.store
        self.fs = self.ctx.fs
        self.dag = self.store.dag
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def session_start(self, agent_id: str, auto_pull_context: bool = True) -> Dict:
        """
        Start a new session for an agent.
        
        Args:
            agent_id: Unique agent identifier
            auto_pull_context: Load recent memories automatically
        
        Returns:
            Session info with loaded context
        """
        return self.ctx.session_start(agent_id, auto_pull_context)
    
    def session_end(self, agent_id: str, conversation_summary: str = None,
                    key_points: List[str] = None) -> Dict:
        """
        End a session and optionally save a checkpoint.
        
        Args:
            agent_id: Agent identifier
            conversation_summary: Summary of the conversation
            key_points: Key decisions/facts from this session
        
        Returns:
            Session end status
        """
        return self.ctx.session_end(agent_id, conversation_summary, key_points)
    
    def checkpoint(self, agent_id: str, summary: str, key_points: List[str] = None) -> Dict:
        """
        Save a conversation checkpoint.
        
        Args:
            agent_id: Agent identifier
            summary: Conversation summary
            key_points: Key points to remember
        
        Returns:
            Checkpoint ID
        """
        return self.ctx.conversation_checkpoint(agent_id, summary, key_points)
    
    # =========================================================================
    # Memory CRUD
    # =========================================================================
    
    def add_memory(self, agent_id: str, memories: List[Dict]) -> Dict:
        """
        Add one or more memories.
        
        Args:
            agent_id: Agent identifier
            memories: List of memory objects:
                - lossless_restatement: Clear statement of fact
                - keywords: Searchable keywords (optional)
                - persons: People mentioned (optional)
                - topic: Category (optional)
                - memory_type: episodic/semantic/procedural (default: semantic)
        
        Returns:
            Created entry IDs
        
        Example:
            api.add_memory("agent-1", [{
                "lossless_restatement": "User prefers dark mode",
                "keywords": ["preference", "dark mode", "UI"],
                "topic": "preferences"
            }])
        """
        return self.ctx.add_memory(agent_id, memories)
    
    def search_memory(self, agent_id: str, query: str, top_k: int = 5) -> Dict:
        """
        Search memories by query.
        
        Args:
            agent_id: Agent identifier
            query: Natural language search query
            top_k: Maximum results to return
        
        Returns:
            Matching memories with scores
        """
        return self.ctx.search_memory(agent_id, query, top_k)
    
    def get_memory(self, agent_id: str, memory_id: str) -> Optional[Dict]:
        """
        Get a specific memory by ID.
        
        Args:
            agent_id: Agent identifier
            memory_id: Memory ID
        
        Returns:
            Memory object or None
        """
        return self.store.get_memory_by_id(agent_id, memory_id)
    
    def update_memory(self, agent_id: str, memory_id: str, updates: Dict) -> bool:
        """
        Update a memory entry.
        
        Args:
            agent_id: Agent identifier
            memory_id: Memory ID to update
            updates: Fields to update
        
        Returns:
            True if updated successfully
        """
        return self.store.update_memory(agent_id, memory_id, updates)
    
    def delete_memory(self, agent_id: str, memory_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            agent_id: Agent identifier
            memory_id: Memory ID to delete
        
        Returns:
            True if deleted successfully
        """
        return self.store.delete_memory(agent_id, memory_id)
    
    def list_memories(self, agent_id: str, memory_type: str = None,
                      limit: int = 50, offset: int = 0,
                      filter_topic: str = None, filter_person: str = None) -> Dict:
        """
        List memories with optional filtering.
        
        Args:
            agent_id: Agent identifier
            memory_type: Filter by type (episodic/semantic/procedural/working) - Note: currently handled by partial filtering in context or ignored if not supported by ctx
            limit: Maximum results
            offset: Pagination offset
            filter_topic: Filter by topic
            filter_person: Filter by person
        
        Returns:
            Paginated memory list
        """
        # Context manager list_memories signature: (agent_id, limit, offset, filter_topic, filter_person)
        # It doesn't support memory_type directly in list yet, but let's pass what we can
        return self.ctx.list_memories(agent_id, limit, offset, filter_topic, filter_person)
    
    def get_context_answer(self, agent_id: str, question: str) -> Dict:
        """
        Get context-aware answer using stored memories.
        
        Args:
            agent_id: Agent identifier
            question: Natural language question
        
        Returns:
            Context answer with source memories
        """
        return self.ctx.get_context_answer(agent_id, question)
    
    # =========================================================================
    # Vector Search
    # =========================================================================
    
    def hybrid_search(self, agent_id: str, query: str, top_k: int = 5) -> Dict:
        """
        Search memories using hybrid (semantic + keyword) search.
        
        Combines vector similarity with keyword matching for best results.
        Requires vector embeddings to be enabled.
        
        Args:
            agent_id: Agent identifier
            query: Natural language search query
            top_k: Maximum results to return
        
        Returns:
            Matching memories with hybrid scores
        """
        try:
            results = self.store.hybrid_search_memory(agent_id, query, top_k)
            return {
                "results": results,
                "count": len(results),
                "search_type": "hybrid"
            }
        except Exception as e:
            # Fallback to keyword search
            return self.search_memory(agent_id, query, top_k)
    
    def semantic_search(self, agent_id: str, query: str, top_k: int = 5) -> Dict:
        """
        Search memories using pure semantic (vector) search.
        
        Uses embedding similarity to find relevant memories.
        
        Args:
            agent_id: Agent identifier
            query: Natural language search query
            top_k: Maximum results to return
        
        Returns:
            Matching memories with semantic scores
        """
        try:
            results = self.store.semantic_search_memory(agent_id, query, top_k)
            return {
                "results": results,
                "count": len(results),
                "search_type": "semantic"
            }
        except Exception as e:
            # Fallback to keyword search
            return self.search_memory(agent_id, query, top_k)
    
    def get_vector_stats(self, agent_id: str) -> Dict:
        """
        Get vector storage statistics for an agent.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Vector statistics including count, dimension, and cache info
        """
        try:
            return self.store.get_vector_stats(agent_id)
        except Exception as e:
            return {
                "agent_id": agent_id,
                "vectors_enabled": False,
                "error": str(e)
            }
    
    def enable_vectors(self, enable: bool = True):
        """
        Enable or disable vector search capabilities.
        
        Args:
            enable: Whether to enable vectors
        """
        self.store.enable_vectors = enable
    
    # =========================================================================
    # Auto-Remember
    # =========================================================================
    
    def auto_remember(self, agent_id: str, user_message: str) -> Dict:
        """
        Automatically extract and store facts from a message.
        
        Args:
            agent_id: Agent identifier
            user_message: User's message to analyze
        
        Returns:
            What was extracted and stored
        """
        return self.ctx.auto_remember(agent_id, user_message)
    
    def should_remember(self, message: str) -> Dict:
        """
        Check if a message contains information worth storing.
        
        Args:
            message: Message to analyze
        
        Returns:
            Recommendations on what to remember
        """
        return self.ctx.should_remember(message)
    
    # =========================================================================
    # File System
    # =========================================================================
    
    def list_dir(self, agent_id: str, path: str = "") -> Dict:
        """
        List contents of a virtual directory.
        
        Args:
            agent_id: Agent identifier
            path: Virtual path (e.g., "context/episodic")
        
        Returns:
            Directory listing with files and folders
        """
        return self.ctx.list_dir(agent_id, path)
    
    def read_file(self, agent_id: str, path: str) -> Dict:
        """
        Read content of a virtual file.
        
        Args:
            agent_id: Agent identifier
            path: Virtual file path
        
        Returns:
            File content and metadata
        """
        return self.ctx.read_file(agent_id, path)
    
    def write_file(self, agent_id: str, path: str, content: str,
                   metadata: Dict = None) -> Optional[str]:
        """
        Write content to a virtual file.
        
        Args:
            agent_id: Agent identifier
            path: Virtual file path
            content: Content to write
            metadata: Optional metadata
        
        Returns:
            Created item ID
        """
        return self.fs.write_file(agent_id, path, content, metadata)
    
    def delete_file(self, agent_id: str, path: str) -> bool:
        """
        Delete a virtual file.
        
        Args:
            agent_id: Agent identifier
            path: Virtual file path
        
        Returns:
            True if deleted
        """
        return self.fs.delete_file(agent_id, path)
    
    # =========================================================================
    # Documents
    # =========================================================================
    
    def add_document(self, agent_id: str, filename: str, content: str,
                     folder: str = "knowledge", **kwargs) -> str:
        """
        Add a document.
        
        Args:
            agent_id: Agent identifier
            filename: Document filename
            content: Document content
            folder: Folder (knowledge/references)
            **kwargs: Additional metadata
        
        Returns:
            Document ID
        """
        return self.store.add_document(agent_id, filename, content, folder, **kwargs)
    
    def get_document(self, agent_id: str, doc_id: str) -> Optional[Dict]:
        """Get a document by ID."""
        return self.store.get_document_by_id(agent_id, doc_id)
    
    def list_documents(self, agent_id: str, folder: str = None, limit: int = 50) -> List[Dict]:
        """List documents."""
        return self.store.list_documents(agent_id, folder, limit)
    
    def delete_document(self, agent_id: str, doc_id: str) -> bool:
        """Delete a document."""
        return self.store.delete_document(agent_id, doc_id)
    
    # =========================================================================
    # Version Control
    # =========================================================================
    
    def commit(self, agent_id: str, message: str) -> Optional[str]:
        """
        Commit current state with a message.
        
        Args:
            agent_id: Agent identifier
            message: Commit message
        
        Returns:
            Commit hash
        """
        return self.store.commit_state(agent_id, message)
    
    def history(self, agent_id: str, limit: int = 10) -> List[Dict]:
        """
        Get commit history.
        
        Args:
            agent_id: Agent identifier
            limit: Maximum commits to return
        
        Returns:
            List of commits
        """
        return self.store.get_history(agent_id, limit)
    
    def rollback(self, agent_id: str, commit_sha: str) -> Dict:
        """
        Rollback to a previous commit.
        
        Args:
            agent_id: Agent identifier
            commit_sha: Commit hash to rollback to
        
        Returns:
            Rollback status
        """
        return self.store.rollback(agent_id, commit_sha)
    
    def diff(self, agent_id: str, sha_a: str, sha_b: str) -> Dict:
        """
        Compare two commits.
        
        Args:
            agent_id: Agent identifier
            sha_a: First commit
            sha_b: Second commit
        
        Returns:
            Diff showing added/removed memories
        """
        self.dag.set_agent(agent_id)
        return self.dag.diff(sha_a, sha_b)
    
    def branch(self, agent_id: str, name: str) -> str:
        """
        Create a new branch.
        
        Args:
            agent_id: Agent identifier
            name: Branch name
        
        Returns:
            Branch origin commit
        """
        self.dag.set_agent(agent_id)
        return self.dag.branch(name)
    
    def list_branches(self) -> Dict[str, str]:
        """List all branches."""
        return self.dag.list_branches()
    
    def tag(self, agent_id: str, name: str) -> str:
        """
        Create a tag for current state.
        
        Args:
            agent_id: Agent identifier
            name: Tag name
        
        Returns:
            Tagged commit
        """
        self.dag.set_agent(agent_id)
        return self.dag.tag(name)
    
    def list_tags(self) -> Dict[str, str]:
        """List all tags."""
        return self.dag.list_tags()
    
    # =========================================================================
    # Agent Management
    # =========================================================================
    
    def list_agents(self) -> List[str]:
        """List all agents."""
        return self.store.list_agents()
    
    def get_agent_stats(self, agent_id: str) -> Dict:
        """Get comprehensive statistics for an agent."""
        return self.ctx.get_agent_stats(agent_id)
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent and all its data."""
        return self.store.delete_agent(agent_id)
    
    # =========================================================================
    # Export/Import
    # =========================================================================
    
    def export_memories(self, agent_id: str) -> Dict:
        """
        Export all memories for backup.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Complete backup data
        """
        return self.store.export_memories(agent_id)
    
    def import_memories(self, agent_id: str, export_data: Dict,
                        merge_mode: str = "append") -> Dict:
        """
        Import memories from backup.
        
        Args:
            agent_id: Agent identifier
            export_data: Data from export_memories
            merge_mode: "append" or "replace"
        
        Returns:
            Import status
        """
        return self.store.import_memories(agent_id, export_data, merge_mode)
    
    # =========================================================================
    # Utility
    # =========================================================================
    
    def memory_summary(self, agent_id: str, focus_topic: str = None) -> Dict:
        """Get a summary of stored memories."""
        return self.ctx.memory_summary(agent_id, focus_topic)
    
    def what_do_i_know(self, agent_id: str) -> Dict:
        """Get a summary of known information about the user."""
        return self.ctx.what_do_i_know(agent_id)
    
    def pre_response_check(self, user_message: str, intended_response: str) -> Dict:
        """Check before responding to avoid mistakes."""
        return self.ctx.pre_response_check(user_message, intended_response)


# Convenience function to get a LocalAPI instance
def get_api(root_path: str = "./.gitmem_data") -> LocalAPI:
    """Get a LocalAPI instance."""
    return LocalAPI(root_path)


# Quick access functions
_default_api: Optional[LocalAPI] = None


def init(root_path: str = "./.gitmem_data") -> LocalAPI:
    """Initialize the default API."""
    global _default_api
    _default_api = LocalAPI(root_path)
    return _default_api


def api() -> LocalAPI:
    """Get the default API instance."""
    global _default_api
    if _default_api is None:
        _default_api = LocalAPI()
    return _default_api
