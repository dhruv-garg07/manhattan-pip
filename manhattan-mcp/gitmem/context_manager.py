"""
GitMem Local - Context Manager

High-level interface for AI agents to manage their context.
Provides smart context retrieval, auto-remembering, and session management.
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .memory_store import LocalMemoryStore
from .file_system import LocalFileSystem


@dataclass
class SessionContext:
    """Current session state."""
    agent_id: str
    session_id: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())
    working_memory: List[Dict] = field(default_factory=list)
    recent_queries: List[str] = field(default_factory=list)
    conversation_summary: str = ""
    key_points: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "last_activity": self.last_activity,
            "working_memory_count": len(self.working_memory),
            "recent_queries": self.recent_queries[-5:],
            "conversation_summary": self.conversation_summary,
            "key_points": self.key_points
        }


class ContextManager:
    """
    High-level AI context management.
    
    Provides:
    - Session management (start, end, checkpoints)
    - Smart context retrieval
    - Auto-remembering from messages
    - Memory hints and suggestions
    - Pre-response checks
    """
    
    def __init__(self, root_path: str = "./.gitmem_data"):
        self.store = LocalMemoryStore(root_path)
        self.fs = LocalFileSystem(self.store)
        self._sessions: Dict[str, SessionContext] = {}
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def session_start(self, agent_id: str, auto_pull_context: bool = True) -> Dict:
        """
        Start a new session for an agent.
        
        Args:
            agent_id: The agent ID
            auto_pull_context: Auto-load relevant memories
        
        Returns:
            Session info and loaded context
        """
        import uuid
        session_id = str(uuid.uuid4())
        
        session = SessionContext(
            agent_id=agent_id,
            session_id=session_id
        )
        self._sessions[agent_id] = session
        
        # Create agent if not exists
        self.store._ensure_agent(agent_id)
        
        # Auto-pull context
        context = []
        if auto_pull_context:
            # Get recent memories
            recent = self.store.list_memories(agent_id, limit=10)
            context = recent[:5]
            session.working_memory = context
        
        # Log session start
        self.store._log_activity(
            agent_id, "access", "session_start", "session", session_id, "system", "system"
        )
        
        return {
            "status": "OK",
            "session_id": session_id,
            "agent_id": agent_id,
            "context_loaded": len(context),
            "message": f"Session started for agent {agent_id}"
        }
    
    def session_end(self, agent_id: str, conversation_summary: str = None,
                    key_points: List[str] = None) -> Dict:
        """
        End a session and save state.
        
        Args:
            agent_id: The agent ID
            conversation_summary: Summary of the conversation
            key_points: Key facts/decisions from this session
        
        Returns:
            Session end status
        """
        session = self._sessions.get(agent_id)
        
        # Create checkpoint with session summary
        if session and (conversation_summary or key_points):
            checkpoint_name = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.store.create_checkpoint(
                agent_id=agent_id,
                name=checkpoint_name,
                checkpoint_type="session",
                description=conversation_summary or "",
                metadata={
                    "key_points": key_points or [],
                    "session_id": session.session_id if session else None
                }
            )
        
        # Remove session
        if agent_id in self._sessions:
            del self._sessions[agent_id]
        
        return {
            "status": "OK",
            "message": f"Session ended for agent {agent_id}",
            "checkpoint_created": bool(conversation_summary or key_points)
        }
    
    def conversation_checkpoint(self, agent_id: str, conversation_summary: str,
                                 key_points: List[str] = None) -> Dict:
        """
        Save a conversation checkpoint.
        
        Args:
            agent_id: The agent ID
            conversation_summary: Summary so far
            key_points: Key decisions/facts
        
        Returns:
            Checkpoint status
        """
        checkpoint_id = self.store.create_checkpoint(
            agent_id=agent_id,
            name=f"Checkpoint_{datetime.now().strftime('%H%M%S')}",
            checkpoint_type="session",
            description=conversation_summary,
            metadata={"key_points": key_points or []}
        )
        
        return {
            "status": "OK",
            "checkpoint_id": checkpoint_id
        }
    
    # =========================================================================
    # Memory Operations
    # =========================================================================
    
    def add_memory(self, agent_id: str, memories: List[Dict]) -> Dict:
        """
        Add one or more memories.
        
        Args:
            agent_id: The agent ID
            memories: List of memory objects with:
                - lossless_restatement: Clear restatement
                - keywords: Searchable keywords
                - persons: People mentioned
                - topic: Category
        
        Returns:
            Entry IDs
        """
        entry_ids = []
        
        for mem in memories:
            entry_id = self.store.add_memory(
                agent_id=agent_id,
                content=mem.get("lossless_restatement", ""),
                lossless_restatement=mem.get("lossless_restatement", ""),
                memory_type=mem.get("memory_type", "semantic"),
                keywords=mem.get("keywords", []),
                persons=mem.get("persons", []),
                entities=mem.get("entities", []),
                topic=mem.get("topic", ""),
                importance=mem.get("importance", 0.5),
                metadata=mem.get("metadata", {})
            )
            entry_ids.append(entry_id)
        
        return {
            "status": "OK",
            "entry_ids": entry_ids,
            "count": len(entry_ids)
        }
    
    def search_memory(self, agent_id: str, query: str, top_k: int = 5,
                      enable_reflection: bool = False) -> Dict:
        """
        Search memories.
        
        Args:
            agent_id: The agent ID
            query: Search query
            top_k: Number of results
            enable_reflection: Enable multi-round retrieval (future)
        
        Returns:
            Search results
        """
        results = self.store.search_memory(agent_id, query, top_k)
        
        # Track query in session
        session = self._sessions.get(agent_id)
        if session:
            session.recent_queries.append(query)
            session.recent_queries = session.recent_queries[-10:]
            session.last_activity = datetime.now().isoformat()
        
        return {
            "status": "OK",
            "results": results,
            "count": len(results),
            "query": query
        }
    
    def get_context_answer(self, agent_id: str, question: str) -> Dict:
        """
        Get a context-aware answer using stored memories.
        
        Args:
            agent_id: The agent ID
            question: Natural language question
        
        Returns:
            Context answer with sources
        """
        # Search for relevant memories
        results = self.store.search_memory(agent_id, question, top_k=10)
        
        # Build context
        context_parts = []
        for mem in results:
            content = mem.get("lossless_restatement") or mem.get("content", "")
            if content:
                context_parts.append(content)
        
        return {
            "status": "OK",
            "context": context_parts,
            "sources": [{"id": m["id"], "content": m.get("content", "")[:100]} for m in results],
            "source_count": len(results)
        }
    
    def list_memories(self, agent_id: str, limit: int = 50, offset: int = 0,
                      filter_topic: str = None, filter_person: str = None) -> Dict:
        """
        List memories with filtering.
        
        Args:
            agent_id: The agent ID
            limit: Max results
            offset: Pagination offset
            filter_topic: Filter by topic
            filter_person: Filter by person
        
        Returns:
            Paginated memory list
        """
        memories = self.store.list_memories(agent_id, limit=limit * 2, offset=0)
        
        # Apply filters
        if filter_topic:
            memories = [m for m in memories if filter_topic.lower() in m.get("topic", "").lower()]
        if filter_person:
            memories = [m for m in memories if filter_person.lower() in str(m.get("persons", [])).lower()]
        
        total = len(memories)
        memories = memories[offset:offset + limit]
        
        return {
            "status": "OK",
            "memories": memories,
            "total": total,
            "offset": offset,
            "limit": limit
        }
    
    def memory_summary(self, agent_id: str, focus_topic: str = None,
                       summary_length: str = "medium") -> Dict:
        """
        Generate a summary of stored memories.
        
        Args:
            agent_id: The agent ID
            focus_topic: Focus on specific topic
            summary_length: brief, medium, or detailed
        
        Returns:
            Summary of memories
        """
        memories = self.store.list_memories(agent_id, limit=100)
        
        if focus_topic:
            memories = [m for m in memories if focus_topic.lower() in str(m).lower()]
        
        # Group by topic
        topics = {}
        for mem in memories:
            topic = mem.get("topic", "general")
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(mem.get("lossless_restatement") or mem.get("content", ""))
        
        # Build summary
        summary_parts = []
        for topic, contents in topics.items():
            summary_parts.append(f"**{topic}**: {len(contents)} memories")
            if summary_length in ("medium", "detailed"):
                for content in contents[:3]:
                    summary_parts.append(f"  - {content[:100]}...")
        
        return {
            "status": "OK",
            "summary": "\n".join(summary_parts),
            "topics": list(topics.keys()),
            "total_memories": len(memories)
        }
    
    # =========================================================================
    # Auto-Remember
    # =========================================================================
    
    def auto_remember(self, agent_id: str, user_message: str) -> Dict:
        """
        Automatically extract and store important facts from a message.
        
        Args:
            agent_id: The agent ID
            user_message: The user's message
        
        Returns:
            What was remembered
        """
        # Simple extraction - look for patterns
        facts = []
        message_lower = user_message.lower()
        
        # Name patterns
        if "my name is" in message_lower or "i'm " in message_lower or "i am " in message_lower:
            facts.append({
                "lossless_restatement": user_message,
                "topic": "personal",
                "keywords": ["name", "identity"],
                "memory_type": "semantic"
            })
        
        # Preference patterns
        if any(word in message_lower for word in ["i prefer", "i like", "i love", "i hate", "i don't like"]):
            facts.append({
                "lossless_restatement": user_message,
                "topic": "preferences",
                "keywords": ["preference", "like", "preference"],
                "memory_type": "semantic"
            })
        
        # Work/project patterns
        if any(word in message_lower for word in ["working on", "my project", "building", "developing"]):
            facts.append({
                "lossless_restatement": user_message,
                "topic": "work",
                "keywords": ["project", "work", "development"],
                "memory_type": "semantic"
            })
        
        # Date/time patterns
        if any(word in message_lower for word in ["birthday", "deadline", "meeting", "appointment"]):
            facts.append({
                "lossless_restatement": user_message,
                "topic": "dates",
                "keywords": ["date", "event", "schedule"],
                "memory_type": "episodic"
            })
        
        # If nothing specific found, still save as episodic if substantial
        if not facts and len(user_message) > 50:
            facts.append({
                "lossless_restatement": user_message,
                "topic": "conversation",
                "keywords": [],
                "memory_type": "episodic"
            })
        
        # Store facts
        result = None
        if facts:
            result = self.add_memory(agent_id, facts)
        
        return {
            "status": "OK",
            "extracted_facts": len(facts),
            "facts": facts,
            "stored": result is not None
        }
    
    def should_remember(self, message: str) -> Dict:
        """
        Analyze if a message contains memorable information.
        
        Args:
            message: Message to analyze
        
        Returns:
            Recommendations on what to remember
        """
        recommendations = []
        message_lower = message.lower()
        
        # Check for valuable patterns
        if any(word in message_lower for word in ["my name", "i am", "i'm"]):
            recommendations.append({"type": "personal_info", "reason": "Contains identity information"})
        
        if any(word in message_lower for word in ["prefer", "like", "love", "hate", "favorite"]):
            recommendations.append({"type": "preference", "reason": "Contains preference information"})
        
        if any(word in message_lower for word in ["remember", "don't forget", "important"]):
            recommendations.append({"type": "explicit_memory", "reason": "User explicitly asked to remember"})
        
        if any(word in message_lower for word in ["work", "project", "building", "developing"]):
            recommendations.append({"type": "work_context", "reason": "Contains work/project information"})
        
        return {
            "should_remember": len(recommendations) > 0,
            "recommendations": recommendations,
            "confidence": min(1.0, len(recommendations) * 0.3 + 0.1)
        }
    
    # =========================================================================
    # Pre-Response Checks
    # =========================================================================
    
    def pre_response_check(self, user_message: str, intended_response_topic: str) -> Dict:
        """
        Check before responding to avoid mistakes.
        
        Args:
            user_message: What the user said
            intended_response_topic: What you plan to respond about
        
        Returns:
            Reminders and relevant context
        """
        reminders = []
        
        # Check if asking about something we might know
        if any(word in user_message.lower() for word in ["what", "do you know", "remember", "tell me"]):
            reminders.append("Check memory before answering - user might be asking about stored info")
        
        # Check for correction patterns
        if any(word in user_message.lower() for word in ["actually", "no,", "wrong", "correct"]):
            reminders.append("User might be correcting previous information - update memory if needed")
        
        return {
            "status": "OK",
            "reminders": reminders,
            "check_memory": True,
            "user_message_length": len(user_message)
        }
    
    def am_i_missing_something(self, about_to_say: str, agent_id: str = None) -> Dict:
        """
        Last-chance check before responding.
        
        Args:
            about_to_say: Summary of planned response
            agent_id: Optional agent ID
        
        Returns:
            Potential issues and warnings
        """
        warnings = []
        
        # Generic checks
        if "?" in about_to_say:
            warnings.append("You're about to ask a question - did you check memory first?")
        
        if agent_id:
            # Check if there's relevant context
            results = self.store.search_memory(agent_id, about_to_say, top_k=3)
            if results:
                warnings.append(f"Found {len(results)} potentially relevant memories - consider referencing them")
        
        return {
            "status": "OK",
            "warnings": warnings,
            "has_issues": len(warnings) > 0
        }
    
    # =========================================================================
    # Stats & Hints
    # =========================================================================
    
    def get_agent_stats(self, agent_id: str) -> Dict:
        """Get comprehensive statistics for an agent."""
        return self.store.get_agent_stats(agent_id)
    
    def get_memory_hints(self, agent_id: str) -> Dict:
        """Get suggestions for memory engagement."""
        stats = self.store.get_agent_stats(agent_id)
        
        suggestions = []
        
        if stats["memories"]["total"] == 0:
            suggestions.append("No memories stored yet - start by remembering user preferences")
        
        if stats["memories"]["semantic"] < stats["memories"]["episodic"]:
            suggestions.append("Consider extracting semantic facts from episodic memories")
        
        if stats["checkpoints"] == 0:
            suggestions.append("Create checkpoints to track conversation milestones")
        
        return {
            "status": "OK",
            "suggestions": suggestions,
            "stats": stats
        }
    
    def what_do_i_know(self, agent_id: str) -> Dict:
        """Get a summary of what's known about the user."""
        memories = self.store.list_memories(agent_id, limit=50)
        
        # Extract people
        all_persons = set()
        all_topics = set()
        for mem in memories:
            all_persons.update(mem.get("persons", []))
            if mem.get("topic"):
                all_topics.add(mem["topic"])
        
        # Recent memories for context
        recent = memories[:5]
        
        return {
            "status": "OK",
            "total_memories": len(memories),
            "known_persons": list(all_persons),
            "topics": list(all_topics),
            "recent_context": [m.get("lossless_restatement", m.get("content", ""))[:100] for m in recent]
        }
    
    # =========================================================================
    # File System Access
    # =========================================================================
    
    def list_dir(self, agent_id: str, path: str = "") -> Dict:
        """List virtual directory contents."""
        nodes = self.fs.list_dir(agent_id, path)
        return {
            "status": "OK",
            "path": path,
            "nodes": nodes,
            "count": len(nodes)
        }
    
    def read_file(self, agent_id: str, path: str) -> Dict:
        """Read a virtual file."""
        content = self.fs.read_file(agent_id, path)
        if content:
            return {
                "status": "OK",
                "path": path,
                **content
            }
        return {
            "status": "ERROR",
            "error": "File not found"
        }
    
    def get_folder_stats(self, agent_id: str) -> Dict:
        """Get storage statistics."""
        return self.fs.get_stats(agent_id)
    
    # =========================================================================
    # Export/Import
    # =========================================================================
    
    def export_memories(self, agent_id: str) -> Dict:
        """Export all memories for backup."""
        return self.store.export_memories(agent_id)
    
    def import_memories(self, agent_id: str, export_data: Dict,
                        merge_mode: str = "append") -> Dict:
        """Import memories from backup."""
        return self.store.import_memories(agent_id, export_data, merge_mode)


# Singleton instance for easy access
_context_manager: Optional[ContextManager] = None


def get_context_manager(root_path: str = None) -> ContextManager:
    """Get or create the global context manager."""
    global _context_manager
    if _context_manager is None or root_path:
        _context_manager = ContextManager(root_path or "./.gitmem_data")
    return _context_manager
