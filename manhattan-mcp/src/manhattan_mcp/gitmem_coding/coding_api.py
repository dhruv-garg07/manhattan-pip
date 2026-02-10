"""
GitMem Coding - Coding Context API

High-level API for coding context storage and retrieval.
Wraps CodingContextStore with a clean, simple interface.

Usage:
    from manhattan_mcp.gitmem_coding import CodingAPI
    
    api = CodingAPI()
    
    # Store file content
    api.store_file("agent-1", "/path/to/file.py", content, "python")
    
    # Retrieve cached content (with freshness check)
    result = api.retrieve_file("agent-1", "/path/to/file.py")
    if result["status"] == "cache_hit" and result["freshness"] == "fresh":
        # Use cached content — tokens saved!
        content = result["content"]
    
    # Search cached files
    results = api.search_files("agent-1", "models python")
    
    # Get stats and token savings
    stats = api.get_stats("agent-1")
"""

import os
from typing import List, Dict, Any, Optional

from .coding_store import CodingContextStore
from .coding_file_system import CodingFileSystem


# Module-level singleton
_coding_api_instance = None


class CodingAPI:
    """
    High-level API for coding context storage.
    
    All operations are local — no network calls.
    Data is stored in JSON files under .gitmem_coding/.
    """
    
    def __init__(self, root_path: str = "./.gitmem_coding"):
        """
        Initialize the Coding API.
        
        Args:
            root_path: Root directory for coding context storage
        """
        self.store = CodingContextStore(root_path=root_path)
        self.fs = CodingFileSystem(store=self.store)
    
    # =========================================================================
    # File Context Operations
    # =========================================================================
    
    def store_file(
        self,
        agent_id: str,
        file_path: str,
        content: str,
        language: str = "other",
        session_id: str = "",
        keywords: List[str] = None,
        content_summary: str = ""
    ) -> Dict[str, Any]:
        """
        Store file content in the coding context cache.
        
        If the file is already cached, updates it. Uses hash-based
        deduplication to detect if content actually changed.
        
        Args:
            agent_id: Agent identifier
            file_path: Absolute path to the file
            content: Full file content
            language: Programming language (python, javascript, etc.)
            session_id: Current session ID (for tracking)
            keywords: Optional searchable keywords
            content_summary: Optional brief description
        
        Returns:
            Dict with status, context_id, token_estimate
        """
        result = self.store.store_file_context(
            agent_id=agent_id,
            file_path=file_path,
            content=content,
            language=language,
            session_id=session_id,
            keywords=keywords,
            content_summary=content_summary
        )
        
        # Track session activity
        if session_id:
            self.store.record_session_activity(
                agent_id=agent_id,
                session_id=session_id,
                action="store",
                file_path=file_path,
                tokens=result.get("token_estimate", 0)
            )
        
        return result
    
    def retrieve_file(
        self,
        agent_id: str,
        file_path: str,
        session_id: str = ""
    ) -> Dict[str, Any]:
        """
        Retrieve cached file content with freshness check.
        
        Returns the cached content along with a freshness status:
        - "fresh": Content matches current file on disk
        - "stale": File has been modified since caching
        - "missing": Original file no longer exists
        - "unknown": Unable to determine freshness
        
        Args:
            agent_id: Agent identifier
            file_path: Path to the file
            session_id: Current session ID (for tracking)
        
        Returns:
            Dict with content, freshness status, token savings
        """
        result = self.store.retrieve_file_context(
            agent_id=agent_id,
            file_path=file_path
        )
        
        # Track session activity
        if session_id:
            if result["status"] == "cache_hit":
                self.store.record_session_activity(
                    agent_id=agent_id,
                    session_id=session_id,
                    action="retrieve",
                    file_path=file_path,
                    tokens=result.get("token_savings", 0)
                )
            else:
                self.store.record_session_activity(
                    agent_id=agent_id,
                    session_id=session_id,
                    action="miss",
                    file_path=file_path,
                    tokens=0
                )
        
        return result
    
    def search_files(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Search across cached file contexts.
        
        Searches by filename, path, language, and keywords.
        Returns summaries without full content for efficiency.
        
        Args:
            agent_id: Agent identifier
            query: Search query
            top_k: Maximum results
        
        Returns:
            Dict with matching file summaries and count
        """
        results = self.store.search_contexts(
            agent_id=agent_id,
            query=query,
            top_k=top_k
        )
        
        return {
            "status": "OK",
            "results": results,
            "count": len(results),
            "query": query
        }
    
    def list_files(
        self,
        agent_id: str,
        limit: int = 50,
        offset: int = 0,
        language: str = None
    ) -> Dict[str, Any]:
        """
        List all cached file contexts.
        
        Returns summaries without full content.
        
        Args:
            agent_id: Agent identifier
            limit: Maximum items
            offset: Pagination offset
            language: Optional language filter
        
        Returns:
            Paginated list of context summaries
        """
        return self.store.list_contexts(
            agent_id=agent_id,
            limit=limit,
            offset=offset,
            language=language
        )
    
    def delete_file(
        self,
        agent_id: str,
        file_path: str = None,
        context_id: str = None
    ) -> Dict[str, Any]:
        """
        Delete a cached file context.
        
        Args:
            agent_id: Agent identifier
            file_path: File path to remove from cache
            context_id: Or context entry ID to remove
        
        Returns:
            Deletion status
        """
        return self.store.delete_context(
            agent_id=agent_id,
            file_path=file_path,
            context_id=context_id
        )
    
    def get_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        Get comprehensive coding context statistics.
        
        Includes file counts, storage size, freshness breakdown,
        and estimated token savings.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Comprehensive statistics dict
        """
        stats = self.store.get_stats(agent_id)
        savings = self.store.get_token_savings_report(agent_id)
        
        return {
            **stats,
            "token_savings_report": savings
        }
    
    def get_token_savings(self, agent_id: str) -> Dict[str, Any]:
        """
        Get focused token savings report.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Token savings breakdown
        """
        return self.store.get_token_savings_report(agent_id)
    
    def cleanup(
        self,
        agent_id: str,
        days_threshold: int = None
    ) -> Dict[str, Any]:
        """
        Clean up stale cached contexts.
        
        Removes files not accessed within the threshold period.
        
        Args:
            agent_id: Agent identifier
            days_threshold: Days since last access (default: 30)
        
        Returns:
            Cleanup results
        """
        return self.store.cleanup_stale_contexts(
            agent_id=agent_id,
            days_threshold=days_threshold
        )
    
    # =========================================================================
    # Virtual File System Operations
    # =========================================================================

    def list_dir(self, agent_id: str, path: str = "") -> Dict[str, Any]:
        """
        List contents of a virtual directory in the coding file system.

        Virtual structure:
            /files/{language}/     - Cached files by language
            /sessions/            - Session read/retrieve logs
            /snapshots/           - Version-controlled snapshots (.gitmem)
            /stats/               - Aggregate statistics

        Args:
            agent_id: Agent identifier
            path: Virtual path (e.g., "files/python")

        Returns:
            Dict with directory listing
        """
        items = self.fs.list_dir(agent_id, path)
        return {
            "status": "OK",
            "path": path or "/",
            "items": items,
            "count": len(items)
        }

    def read_vfs_file(self, agent_id: str, path: str) -> Dict[str, Any]:
        """
        Read a file from the virtual file system.

        Args:
            agent_id: Agent identifier
            path: Virtual file path (e.g., "files/python/abc12345_server.py")

        Returns:
            Dict with content, metadata, and type
        """
        result = self.fs.read_file(agent_id, path)
        if result is None:
            return {"status": "not_found", "path": path}
        return {"status": "OK", **result}

    def write_vfs_file(
        self, agent_id: str, path: str, content: str,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """
        Write content to the virtual file system.

        Args:
            agent_id: Agent identifier
            path: Virtual path (e.g., "files/python/my_script.py")
            content: File content
            metadata: Optional metadata (file_path, keywords, etc.)

        Returns:
            Dict with created context_id
        """
        context_id = self.fs.write_file(agent_id, path, content, metadata)
        if context_id:
            return {"status": "OK", "context_id": context_id}
        return {"status": "error", "message": "Could not write to path."}

    def delete_vfs_file(self, agent_id: str, path: str) -> Dict[str, Any]:
        """
        Delete a file from the virtual file system.

        Args:
            agent_id: Agent identifier
            path: Virtual path to delete

        Returns:
            Deletion status
        """
        deleted = self.fs.delete_file(agent_id, path)
        return {
            "status": "OK" if deleted else "not_found",
            "deleted": deleted
        }

    # =========================================================================
    # Version Control (.gitmem snapshots)
    # =========================================================================

    def commit_snapshot(
        self, agent_id: str, message: str = "Coding context snapshot"
    ) -> Dict[str, Any]:
        """
        Create an immutable snapshot of coding contexts in .gitmem.

        Args:
            agent_id: Agent identifier
            message: Commit message

        Returns:
            Dict with commit SHA
        """
        sha = self.fs.commit_snapshot(agent_id, message)
        if sha:
            return {"status": "OK", "commit_sha": sha, "message": message}
        return {"status": "error", "message": ".gitmem not available or commit failed."}

    def get_history(
        self, agent_id: str, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get commit history from .gitmem.

        Args:
            agent_id: Agent identifier
            limit: Maximum commits to return

        Returns:
            Dict with commit history
        """
        commits = self.fs.get_history(agent_id, limit)
        return {
            "status": "OK",
            "commits": commits,
            "count": len(commits)
        }

    def get_snapshot(
        self, agent_id: str, commit_sha: str
    ) -> Dict[str, Any]:
        """
        Get a specific snapshot from .gitmem.

        Args:
            agent_id: Agent identifier
            commit_sha: Commit hash

        Returns:
            Snapshot data
        """
        state = self.fs.get_snapshot(agent_id, commit_sha)
        if state:
            return {"status": "OK", "snapshot": state}
        return {"status": "not_found", "commit_sha": commit_sha}

    # =========================================================================
    # Agent Operations
    # =========================================================================

    def list_agents(self) -> List[str]:
        """List all agents with coding context data."""
        return self.store.list_agents()

    def delete_agent_data(self, agent_id: str) -> bool:
        """Delete all coding context data for an agent."""
        return self.store.delete_agent_data(agent_id)


def get_coding_api(root_path: str = None) -> CodingAPI:
    """
    Get or create the global CodingAPI singleton.

    Args:
        root_path: Optional custom root path

    Returns:
        CodingAPI singleton instance
    """
    global _coding_api_instance
    if _coding_api_instance is None:
        if root_path:
            _coding_api_instance = CodingAPI(root_path=root_path)
        else:
            _coding_api_instance = CodingAPI()
    return _coding_api_instance
