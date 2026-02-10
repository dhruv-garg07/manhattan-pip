"""
GitMem Coding - Virtual File System

Provides a virtual file system abstraction over the coding context store,
enabling agents to navigate, read, and write coding contexts through a
hierarchical folder structure — backed by the .gitmem object store.

Virtual Structure:
    /
    ├── files/
    │   ├── python/
    │   │   └── {filename}.py          # Cached Python files
    │   ├── javascript/
    │   │   └── {filename}.js
    │   ├── typescript/
    │   │   └── {filename}.ts
    │   └── {language}/
    │       └── {filename}
    ├── sessions/
    │   └── {session_id}/
    │       └── session_info.json      # Session read/retrieve logs
    ├── snapshots/
    │   └── {commit_sha}.json          # Version-controlled snapshots
    └── stats/
        └── overview.json              # Aggregate statistics
"""

import os
import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .coding_store import CodingContextStore


@dataclass
class CodingFileNode:
    """Represents a file or directory node in the coding file system."""
    name: str
    path: str
    type: str  # "file" or "directory"
    size: int = 0
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    content_type: str = "file"
    id: Optional[str] = None
    language: Optional[str] = None
    line_count: int = 0
    token_estimate: int = 0

    def to_dict(self) -> Dict:
        d = {
            "name": self.name,
            "path": self.path.strip("/"),
            "type": self.type,
            "size": self.size,
            "last_modified": self.last_modified,
            "content_type": self.content_type,
            "id": self.id
        }
        if self.language:
            d["language"] = self.language
        if self.line_count:
            d["line_count"] = self.line_count
        if self.token_estimate:
            d["token_estimate"] = self.token_estimate
        return d


# Extension mapping for language-based folder display
LANGUAGE_EXTENSIONS = {
    "python": [".py", ".pyw", ".pyi"],
    "javascript": [".js", ".jsx", ".mjs", ".cjs"],
    "typescript": [".ts", ".tsx"],
    "java": [".java"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".h"],
    "c": [".c", ".h"],
    "csharp": [".cs"],
    "go": [".go"],
    "rust": [".rs"],
    "ruby": [".rb"],
    "php": [".php"],
    "swift": [".swift"],
    "kotlin": [".kt", ".kts"],
    "html": [".html", ".htm"],
    "css": [".css", ".less", ".scss", ".sass"],
    "sql": [".sql"],
    "shell": [".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd"],
    "yaml": [".yml", ".yaml"],
    "json": [".json", ".jsonc"],
    "toml": [".toml"],
    "markdown": [".md", ".mdx"],
    "xml": [".xml", ".xsd", ".xsl"],
}


class CodingFileSystem:
    """
    Virtual file system for coding context storage.
    
    Maps cached file contexts to a navigable folder hierarchy organized
    by programming language. Supports list/read/write/delete operations.
    
    Backed by CodingContextStore (JSON) and optionally MemoryDAG (.gitmem)
    for version-controlled snapshots.
    """

    def __init__(self, store: 'CodingContextStore'):
        self.store = store
        self._dag = None  # Lazy-loaded MemoryDAG

    @property
    def dag(self):
        """Lazy-load the MemoryDAG from .gitmem inside the coding store root."""
        if self._dag is None:
            try:
                # Import from the sibling gitmem module
                from ..gitmem.object_store import MemoryDAG
                gitmem_path = str(self.store.root_path / ".gitmem")
                self._dag = MemoryDAG(root_path=gitmem_path)
            except ImportError:
                try:
                    from gitmem.object_store import MemoryDAG
                    gitmem_path = str(self.store.root_path / ".gitmem")
                    self._dag = MemoryDAG(root_path=gitmem_path)
                except ImportError:
                    self._dag = None
        return self._dag

    def _node(
        self,
        name: str,
        is_dir: bool,
        path: str,
        size: int = 0,
        date: str = None,
        content_type: str = "file",
        id: str = None,
        language: str = None,
        line_count: int = 0,
        token_estimate: int = 0
    ) -> Dict:
        """Create a file node dictionary."""
        return CodingFileNode(
            name=name,
            path=path.strip("/"),
            type="directory" if is_dir else "file",
            size=size,
            last_modified=date or datetime.now().isoformat(),
            content_type=content_type,
            id=id,
            language=language,
            line_count=line_count,
            token_estimate=token_estimate
        ).to_dict()

    # =========================================================================
    # Directory Listing
    # =========================================================================

    def list_dir(self, agent_id: str, path: str = "") -> List[Dict]:
        """
        List contents of a virtual path.

        Args:
            agent_id: The agent ID
            path: Virtual path (e.g., "files/python")

        Returns:
            List of file/directory node dicts
        """
        path = path.strip("/")
        parts = path.split("/") if path else []

        # Root
        if not path:
            return [
                self._node("files", True, "files"),
                self._node("sessions", True, "sessions"),
                self._node("snapshots", True, "snapshots"),
                self._node("stats", True, "stats"),
            ]

        category = parts[0].lower()

        if category == "files":
            return self._list_files(agent_id, parts)
        elif category == "sessions":
            return self._list_sessions(agent_id, parts)
        elif category == "snapshots":
            return self._list_snapshots(agent_id, parts)
        elif category == "stats":
            return self._list_stats(agent_id, parts)

        return []

    def _list_files(self, agent_id: str, parts: List[str]) -> List[Dict]:
        """List cached files, organized by language."""
        contexts = self.store._load_agent_data(agent_id, "file_contexts")

        if len(parts) == 1:
            # Show language folders (only languages that have files)
            languages = set()
            for ctx in contexts:
                lang = ctx.get("language", "other")
                languages.add(lang)

            nodes = []
            for lang in sorted(languages):
                count = sum(1 for c in contexts if c.get("language") == lang)
                nodes.append(self._node(
                    f"{lang} ({count})", True, f"files/{lang}"
                ))
            if not nodes:
                nodes.append(self._node("(empty)", True, "files"))
            return nodes

        elif len(parts) == 2:
            # Show files for a specific language
            language = parts[1]
            lang_files = [c for c in contexts if c.get("language") == language]
            return self._contexts_to_nodes(lang_files, f"files/{language}")

        return []

    def _list_sessions(self, agent_id: str, parts: List[str]) -> List[Dict]:
        """List coding sessions."""
        sessions = self.store._load_agent_data(agent_id, "coding_sessions")

        if len(parts) == 1:
            nodes = []
            for sess in sessions[-20:]:  # Last 20 sessions
                sid = sess.get("session_id", "unknown")[:12]
                started = sess.get("started_at", "")[:19]
                nodes.append(self._node(
                    f"session_{sid}.json", False,
                    f"sessions/{sid}.json",
                    content_type="json",
                    date=sess.get("started_at")
                ))
            return nodes

        return []

    def _list_snapshots(self, agent_id: str, parts: List[str]) -> List[Dict]:
        """List version-controlled snapshots from .gitmem."""
        if self.dag is None:
            return [self._node("(no .gitmem initialized)", False, "snapshots")]

        if len(parts) == 1:
            try:
                self.dag.set_agent(agent_id)
                commits = self.dag.log(limit=20)
                nodes = []
                for commit in commits:
                    sha = commit.get("sha", "unknown")[:8]
                    msg = commit.get("message", "snapshot")[:40]
                    nodes.append(self._node(
                        f"{sha}_{msg}.json", False,
                        f"snapshots/{sha}.json",
                        content_type="json",
                        date=commit.get("timestamp")
                    ))
                return nodes
            except Exception:
                return []

        return []

    def _list_stats(self, agent_id: str, parts: List[str]) -> List[Dict]:
        """List statistics files."""
        if len(parts) == 1:
            return [
                self._node("overview.json", False, "stats/overview.json",
                           content_type="json"),
                self._node("token_savings.json", False, "stats/token_savings.json",
                           content_type="json"),
            ]
        return []

    def _contexts_to_nodes(
        self, contexts: List[Dict], base_path: str
    ) -> List[Dict]:
        """Convert file context entries to file nodes."""
        nodes = []
        for ctx in contexts:
            try:
                file_name = ctx.get("file_name", "unknown")
                ctx_id = ctx.get("id", "unknown")[:8]
                nodes.append(self._node(
                    name=file_name,
                    is_dir=False,
                    path=f"{base_path}/{ctx_id}_{file_name}",
                    size=ctx.get("size_bytes", 0),
                    date=ctx.get("last_accessed_at") or ctx.get("created_at"),
                    content_type=ctx.get("language", "other"),
                    id=ctx.get("id"),
                    language=ctx.get("language"),
                    line_count=ctx.get("line_count", 0),
                    token_estimate=ctx.get("token_estimate", 0)
                ))
            except Exception:
                continue
        return nodes

    # =========================================================================
    # Read / Write / Delete
    # =========================================================================

    def read_file(self, agent_id: str, virtual_path: str) -> Optional[Dict]:
        """
        Read a file from the virtual file system.

        Args:
            agent_id: The agent ID
            virtual_path: Path like "files/python/abc12345_server.py"

        Returns:
            Dict with content, metadata, and type; or None if not found
        """
        path = virtual_path.strip("/")
        parts = path.split("/")

        if len(parts) < 2:
            return None

        category = parts[0].lower()

        if category == "files" and len(parts) >= 3:
            # Extract context ID from filename: {ctx_id_8chars}_{filename}
            filename = parts[-1]
            ctx_id_prefix = filename.split("_", 1)[0] if "_" in filename else None

            if ctx_id_prefix:
                contexts = self.store._load_agent_data(agent_id, "file_contexts")
                for ctx in contexts:
                    if ctx.get("id", "").startswith(ctx_id_prefix):
                        return {
                            "content": ctx.get("content", ""),
                            "metadata": {
                                "file_path": ctx.get("file_path", ""),
                                "language": ctx.get("language", ""),
                                "line_count": ctx.get("line_count", 0),
                                "size_bytes": ctx.get("size_bytes", 0),
                                "content_hash": ctx.get("content_hash", ""),
                                "token_estimate": ctx.get("token_estimate", 0),
                                "access_count": ctx.get("access_count", 0),
                                "created_at": ctx.get("created_at", ""),
                                "last_accessed_at": ctx.get("last_accessed_at", ""),
                                "keywords": ctx.get("keywords", []),
                            },
                            "type": ctx.get("language", "text")
                        }

        elif category == "sessions" and len(parts) >= 2:
            session_file = parts[-1]
            sid = session_file.replace(".json", "")
            sessions = self.store._load_agent_data(agent_id, "coding_sessions")
            for sess in sessions:
                if sess.get("session_id", "").startswith(sid):
                    return {
                        "content": json.dumps(sess, indent=2, default=str),
                        "metadata": {},
                        "type": "json"
                    }

        elif category == "snapshots" and len(parts) >= 2:
            if self.dag:
                sha_file = parts[-1]
                sha = sha_file.replace(".json", "").split("_")[0]
                try:
                    self.dag.set_agent(agent_id)
                    state = self.dag.export_state(sha)
                    return {
                        "content": json.dumps(state, indent=2, default=str),
                        "metadata": {"commit_sha": sha},
                        "type": "json"
                    }
                except Exception:
                    return None

        elif category == "stats":
            if len(parts) >= 2 and "overview" in parts[-1]:
                stats = self.store.get_stats(agent_id)
                return {
                    "content": json.dumps(stats, indent=2, default=str),
                    "metadata": {},
                    "type": "json"
                }
            elif len(parts) >= 2 and "token_savings" in parts[-1]:
                savings = self.store.get_token_savings_report(agent_id)
                return {
                    "content": json.dumps(savings, indent=2, default=str),
                    "metadata": {},
                    "type": "json"
                }

        return None

    def write_file(
        self,
        agent_id: str,
        virtual_path: str,
        content: str,
        metadata: Dict = None
    ) -> Optional[str]:
        """
        Write content to the virtual file system.

        Supports writing to files/{language}/{filename} which stores a new
        coding context entry.

        Args:
            agent_id: The agent ID
            virtual_path: Target path, e.g. "files/python/my_script.py"
            content: File content
            metadata: Optional metadata (file_path, keywords, etc.)

        Returns:
            Context ID of created entry, or None on failure
        """
        path = virtual_path.strip("/")
        parts = path.split("/")

        if len(parts) < 3:
            return None

        category = parts[0].lower()

        if category == "files":
            language = parts[1]
            filename = parts[2]

            meta = metadata or {}
            file_path = meta.get("file_path", f"virtual/{language}/{filename}")
            keywords = meta.get("keywords", [])
            session_id = meta.get("session_id", "")
            summary = meta.get("content_summary", "")

            result = self.store.store_file_context(
                agent_id=agent_id,
                file_path=file_path,
                content=content,
                language=language,
                session_id=session_id,
                keywords=keywords,
                content_summary=summary
            )
            return result.get("context_id")

        return None

    def delete_file(self, agent_id: str, virtual_path: str) -> bool:
        """
        Delete a file from the virtual file system.

        Args:
            agent_id: The agent ID
            virtual_path: Path to delete

        Returns:
            True if deleted, False otherwise
        """
        path = virtual_path.strip("/")
        parts = path.split("/")

        if len(parts) < 3:
            return False

        category = parts[0].lower()

        if category == "files":
            filename = parts[-1]
            ctx_id_prefix = filename.split("_", 1)[0] if "_" in filename else None

            if ctx_id_prefix:
                contexts = self.store._load_agent_data(agent_id, "file_contexts")
                for ctx in contexts:
                    if ctx.get("id", "").startswith(ctx_id_prefix):
                        result = self.store.delete_context(
                            agent_id=agent_id,
                            context_id=ctx["id"]
                        )
                        return result.get("deleted_count", 0) > 0

        return False

    # =========================================================================
    # Version Control Operations (via .gitmem MemoryDAG)
    # =========================================================================

    def commit_snapshot(
        self,
        agent_id: str,
        message: str = "Coding context snapshot"
    ) -> Optional[str]:
        """
        Commit the current coding context state to .gitmem.

        Creates an immutable snapshot of all cached file contexts,
        stored as blobs in the MemoryDAG.

        Args:
            agent_id: The agent ID
            message: Commit message

        Returns:
            Commit SHA, or None if .gitmem unavailable
        """
        if self.dag is None:
            return None

        try:
            self.dag.set_agent(agent_id)
            self.dag.reset()  # Clear staging area

            contexts = self.store._load_agent_data(agent_id, "file_contexts")

            for ctx in contexts:
                # Stage each file context as a blob
                self.dag.add(
                    content=json.dumps({
                        "file_path": ctx.get("file_path", ""),
                        "file_name": ctx.get("file_name", ""),
                        "content_hash": ctx.get("content_hash", ""),
                        "language": ctx.get("language", ""),
                        "line_count": ctx.get("line_count", 0),
                        "size_bytes": ctx.get("size_bytes", 0),
                        "token_estimate": ctx.get("token_estimate", 0),
                    }, default=str),
                    memory_type="coding_context",
                    importance=0.7,
                    tags=ctx.get("keywords", []),
                    metadata={
                        "file_path": ctx.get("file_path", ""),
                        "language": ctx.get("language", "")
                    }
                )

            sha = self.dag.commit(message=message, author=f"agent:{agent_id}")
            return sha

        except Exception:
            return None

    def get_history(self, agent_id: str, limit: int = 10) -> List[Dict]:
        """Get commit history from .gitmem."""
        if self.dag is None:
            return []

        try:
            self.dag.set_agent(agent_id)
            return self.dag.log(limit=limit)
        except Exception:
            return []

    def get_snapshot(self, agent_id: str, commit_sha: str) -> Optional[Dict]:
        """Get a specific snapshot from .gitmem."""
        if self.dag is None:
            return None

        try:
            self.dag.set_agent(agent_id)
            return self.dag.export_state(commit_sha)
        except Exception:
            return None

    def get_diff(
        self, agent_id: str, sha_a: str, sha_b: str
    ) -> Optional[Dict]:
        """Compare two snapshots."""
        if self.dag is None:
            return None

        try:
            self.dag.set_agent(agent_id)
            return self.dag.diff(sha_a, sha_b)
        except Exception:
            return None

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        Get comprehensive file system statistics.

        Returns:
            Dict with counts and sizes by folder/language
        """
        contexts = self.store._load_agent_data(agent_id, "file_contexts")
        sessions = self.store._load_agent_data(agent_id, "coding_sessions")

        lang_stats = {}
        for ctx in contexts:
            lang = ctx.get("language", "other")
            if lang not in lang_stats:
                lang_stats[lang] = {"count": 0, "total_bytes": 0, "total_tokens": 0}
            lang_stats[lang]["count"] += 1
            lang_stats[lang]["total_bytes"] += ctx.get("size_bytes", 0)
            lang_stats[lang]["total_tokens"] += ctx.get("token_estimate", 0)

        snapshot_count = 0
        if self.dag is not None:
            try:
                self.dag.set_agent(agent_id)
                snapshot_count = len(self.dag.log(limit=100))
            except Exception:
                pass

        return {
            "agent_id": agent_id,
            "files": {
                "total": len(contexts),
                "by_language": lang_stats
            },
            "sessions": {
                "total": len(sessions)
            },
            "snapshots": {
                "total": snapshot_count,
                "gitmem_available": self.dag is not None
            },
            "total_size_bytes": sum(c.get("size_bytes", 0) for c in contexts),
            "total_token_estimate": sum(c.get("token_estimate", 0) for c in contexts),
        }
