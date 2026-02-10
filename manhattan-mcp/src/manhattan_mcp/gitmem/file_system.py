"""
GitMem Local - Virtual File System

Provides a virtual file system abstraction over the object store.
Maps memory types to a folder-like structure for easy navigation.

Virtual Structure:
    /
    ├── context/
    │   ├── episodic/
    │   ├── semantic/
    │   ├── procedural/
    │   └── working/
    ├── documents/
    │   ├── knowledge/
    │   └── references/
    ├── checkpoints/
    │   ├── snapshots/
    │   └── sessions/
    ├── logs/
    │   └── activity/
    └── agents/
        └── {agent_id}/
"""

import os
import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .memory_store import LocalMemoryStore


class FolderType(Enum):
    """Types of folders in the GitMem structure."""
    # Context Store
    EPISODIC = "context/episodic"
    SEMANTIC = "context/semantic"
    PROCEDURAL = "context/procedural"
    WORKING = "context/working"
    
    # Documents
    KNOWLEDGE = "documents/knowledge"
    REFERENCES = "documents/references"
    
    # Checkpoints
    SNAPSHOTS = "checkpoints/snapshots"
    SESSIONS = "checkpoints/sessions"
    
    # Logs
    ACTIVITY = "logs/activity"
    ERRORS = "logs/errors"


class AccessLevel(Enum):
    """Access control levels for folders."""
    READ = "read"
    WRITE = "write"
    APPEND = "append"
    DELETE = "delete"
    FULL = "full"
    NONE = "none"


@dataclass
class FolderPermissions:
    """Permissions for a folder."""
    user_read: bool = True
    user_write: bool = False
    user_delete: bool = False
    agent_read: bool = True
    agent_write: bool = True
    agent_delete: bool = False


# Default permissions for each folder type
FOLDER_PERMISSIONS = {
    # Context Store - Agent-managed
    FolderType.EPISODIC: FolderPermissions(user_read=True, user_write=False, agent_write=True),
    FolderType.SEMANTIC: FolderPermissions(user_read=True, user_write=True, agent_write=True),
    FolderType.PROCEDURAL: FolderPermissions(user_read=True, user_write=True, agent_write=True),
    FolderType.WORKING: FolderPermissions(user_read=True, user_write=False, agent_write=True),
    
    # Documents - User-managed
    FolderType.KNOWLEDGE: FolderPermissions(user_read=True, user_write=True, user_delete=True, agent_write=False),
    FolderType.REFERENCES: FolderPermissions(user_read=True, user_write=True, user_delete=True, agent_write=False),
    
    # Checkpoints - Agent-managed
    FolderType.SNAPSHOTS: FolderPermissions(user_read=True, user_write=False, agent_write=True),
    FolderType.SESSIONS: FolderPermissions(user_read=True, user_write=False, agent_write=True),
    
    # Logs - System-managed
    FolderType.ACTIVITY: FolderPermissions(user_read=True, user_write=False, agent_write=True),
    FolderType.ERRORS: FolderPermissions(user_read=True, user_write=False, agent_write=True),
}


@dataclass
class FileNode:
    """Represents a file or directory node."""
    name: str
    path: str
    type: str  # "file" or "directory"
    size: int = 0
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    content_type: str = "file"
    id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "path": self.path.strip("/"),
            "type": self.type,
            "size": self.size,
            "last_modified": self.last_modified,
            "content_type": self.content_type,
            "id": self.id
        }


class LocalFileSystem:
    """
    Virtual file system implementation for local AI context storage.
    
    Maps memory entries to a hierarchical folder structure.
    """
    
    def __init__(self, store: 'LocalMemoryStore'):
        self.store = store
    
    def _create_node(self, name: str, is_dir: bool, path: str, 
                     size: int = 0, date: str = None, 
                     content_type: str = "file", id: str = None) -> Dict:
        """Create a file node dictionary."""
        return FileNode(
            name=name,
            path=path.strip("/"),
            type="directory" if is_dir else "file",
            size=size,
            last_modified=date or datetime.now().isoformat(),
            content_type=content_type,
            id=id
        ).to_dict()
    
    def list_dir(self, agent_id: str, path: str = "") -> List[Dict]:
        """
        List contents of a virtual path.
        
        Args:
            agent_id: The agent ID
            path: Virtual path (e.g., "context/episodic")
        
        Returns:
            List of file/directory nodes
        """
        path = path.strip("/")
        parts = path.split("/") if path else []
        
        # Root
        if not path:
            return [
                self._create_node("context", True, "context"),
                self._create_node("documents", True, "documents"),
                self._create_node("checkpoints", True, "checkpoints"),
                self._create_node("logs", True, "logs"),
                self._create_node("agents", True, "agents")
            ]
        
        category = parts[0].lower()
        
        # Context Store
        if category == "context":
            if len(parts) == 1:
                return [
                    self._create_node("episodic", True, "context/episodic"),
                    self._create_node("semantic", True, "context/semantic"),
                    self._create_node("procedural", True, "context/procedural"),
                    self._create_node("working", True, "context/working")
                ]
            else:
                memory_type = parts[1]
                items = self.store.list_memories(agent_id, memory_type, limit=100)
                return self._memories_to_nodes(items, f"context/{memory_type}")
        
        # Documents
        elif category == "documents":
            if len(parts) == 1:
                return [
                    self._create_node("knowledge", True, "documents/knowledge"),
                    self._create_node("references", True, "documents/references")
                ]
            else:
                doc_type = parts[1]
                items = self.store.list_documents(agent_id, doc_type, limit=100)
                return self._documents_to_nodes(items, f"documents/{doc_type}")
        
        # Checkpoints
        elif category == "checkpoints":
            if len(parts) == 1:
                return [
                    self._create_node("snapshots", True, "checkpoints/snapshots"),
                    self._create_node("sessions", True, "checkpoints/sessions")
                ]
            else:
                checkpoint_type = parts[1]
                items = self.store.list_checkpoints(agent_id, checkpoint_type, limit=50)
                return self._checkpoints_to_nodes(items, f"checkpoints/{checkpoint_type}")
        
        # Logs
        elif category == "logs":
            if len(parts) == 1:
                return [
                    self._create_node("activity", True, "logs/activity"),
                    self._create_node("errors", True, "logs/errors")
                ]
            else:
                log_type = parts[1]
                items = self.store.list_logs(agent_id, log_type, limit=100)
                return self._logs_to_nodes(items, f"logs/{log_type}")
        
        # Agents
        elif category == "agents":
            if len(parts) == 1:
                agents = self.store.list_agents()
                return [self._create_node(a, True, f"agents/{a}") for a in agents]
            else:
                target_agent = parts[1]
                # Show that agent's memory structure
                return [
                    self._create_node("context", True, f"agents/{target_agent}/context"),
                    self._create_node("stats", False, f"agents/{target_agent}/stats.json", 
                                      content_type="json")
                ]
        
        return []
    
    def _memories_to_nodes(self, memories: List[Dict], base_path: str) -> List[Dict]:
        """Convert memory entries to file nodes."""
        nodes = []
        for mem in memories:
            try:
                ts = str(mem.get('created_at', ''))[:19].replace(':', '-') or "unknown"
                mem_id = str(mem.get('id', 'unknown'))
                content = mem.get('content', '') or mem.get('lossless_restatement', '')
                nodes.append(self._create_node(
                    name=f"{ts}_{mem_id[:8]}.json",
                    is_dir=False,
                    path=f"{base_path}/{ts}_{mem_id[:8]}.json",
                    size=len(str(content)),
                    date=mem.get('created_at'),
                    content_type="json",
                    id=mem_id
                ))
            except Exception:
                continue
        return nodes
    
    def _documents_to_nodes(self, documents: List[Dict], base_path: str) -> List[Dict]:
        """Convert documents to file nodes."""
        nodes = []
        for doc in documents:
            try:
                doc_id = str(doc.get('id', 'unknown'))
                filename = doc.get('filename', f'Doc_{doc_id[:8]}.md')
                nodes.append(self._create_node(
                    name=filename,
                    is_dir=False,
                    path=f"{base_path}/{filename}",
                    size=doc.get('size_bytes', 0),
                    date=doc.get('created_at'),
                    content_type=doc.get('content_type', 'markdown'),
                    id=doc_id
                ))
            except Exception:
                continue
        return nodes
    
    def _checkpoints_to_nodes(self, checkpoints: List[Dict], base_path: str) -> List[Dict]:
        """Convert checkpoints to file nodes."""
        nodes = []
        for cp in checkpoints:
            try:
                ts = str(cp.get('created_at', ''))[:19].replace(':', '-') or "unknown"
                cp_id = str(cp.get('id', 'unknown'))
                nodes.append(self._create_node(
                    name=f"Checkpoint_{ts}_{cp_id[:8]}.json",
                    is_dir=False,
                    path=f"{base_path}/Checkpoint_{ts}_{cp_id[:8]}.json",
                    size=len(str(cp.get('metadata', {}))),
                    date=cp.get('created_at'),
                    content_type="json",
                    id=cp_id
                ))
            except Exception:
                continue
        return nodes
    
    def _logs_to_nodes(self, logs: List[Dict], base_path: str) -> List[Dict]:
        """Convert logs to file nodes."""
        nodes = []
        for log in logs:
            try:
                ts = str(log.get('created_at', ''))[:19].replace(':', '-') or "unknown"
                log_id = str(log.get('id', 'unknown'))
                nodes.append(self._create_node(
                    name=f"Log_{ts}_{log_id[:8]}.txt",
                    is_dir=False,
                    path=f"{base_path}/Log_{ts}_{log_id[:8]}.txt",
                    size=len(str(log.get('details', {}))),
                    date=log.get('created_at'),
                    content_type="text",
                    id=log_id
                ))
            except Exception:
                continue
        return nodes
    
    def read_file(self, agent_id: str, virtual_path: str) -> Optional[Dict]:
        """
        Read file content from a virtual path.
        
        Args:
            agent_id: The agent ID
            virtual_path: Path like "context/episodic/2024-01-01_abc123.json"
        
        Returns:
            Dict with content, metadata, and type
        """
        path = virtual_path.strip("/")
        parts = path.split("/")
        
        if len(parts) < 3:
            return None
        
        category = parts[0].lower()
        subtype = parts[1]
        filename = parts[2]
        
        # Extract ID from filename (format: ..._ID.ext)
        try:
            file_id_part = filename.rsplit('_', 1)[-1]
            item_id = file_id_part.split('.')[0]
        except Exception:
            return None
        
        if category == "context":
            mem = self.store.get_memory_by_id(agent_id, item_id)
            if mem:
                return {
                    "content": json.dumps(mem, indent=2, sort_keys=True, default=str),
                    "metadata": mem.get('metadata', {}),
                    "type": "json"
                }
        
        elif category == "documents":
            doc = self.store.get_document_by_id(agent_id, item_id)
            if doc:
                return {
                    "content": doc.get('content', ''),
                    "metadata": doc.get('metadata', {}),
                    "type": doc.get('content_type', 'markdown')
                }
        
        elif category == "checkpoints":
            cp = self.store.get_checkpoint_by_id(agent_id, item_id)
            if cp:
                return {
                    "content": json.dumps(cp, indent=2, default=str),
                    "metadata": cp.get('metadata', {}),
                    "type": "json"
                }
        
        elif category == "logs":
            log = self.store.get_log_by_id(agent_id, item_id)
            if log:
                text = f"{log.get('created_at', '')} - {log.get('action', '')}\n\n"
                text += f"Details:\n{json.dumps(log.get('details', {}), indent=2)}"
                return {
                    "content": text,
                    "metadata": {},
                    "type": "text"
                }
        
        return None
    
    def write_file(self, agent_id: str, virtual_path: str, content: str, 
                   metadata: Dict = None) -> Optional[str]:
        """
        Write content to a virtual path.
        
        Args:
            agent_id: The agent ID
            virtual_path: Target path
            content: Content to write
            metadata: Optional metadata
        
        Returns:
            ID of created item, or None on failure
        """
        path = virtual_path.strip("/")
        parts = path.split("/")
        
        if len(parts) < 2:
            return None
        
        category = parts[0].lower()
        subtype = parts[1]
        
        if category == "context":
            return self.store.add_memory(
                agent_id=agent_id,
                content=content,
                memory_type=subtype,
                metadata=metadata or {}
            )
        
        elif category == "documents":
            filename = parts[2] if len(parts) > 2 else f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            return self.store.add_document(
                agent_id=agent_id,
                filename=filename,
                content=content,
                folder=subtype,
                metadata=metadata or {}
            )
        
        return None
    
    def delete_file(self, agent_id: str, virtual_path: str) -> bool:
        """
        Delete a file from a virtual path.
        
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
        filename = parts[-1]
        
        # Extract ID from filename
        try:
            file_id_part = filename.rsplit('_', 1)[-1]
            item_id = file_id_part.split('.')[0]
        except Exception:
            return False
        
        if category == "context":
            return self.store.delete_memory(agent_id, item_id)
        elif category == "documents":
            return self.store.delete_document(agent_id, item_id)
        elif category == "checkpoints":
            return self.store.delete_checkpoint(agent_id, item_id)
        elif category == "logs":
            return self.store.delete_log(agent_id, item_id)
        
        return False
    
    def get_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        Get storage statistics for an agent.
        
        Returns:
            Dict with counts and sizes for each folder type
        """
        stats = {
            "agent_id": agent_id,
            "context": {
                "episodic": self.store.count_memories(agent_id, "episodic"),
                "semantic": self.store.count_memories(agent_id, "semantic"),
                "procedural": self.store.count_memories(agent_id, "procedural"),
                "working": self.store.count_memories(agent_id, "working"),
            },
            "documents": {
                "knowledge": self.store.count_documents(agent_id, "knowledge"),
                "references": self.store.count_documents(agent_id, "references"),
            },
            "checkpoints": {
                "snapshots": self.store.count_checkpoints(agent_id, "snapshots"),
                "sessions": self.store.count_checkpoints(agent_id, "sessions"),
            },
            "logs": {
                "activity": self.store.count_logs(agent_id, "activity"),
                "errors": self.store.count_logs(agent_id, "errors"),
            },
            "total_memories": 0,
            "total_documents": 0,
        }
        
        # Calculate totals
        stats["total_memories"] = sum(stats["context"].values())
        stats["total_documents"] = sum(stats["documents"].values())
        
        return stats
    
    def check_permission(self, folder_type: FolderType, actor: str, action: str) -> bool:
        """Check if an actor has permission to perform an action on a folder."""
        perms = FOLDER_PERMISSIONS.get(folder_type)
        if not perms:
            return False
        
        if actor == "user":
            if action == "read":
                return perms.user_read
            elif action == "write":
                return perms.user_write
            elif action == "delete":
                return perms.user_delete
        elif actor == "agent":
            if action == "read":
                return perms.agent_read
            elif action == "write":
                return perms.agent_write
            elif action == "delete":
                return perms.agent_delete
        
        return False
