"""
GitMem Local - Object Store

Content-addressable storage for AI memory, mimicking Git's object model:
- Blobs: Raw memory content (SHA-256 hashed)
- Trees: Cognitive state snapshots (collections of memories)
- Commits: Immutable snapshots with parent links

Storage structure:
    .gitmem/
    ├── objects/
    │   ├── 2e/d5a7...  (first 2 chars as directory)
    │   └── ...
    ├── refs/
    │   ├── heads/main
    │   ├── tags/v1.0
    │   └── agents/agent-007
    └── HEAD
"""

import os
import json
import hashlib
import zlib
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


class ObjectType(str, Enum):
    BLOB = "blob"       # Raw memory content
    TREE = "tree"       # Directory/collection of memories
    COMMIT = "commit"   # Snapshot with parent links


@dataclass
class MemoryBlob:
    """Raw memory content, content-addressed by SHA-256."""
    content: str
    memory_type: str = "episodic"
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    persons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def sha(self) -> str:
        """Compute SHA-256 hash of the blob content."""
        raw = json.dumps(self.to_dict(), sort_keys=True).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()
    
    def to_dict(self) -> Dict:
        return {
            "type": ObjectType.BLOB.value,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "tags": self.tags,
            "keywords": self.keywords,
            "persons": self.persons,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryBlob':
        return cls(
            content=data["content"],
            memory_type=data.get("memory_type", "episodic"),
            importance=data.get("importance", 0.5),
            tags=data.get("tags", []),
            keywords=data.get("keywords", []),
            persons=data.get("persons", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat())
        )


@dataclass
class TreeEntry:
    """An entry in a cognitive tree - references a blob."""
    mode: str           # "memory", "fact", "procedure", "state"
    sha: str            # SHA of the blob
    path: str           # Logical path (e.g., "episodic/observation-001")
    name: str           # Human-readable name
    
    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "sha": self.sha,
            "path": self.path,
            "name": self.name
        }


@dataclass
class CognitiveTree:
    """A snapshot of cognitive state - collection of memory references."""
    entries: List[TreeEntry] = field(default_factory=list)
    
    @property
    def sha(self) -> str:
        """Compute SHA-256 hash of the tree."""
        raw = json.dumps(self.to_dict(), sort_keys=True).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()
    
    def to_dict(self) -> Dict:
        return {
            "type": ObjectType.TREE.value,
            "entries": [e.to_dict() for e in self.entries]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CognitiveTree':
        entries = [TreeEntry(**e) for e in data.get("entries", [])]
        return cls(entries=entries)
    
    def add_entry(self, entry: TreeEntry):
        self.entries.append(entry)
    
    def get_entry(self, path: str) -> Optional[TreeEntry]:
        for e in self.entries:
            if e.path == path:
                return e
        return None


@dataclass
class MemoryCommit:
    """Immutable commit object - snapshot of cognitive state."""
    tree_sha: str                           # SHA of root tree
    message: str
    author: str
    agent_id: str
    parents: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    stats: Dict[str, int] = field(default_factory=dict)
    
    @property
    def sha(self) -> str:
        """Compute SHA-256 hash of the commit."""
        raw = json.dumps(self.to_dict(), sort_keys=True).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()
    
    def to_dict(self) -> Dict:
        return {
            "type": ObjectType.COMMIT.value,
            "tree": self.tree_sha,
            "parents": self.parents,
            "author": self.author,
            "agent_id": self.agent_id,
            "message": self.message,
            "timestamp": self.timestamp,
            "stats": self.stats
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryCommit':
        return cls(
            tree_sha=data["tree"],
            message=data["message"],
            author=data.get("author", "system"),
            agent_id=data.get("agent_id", "system"),
            parents=data.get("parents", []),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            stats=data.get("stats", {})
        )


class ObjectStore:
    """
    Content-addressable object storage for GitMem.
    
    Objects are stored in .gitmem/objects/ using the first 2 characters
    of the SHA as a subdirectory (like Git).
    """
    
    def __init__(self, root_path: str = "./.gitmem"):
        self.root_path = os.path.abspath(root_path)
        self.objects_path = os.path.join(self.root_path, "objects")
        self.refs_path = os.path.join(self.root_path, "refs")
        self.heads_path = os.path.join(self.refs_path, "heads")
        self.tags_path = os.path.join(self.refs_path, "tags")
        self.agents_path = os.path.join(self.refs_path, "agents")
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """Create necessary directory structure."""
        for path in [
            self.objects_path,
            self.heads_path,
            self.tags_path,
            self.agents_path
        ]:
            os.makedirs(path, exist_ok=True)
        
        # Initialize HEAD if not exists
        head_path = os.path.join(self.root_path, "HEAD")
        if not os.path.exists(head_path):
            with open(head_path, "w") as f:
                f.write("ref: refs/heads/main\n")
    
    def _object_path(self, sha: str) -> str:
        """Get the file path for an object by its SHA."""
        return os.path.join(self.objects_path, sha[:2], sha[2:])
    
    # ========== Object Storage ==========
    
    def write_object(self, obj_data: Dict, sha: str) -> str:
        """Write an object to the store."""
        path = self._object_path(sha)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Compress the object (like Git)
        raw = json.dumps(obj_data, sort_keys=True).encode('utf-8')
        compressed = zlib.compress(raw)
        
        with open(path, "wb") as f:
            f.write(compressed)
        
        return sha
    
    def read_object(self, sha: str) -> Optional[Dict]:
        """Read an object from the store."""
        path = self._object_path(sha)
        if not os.path.exists(path):
            return None
        
        with open(path, "rb") as f:
            compressed = f.read()
        
        raw = zlib.decompress(compressed)
        return json.loads(raw.decode('utf-8'))
    
    def object_exists(self, sha: str) -> bool:
        """Check if an object exists."""
        return os.path.exists(self._object_path(sha))
    
    # ========== High-Level Operations ==========
    
    def store_blob(self, blob: MemoryBlob) -> str:
        """Store a memory blob and return its SHA."""
        sha = blob.sha
        if not self.object_exists(sha):
            self.write_object(blob.to_dict(), sha)
        return sha
    
    def store_tree(self, tree: CognitiveTree) -> str:
        """Store a cognitive tree and return its SHA."""
        sha = tree.sha
        if not self.object_exists(sha):
            self.write_object(tree.to_dict(), sha)
        return sha
    
    def store_commit(self, commit: MemoryCommit) -> str:
        """Store a commit and return its SHA."""
        sha = commit.sha
        if not self.object_exists(sha):
            self.write_object(commit.to_dict(), sha)
        return sha
    
    def get_blob(self, sha: str) -> Optional[MemoryBlob]:
        """Retrieve a blob by SHA."""
        data = self.read_object(sha)
        if data and data.get("type") == ObjectType.BLOB.value:
            return MemoryBlob.from_dict(data)
        return None
    
    def get_tree(self, sha: str) -> Optional[CognitiveTree]:
        """Retrieve a tree by SHA."""
        data = self.read_object(sha)
        if data and data.get("type") == ObjectType.TREE.value:
            return CognitiveTree.from_dict(data)
        return None
    
    def get_commit(self, sha: str) -> Optional[MemoryCommit]:
        """Retrieve a commit by SHA."""
        data = self.read_object(sha)
        if data and data.get("type") == ObjectType.COMMIT.value:
            return MemoryCommit.from_dict(data)
        return None
    
    # ========== Refs Management ==========
    
    def get_head(self) -> str:
        """Get the current HEAD reference."""
        head_path = os.path.join(self.root_path, "HEAD")
        with open(head_path, "r") as f:
            content = f.read().strip()
        
        # Check if it's a symbolic ref
        if content.startswith("ref: "):
            ref_path = content[5:]
            return self._resolve_ref(ref_path)
        
        return content  # Detached HEAD (direct SHA)
    
    def set_head(self, ref_or_sha: str, symbolic: bool = True):
        """Set the HEAD reference."""
        head_path = os.path.join(self.root_path, "HEAD")
        with open(head_path, "w") as f:
            if symbolic:
                f.write(f"ref: {ref_or_sha}\n")
            else:
                f.write(f"{ref_or_sha}\n")
    
    def _resolve_ref(self, ref_path: str) -> Optional[str]:
        """Resolve a reference to a SHA."""
        full_path = os.path.join(self.root_path, ref_path)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                return f.read().strip()
        return None
    
    def update_ref(self, ref_path: str, sha: str):
        """Update a reference to point to a SHA."""
        full_path = os.path.join(self.root_path, ref_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(sha)
    
    def get_branch(self, name: str) -> Optional[str]:
        """Get the SHA a branch points to."""
        return self._resolve_ref(f"refs/heads/{name}")
    
    def set_branch(self, name: str, sha: str):
        """Update or create a branch."""
        self.update_ref(f"refs/heads/{name}", sha)
    
    def list_branches(self) -> Dict[str, str]:
        """List all branches and their SHAs."""
        branches = {}
        if os.path.exists(self.heads_path):
            for name in os.listdir(self.heads_path):
                sha = self.get_branch(name)
                if sha:
                    branches[name] = sha
        return branches
    
    def create_tag(self, name: str, sha: str):
        """Create a tag pointing to a commit."""
        self.update_ref(f"refs/tags/{name}", sha)
    
    def get_tag(self, name: str) -> Optional[str]:
        """Get the SHA a tag points to."""
        return self._resolve_ref(f"refs/tags/{name}")
    
    def list_tags(self) -> Dict[str, str]:
        """List all tags and their SHAs."""
        tags = {}
        if os.path.exists(self.tags_path):
            for name in os.listdir(self.tags_path):
                sha = self.get_tag(name)
                if sha:
                    tags[name] = sha
        return tags
    
    # ========== Agent-Specific Refs ==========
    
    def get_agent_head(self, agent_id: str) -> Optional[str]:
        """Get the current HEAD for an agent."""
        return self._resolve_ref(f"refs/agents/{agent_id}")
    
    def set_agent_head(self, agent_id: str, sha: str):
        """Set an agent's HEAD to a commit."""
        self.update_ref(f"refs/agents/{agent_id}", sha)
    
    def list_agent_refs(self) -> Dict[str, str]:
        """List all agent refs and their current commits."""
        agents = {}
        if os.path.exists(self.agents_path):
            for name in os.listdir(self.agents_path):
                sha = self.get_agent_head(name)
                if sha:
                    agents[name] = sha
        return agents


class MemoryDAG:
    """
    High-level interface for the Memory DAG.
    
    Provides Git-like operations:
    - add: Stage a memory
    - commit: Create immutable snapshot
    - checkout: Restore cognitive state
    - diff: Compare mental states
    - log: View history
    - branch/merge: Manage reasoning paths
    """
    
    def __init__(self, root_path: str = "./.gitmem"):
        self.store = ObjectStore(root_path)
        self.index: List[MemoryBlob] = []  # Staging area
        self._current_agent: str = "default"
    
    def set_agent(self, agent_id: str):
        """Set the current working agent."""
        self._current_agent = agent_id
    
    @property
    def current_agent(self) -> str:
        return self._current_agent
    
    # ========== Staging (Index) ==========
    
    def add(self, content: str, memory_type: str = "episodic", 
            importance: float = 0.5, tags: List[str] = None,
            keywords: List[str] = None, persons: List[str] = None,
            metadata: Dict[str, Any] = None) -> str:
        """Stage a memory for commit. Returns the blob SHA."""
        blob = MemoryBlob(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
            keywords=keywords or [],
            persons=persons or [],
            metadata=metadata or {}
        )
        
        # Store the blob immediately (objects are immutable)
        sha = self.store.store_blob(blob)
        
        # Add to staging
        self.index.append(blob)
        
        return sha
    
    def status(self) -> Dict[str, Any]:
        """Show staging area status."""
        return {
            "staged": len(self.index),
            "memories": [
                {"sha": b.sha[:8], "content": b.content[:50], "type": b.memory_type}
                for b in self.index
            ],
            "current_agent": self._current_agent,
            "head": self.store.get_agent_head(self._current_agent)
        }
    
    def reset(self):
        """Clear the staging area."""
        self.index = []
    
    # ========== Commit ==========
    
    def commit(self, message: str, author: str = None) -> str:
        """
        Commit staged memories to create an immutable cognitive snapshot.
        Returns the commit SHA.
        """
        if not self.index:
            raise ValueError("Nothing to commit (staging area empty)")
        
        # 1. Get parent commit and its tree
        parent_sha = self.store.get_agent_head(self._current_agent)
        parent_tree = None
        if parent_sha:
            parent_commit = self.store.get_commit(parent_sha)
            if parent_commit:
                parent_tree = self.store.get_tree(parent_commit.tree_sha)
        
        # 2. Build new tree (inherit from parent + add new)
        tree = CognitiveTree()
        
        # Copy parent entries
        if parent_tree:
            tree.entries = list(parent_tree.entries)
        
        # Add new staged blobs
        for blob in self.index:
            entry = TreeEntry(
                mode=blob.memory_type,
                sha=blob.sha,
                path=f"{blob.memory_type}/{blob.sha[:12]}",
                name=blob.content[:30].replace("\n", " ")
            )
            tree.add_entry(entry)
        
        # 3. Store tree
        tree_sha = self.store.store_tree(tree)
        
        # 4. Create commit
        commit = MemoryCommit(
            tree_sha=tree_sha,
            message=message,
            author=author or self._current_agent,
            agent_id=self._current_agent,
            parents=[parent_sha] if parent_sha else [],
            stats={"added": len(self.index), "total": len(tree.entries)}
        )
        
        commit_sha = self.store.store_commit(commit)
        
        # 5. Update agent HEAD
        self.store.set_agent_head(self._current_agent, commit_sha)
        
        # 6. Clear staging
        self.reset()
        
        return commit_sha
    
    # ========== History ==========
    
    def log(self, limit: int = 10) -> List[Dict]:
        """Get commit history for current agent."""
        history = []
        sha = self.store.get_agent_head(self._current_agent)
        
        while sha and len(history) < limit:
            commit = self.store.get_commit(sha)
            if not commit:
                break
            
            history.append({
                "sha": sha,
                "message": commit.message,
                "author": commit.author,
                "timestamp": commit.timestamp,
                "stats": commit.stats,
                "parents": commit.parents
            })
            
            # Walk to first parent
            sha = commit.parents[0] if commit.parents else None
        
        return history
    
    def show(self, sha: str) -> Dict:
        """Show details of a specific commit."""
        commit = self.store.get_commit(sha)
        if not commit:
            return {"error": "Commit not found"}
        
        tree = self.store.get_tree(commit.tree_sha)
        
        return {
            "sha": sha,
            "message": commit.message,
            "author": commit.author,
            "agent_id": commit.agent_id,
            "timestamp": commit.timestamp,
            "parents": commit.parents,
            "stats": commit.stats,
            "tree": {
                "sha": commit.tree_sha,
                "entries": [e.to_dict() for e in tree.entries] if tree else []
            }
        }
    
    # ========== Diff ==========
    
    def diff(self, sha_a: str, sha_b: str) -> Dict:
        """
        Compute diff between two commits.
        Returns added, removed, and modified memories.
        """
        commit_a = self.store.get_commit(sha_a) if sha_a else None
        commit_b = self.store.get_commit(sha_b) if sha_b else None
        
        tree_a = self.store.get_tree(commit_a.tree_sha) if commit_a else CognitiveTree()
        tree_b = self.store.get_tree(commit_b.tree_sha) if commit_b else CognitiveTree()
        
        # Build sets of memory SHAs
        shas_a = {e.sha for e in tree_a.entries}
        shas_b = {e.sha for e in tree_b.entries}
        
        added = shas_b - shas_a
        removed = shas_a - shas_b
        
        # Get blob details
        added_blobs = []
        for sha in added:
            blob = self.store.get_blob(sha)
            if blob:
                added_blobs.append({
                    "sha": sha,
                    "content": blob.content,
                    "type": blob.memory_type,
                    "importance": blob.importance
                })
        
        removed_blobs = []
        for sha in removed:
            blob = self.store.get_blob(sha)
            if blob:
                removed_blobs.append({
                    "sha": sha,
                    "content": blob.content,
                    "type": blob.memory_type,
                    "importance": blob.importance
                })
        
        return {
            "from": sha_a,
            "to": sha_b,
            "summary": {
                "added": len(added),
                "removed": len(removed),
                "total_a": len(shas_a),
                "total_b": len(shas_b)
            },
            "added": added_blobs,
            "removed": removed_blobs
        }
    
    # ========== Checkout ==========
    
    def checkout(self, sha: str) -> Dict:
        """
        Checkout a specific commit (restore cognitive state).
        Returns the tree entries at that commit.
        """
        commit = self.store.get_commit(sha)
        if not commit:
            return {"error": "Commit not found"}
        
        # Update agent HEAD (detached HEAD state)
        self.store.set_agent_head(self._current_agent, sha)
        
        # Get tree
        tree = self.store.get_tree(commit.tree_sha)
        
        return {
            "checked_out": sha,
            "message": commit.message,
            "entries": len(tree.entries) if tree else 0,
            "timestamp": commit.timestamp
        }
    
    # ========== Branching ==========
    
    def branch(self, name: str, from_sha: str = None) -> str:
        """Create a new branch."""
        if from_sha is None:
            from_sha = self.store.get_agent_head(self._current_agent)
        
        if not from_sha:
            raise ValueError("No commit to branch from")
        
        self.store.set_branch(name, from_sha)
        return from_sha
    
    def checkout_branch(self, name: str) -> Dict:
        """Switch to a branch."""
        sha = self.store.get_branch(name)
        if not sha:
            return {"error": f"Branch '{name}' not found"}
        
        return self.checkout(sha)
    
    def list_branches(self) -> Dict[str, str]:
        """List all branches."""
        return self.store.list_branches()
    
    # ========== Tags ==========
    
    def tag(self, name: str, sha: str = None, message: str = None) -> str:
        """Create a tag (named snapshot)."""
        if sha is None:
            sha = self.store.get_agent_head(self._current_agent)
        
        if not sha:
            raise ValueError("No commit to tag")
        
        self.store.create_tag(name, sha)
        return sha
    
    def list_tags(self) -> Dict[str, str]:
        """List all tags."""
        return self.store.list_tags()
    
    # ========== Export ==========
    
    def export_state(self, sha: str = None) -> Dict:
        """
        Export the complete cognitive state at a commit.
        Returns all memories with their content.
        """
        if sha is None:
            sha = self.store.get_agent_head(self._current_agent)
        
        if not sha:
            return {"error": "No commits yet"}
        
        commit = self.store.get_commit(sha)
        if not commit:
            return {"error": "Commit not found"}
        
        tree = self.store.get_tree(commit.tree_sha)
        if not tree:
            return {"error": "Tree not found"}
        
        memories = []
        for entry in tree.entries:
            blob = self.store.get_blob(entry.sha)
            if blob:
                memories.append({
                    "sha": entry.sha,
                    "path": entry.path,
                    "type": blob.memory_type,
                    "content": blob.content,
                    "importance": blob.importance,
                    "tags": blob.tags,
                    "keywords": blob.keywords,
                    "persons": blob.persons,
                    "created_at": blob.created_at
                })
        
        return {
            "commit": sha,
            "message": commit.message,
            "timestamp": commit.timestamp,
            "agent": commit.agent_id,
            "memory_count": len(memories),
            "memories": memories
        }
    
    # ========== Search ==========
    
    def search(self, query: str, memory_type: str = None, limit: int = 10) -> List[Dict]:
        """
        Simple text-based search across all memories.
        For semantic search, use the vector_engine.
        """
        sha = self.store.get_agent_head(self._current_agent)
        if not sha:
            return []
        
        commit = self.store.get_commit(sha)
        if not commit:
            return []
        
        tree = self.store.get_tree(commit.tree_sha)
        if not tree:
            return []
        
        query_lower = query.lower()
        results = []
        
        for entry in tree.entries:
            # Filter by type if specified
            if memory_type and entry.mode != memory_type:
                continue
                
            blob = self.store.get_blob(entry.sha)
            if not blob:
                continue
            
            # Simple text matching
            content_lower = blob.content.lower()
            if query_lower in content_lower:
                score = content_lower.count(query_lower) / len(content_lower)
                results.append({
                    "sha": entry.sha,
                    "content": blob.content,
                    "type": blob.memory_type,
                    "importance": blob.importance,
                    "tags": blob.tags,
                    "keywords": blob.keywords,
                    "score": score
                })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
