"""
GitMem Local - Data Models

Core data models for the local AI context storage system.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid
import json
import hashlib


class MemoryType(str, Enum):
    """Types of memory for AI context."""
    EPISODIC = "episodic"       # Conversation history, events
    SEMANTIC = "semantic"       # Facts, knowledge
    PROCEDURAL = "procedural"   # Skills, how-to
    WORKING = "working"         # Short-term, session context
    STATE = "state"             # Agent state snapshots


class MemoryScope(str, Enum):
    """Visibility scope for memories."""
    PRIVATE = "private"   # Only this agent
    SHARED = "shared"     # Shared with linked agents
    GLOBAL = "global"     # All agents


class ObjectType(str, Enum):
    """Git-like object types."""
    BLOB = "blob"       # Raw memory content
    TREE = "tree"       # Directory of memories
    COMMIT = "commit"   # Snapshot with parent links


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    lossless_restatement: str = ""  # Clear, factual restatement
    memory_type: str = MemoryType.EPISODIC.value
    importance: float = 0.5
    
    # Metadata
    agent_id: str = ""
    keywords: List[str] = field(default_factory=list)
    persons: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    location: str = ""
    topic: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Governance
    scope: str = MemoryScope.PRIVATE.value
    provenance: str = ""
    
    # Versioning
    commit_hash: Optional[str] = None
    parent_hash: Optional[str] = None
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Embedding (optional, for vector search)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        data = asdict(self)
        # Remove embedding from serialization to save space
        data.pop('embedding', None)
        return {k: v for k, v in data.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @property
    def sha(self) -> str:
        """Compute content hash."""
        content = json.dumps({
            'content': self.content or self.lossless_restatement,
            'agent_id': self.agent_id,
            'memory_type': self.memory_type
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class Commit:
    """Immutable snapshot commit."""
    hash: str = ""
    parent_hash: Optional[str] = None
    agent_id: str = ""
    author_id: str = "system"
    message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Snapshot state
    tree_hash: str = ""
    memory_snapshot: List[str] = field(default_factory=list)  # List of memory IDs
    
    # Stats
    stats: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Commit':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Checkpoint:
    """Agent state checkpoint."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    checkpoint_type: str = "snapshot"  # snapshot, session, recovery, auto
    name: str = ""
    description: str = ""
    commit_hash: str = ""
    parent_checkpoint_id: Optional[str] = None
    memory_counts: Dict[str, int] = field(default_factory=dict)
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ActivityLog:
    """Activity log entry."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    log_type: str = "access"  # access, mutation, error, system
    action: str = ""  # read, write, delete, create
    resource_type: str = ""  # memory, document, checkpoint
    resource_id: str = ""
    actor_type: str = ""  # user, agent, system
    actor_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentContext:
    """Current context state for an agent."""
    agent_id: str = ""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_head: Optional[str] = None
    working_memory: List[MemoryEntry] = field(default_factory=list)
    recent_queries: List[str] = field(default_factory=list)
    conversation_summary: str = ""
    key_points: List[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['working_memory'] = [m.to_dict() if isinstance(m, MemoryEntry) else m for m in self.working_memory]
        return data
