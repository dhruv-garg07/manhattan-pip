"""
GitMem Coding - Data Models

Core data models for coding context storage and retrieval.
Designed for cross-session file content caching to reduce token usage.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid
import json
import hashlib


class FileLanguage(str, Enum):
    """Supported programming languages for context tagging."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    SHELL = "shell"
    YAML = "yaml"
    JSON_LANG = "json"
    TOML = "toml"
    MARKDOWN = "markdown"
    XML = "xml"
    OTHER = "other"


class ContextStatus(str, Enum):
    """Status of a cached file context."""
    FRESH = "fresh"         # Content matches current file on disk
    STALE = "stale"         # File has been modified since caching
    MISSING = "missing"     # Original file no longer exists
    UNKNOWN = "unknown"     # Unable to determine (e.g., remote file)


# Approximate tokens per character ratio (GPT-style tokenization)
TOKENS_PER_CHAR_RATIO = 0.25


@dataclass
class FileContext:
    """
    A cached file context entry.
    
    Stores the content of a file read by an agent, along with metadata
    for freshness detection and token savings tracking.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # File identification
    file_path: str = ""             # Absolute path to the file
    file_name: str = ""             # Just the filename (basename)
    relative_path: str = ""         # Relative path (workspace-relative)
    
    # Content
    content: str = ""               # Full file content (empty when storage_mode='skeleton')
    content_hash: str = ""          # SHA-256 of ORIGINAL content for freshness checks
    content_summary: str = ""       # Optional brief summary of the file
    compact_skeleton: str = ""      # AST-generated skeleton (signatures, docstrings, structure)
    
    # Storage mode
    storage_mode: str = "skeleton"  # 'skeleton' (compact AST) or 'full' (raw content)
    
    # File metadata
    language: str = FileLanguage.OTHER.value
    line_count: int = 0
    size_bytes: int = 0
    
    # Ownership
    agent_id: str = ""
    session_id: str = ""            # Session in which file was first cached
    
    # Usage tracking
    access_count: int = 1           # Number of times retrieved from cache
    token_estimate: int = 0         # Estimated tokens for stored representation
    original_token_estimate: int = 0  # Tokens the full file would have consumed
    skeleton_token_estimate: int = 0  # Tokens the skeleton consumes
    compression_ratio: float = 0.0  # 1 - (skeleton_tokens / original_tokens)
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    file_modified_at: str = ""      # Last known modification time of the source file
    
    # Tags for searchability
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Chunking references
    chunk_hashes: List[str] = field(default_factory=list)  # IDs of semantic chunks in this file
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding content for listings."""
        return asdict(self)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary WITHOUT full content (for listings/search)."""
        data = asdict(self)
        data.pop("content", None)
        data["has_content"] = bool(self.content)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileContext':
        """Create from dictionary."""
        valid_fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in valid_fields})
    
    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def estimate_tokens(content: str) -> int:
        """Estimate token count for content (approximate)."""
        return int(len(content) * TOKENS_PER_CHAR_RATIO)
    
    def refresh_metadata(self, original_content: str = ""):
        """Recalculate derived metadata from content.
        
        Args:
            original_content: The full source content (used only to compute
                hash and original token estimate when in skeleton mode).
        """
        # Use original content for hash/size if provided, else stored content
        source = original_content or self.content
        if source:
            self.content_hash = self.compute_hash(source)
            self.size_bytes = len(source.encode('utf-8'))
            self.line_count = source.count('\n') + 1
            self.original_token_estimate = self.estimate_tokens(source)
        
        if self.compact_skeleton:
            self.skeleton_token_estimate = self.estimate_tokens(self.compact_skeleton)
            self.token_estimate = self.skeleton_token_estimate
            if self.original_token_estimate > 0:
                self.compression_ratio = round(
                    1.0 - (self.skeleton_token_estimate / self.original_token_estimate), 4
                )
        elif self.content:
            self.token_estimate = self.estimate_tokens(self.content)
            self.original_token_estimate = self.token_estimate


@dataclass
class CodingSession:
    """
    A coding session record tracking which files were read.
    
    Used for analytics and token savings estimation.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    
    # Files tracked in this session
    files_stored: List[str] = field(default_factory=list)     # File paths stored
    files_retrieved: List[str] = field(default_factory=list)  # File paths retrieved from cache
    
    # Token savings
    tokens_stored: int = 0          # Total tokens worth of content stored
    tokens_retrieved: int = 0       # Total tokens served from cache (savings)
    cache_hits: int = 0             # Number of successful cache retrievals
    cache_misses: int = 0           # Number of cache misses
    
    # Timestamps
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodingSession':
        """Create from dictionary."""
        valid_fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in valid_fields})
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return round((self.cache_hits / total) * 100, 2)


@dataclass
class CodeChunk:
    """
    A semantic unit of code (Function, Class, or Block).
    Used for granular retrieval and deduplication.
    """
    hash_id: str                    # SHA-256 of normalized tokens
    content: str                    # The actual code text
    type: str                       # function, class, import, code, etc.
    name: str                       # Name of the entity (e.g. function name)
    start_line: int                 # Start line in original file
    end_line: int                   # End line in original file
    
    # Metadata
    language: str = "text"
    embedding_id: Optional[str] = None  # Reference to vector embedding
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkRegistry:
    """
    Global registry mapping Hashes to CodeChunks.
    This acts as the deduplication layer.
    """
    chunks: Dict[str, CodeChunk] = field(default_factory=dict)
    
    def add_chunk(self, chunk: CodeChunk) -> bool:
        """Add chunk if not exists. Returns True if new."""
        if chunk.hash_id not in self.chunks:
            self.chunks[chunk.hash_id] = chunk
            return True
        return False
    
    def get_chunk(self, hash_id: str) -> Optional[CodeChunk]:
        return self.chunks.get(hash_id)

