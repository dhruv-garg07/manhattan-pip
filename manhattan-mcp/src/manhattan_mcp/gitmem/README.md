# GitMem Local - Local File System for AI Context Storage

A GitHub-like version-controlled memory system for AI agents, running entirely locally without any web APIs.

## Features

- **Local-First**: All data stored in JSON files on your local file system
- **Git-Like Version Control**: Commits, branches, tags, rollback
- **Virtual File System**: Navigate memories like a file explorer
- **Smart Context**: Auto-remember, search, and context retrieval
- **No Dependencies**: No external databases or APIs required

## Quick Start

```python
from gitmem.api import LocalAPI

# Initialize
api = LocalAPI()

# Start a session
api.session_start("my-agent")

# Add memories
api.add_memory("my-agent", [{
    "lossless_restatement": "User prefers Python over JavaScript",
    "keywords": ["preference", "Python", "JavaScript", "programming"],
    "topic": "preferences"
}])

# Search memories
results = api.search_memory("my-agent", "Python")
print(results)

# End session
api.session_end("my-agent", 
    conversation_summary="Discussed programming preferences",
    key_points=["User prefers Python"]
)
```

## Architecture

```
.gitmem_data/
├── agents/
│   └── {agent_id}/
│       ├── memories.json
│       ├── documents.json
│       ├── checkpoints.json
│       ├── logs.json
│       └── settings.json
├── .gitmem/
│   ├── objects/          # Content-addressable storage
│   │   └── ab/cd1234...  # Compressed blobs
│   ├── refs/
│   │   ├── heads/        # Branches
│   │   ├── tags/         # Named snapshots
│   │   └── agents/       # Agent HEADs
│   └── HEAD
├── index/
└── config.json
```

## Core Components

### LocalAPI (`gitmem/api.py`)
Main interface for all operations:
- Session management
- Memory CRUD
- File system navigation
- Version control

### ContextManager (`gitmem/context_manager.py`)
High-level AI context operations:
- Auto-remembering
- Smart retrieval
- Pre-response checks
- Session checkpoints

### LocalMemoryStore (`gitmem/memory_store.py`)
JSON-based storage backend:
- Thread-safe operations
- Automatic persistence
- Export/Import

### ObjectStore & MemoryDAG (`gitmem/object_store.py`)
Git-like version control:
- Content-addressable storage
- Commit history
- Branching and tagging
- Rollback capabilities

### LocalFileSystem (`gitmem/file_system.py`)
Virtual file system abstraction:
- Folder structure for memories
- Read/Write operations
- Permission management

## API Reference

### Session Management

```python
# Start session
api.session_start(agent_id, auto_pull_context=True)

# End session with summary
api.session_end(agent_id, 
    conversation_summary="...",
    key_points=["..."]
)

# Create checkpoint during session
api.checkpoint(agent_id, "Summary...", ["key point 1"])
```

### Memory Operations

```python
# Add memory
api.add_memory(agent_id, [{
    "lossless_restatement": "Clear statement",
    "keywords": ["keyword1", "keyword2"],
    "persons": ["name"],
    "topic": "category",
    "memory_type": "semantic"  # or episodic, procedural, working
}])

# Search memories
results = api.search_memory(agent_id, "query", top_k=5)

# Get specific memory
memory = api.get_memory(agent_id, memory_id)

# Update memory
api.update_memory(agent_id, memory_id, {"topic": "new_topic"})

# Delete memory
api.delete_memory(agent_id, memory_id)

# List all memories
memories = api.list_memories(agent_id, memory_type="semantic")
```

### Auto-Remember

```python
# Automatically extract and store facts
result = api.auto_remember(agent_id, "My name is John and I love Python")

# Check if message should be remembered
should = api.should_remember("I prefer dark mode")
```

### File System

```python
# List directory
nodes = api.list_dir(agent_id, "context/episodic")

# Read file
content = api.read_file(agent_id, "context/episodic/2024-01-01_abc123.json")

# Write file
api.write_file(agent_id, "documents/knowledge/notes.md", "# Notes...")

# Delete file
api.delete_file(agent_id, "documents/knowledge/notes.md")
```

### Version Control

```python
# Commit current state
commit_sha = api.commit(agent_id, "Added user preferences")

# View history
history = api.history(agent_id, limit=10)

# Rollback to previous commit
api.rollback(agent_id, commit_sha)

# Compare commits
diff = api.diff(agent_id, sha_a, sha_b)

# Create branch
api.branch(agent_id, "experiment-1")

# Create tag
api.tag(agent_id, "v1.0")
```

### Export/Import

```python
# Export all memories
backup = api.export_memories(agent_id)

# Save to file
with open("backup.json", "w") as f:
    json.dump(backup, f)

# Import from backup
with open("backup.json") as f:
    data = json.load(f)
api.import_memories(agent_id, data, merge_mode="append")
```

## Virtual Folder Structure

```
/
├── context/
│   ├── episodic/     # Conversation history, events
│   ├── semantic/      # Facts, knowledge
│   ├── procedural/    # Skills, how-to
│   └── working/       # Short-term, session context
├── documents/
│   ├── knowledge/     # User documents
│   └── references/    # External references
├── checkpoints/
│   ├── snapshots/     # Manual checkpoints
│   └── sessions/      # Session checkpoints
├── logs/
│   ├── activity/      # Activity logs
│   └── errors/        # Error logs
└── agents/
    └── {agent_id}/    # Per-agent view
```

## Memory Types

| Type | Description | Use Case |
|------|-------------|----------|
| `episodic` | Conversation history, events | "User asked about X yesterday" |
| `semantic` | Facts, knowledge | "User's name is John" |
| `procedural` | Skills, how-to | "To deploy, run npm build" |
| `working` | Short-term context | Current session state |

## Thread Safety

The LocalMemoryStore uses `threading.RLock` for thread-safe operations. Multiple threads can safely read and write to the store.

## License

MIT License
