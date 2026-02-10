# GitMem Coding - Walkthrough

This module enables **efficient cross-session coding context retrieval**, saving tokens by caching file contents locally instead of re-reading them every time.

## 1. Concept: Token Reduction Loop

1.  **Read & Store**: When you first read a file, call `store_file_context`.
2.  **Next Session**: Before reading, call `retrieve_file_context`.
3.  **Check Freshness**:
    *   If `freshness: "fresh"` → USE CACHED CONTENT (Free! No tokens used to read file)
    *   If `freshness: "stale"` → Re-read file from disk (File changed)

## 2. Typical Workflow

### A. Storing Context (Session 1)

When you read a file `server.py`, store it:

```python
# MCP Tool: store_file_context
store_file_context(
    agent_id="my-agent",
    file_path="/path/to/server.py",
    content="...file content...",
    language="python",
    keywords=["server", "mcp", "api"]
)
```

### B. Retrieving Context (Session 2)

Next time you need `server.py`:

```python
# MCP Tool: retrieve_file_context
result = retrieve_file_context(
    agent_id="my-agent",
    file_path="/path/to/server.py"
)

if result["freshness"] == "fresh":
    # Use result["content"] directly!
    print(f"Loaded from cache! Saved {result['token_savings']} tokens.")
else:
    # File changed or missing, read from disk
    content = open("/path/to/server.py").read()
```

### C. Searching Context

Find relevant files you've worked on before:

```python
# MCP Tool: search_coding_context
search_coding_context(
    agent_id="my-agent",
    query="authentication logic"
)
```

### D. Virtual File System (VFS)

Navigate cached context like a file system:

```python
# MCP Tool: coding_list_dir
coding_list_dir(agent_id="my-agent", path="files/python")
# Returns: ["file1.py", "server.py", ...]

# MCP Tool: coding_read_file
coding_read_file(agent_id="my-agent", path="files/python/server.py")
```

### E. Version Control (.gitmem)

Create immutable snapshots of your coding context:

```python
# MCP Tool: coding_commit_snapshot
coding_commit_snapshot(
    agent_id="my-agent",
    message="Completed initial implementation of auth module"
)
# Returns commit SHA
```

Retrieve historical snapshots:

```python
coding_list_dir(agent_id="my-agent", path="snapshots")
coding_read_file(agent_id="my-agent", path="snapshots/{sha}.json")
```

## 3. Storage Structure

Data is stored in `.gitmem_coding/`:

*   `agents/{agent_id}/file_contexts.json`: Cached content & metadata
*   `agents/{agent_id}/coding_sessions.json`: Usage logs & token savings
*   `.gitmem/`: Git-like object store for snapshots

## 4. Stats & Savings

Check how many tokens you've saved:

```python
# MCP Tool: coding_context_stats
coding_context_stats(agent_id="my-agent")
```
