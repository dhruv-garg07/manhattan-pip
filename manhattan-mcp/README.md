# GitMem MCP (Manhattan)

**Token-Efficient Codebase Navigation** - MCP Server for the Manhattan Memory System.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GitMem MCP is a local [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that provides AI agents (Claude Desktop, Cursor, Windsurf, etc.) with a **Virtual File System (VFS)** backed by compressed, cached code context. It allows agents to understand large codebases while saving 50-80% on tokens.

## Features

- 🏗️ **GitMem Context** - Compressed semantic code skeletons (signatures, summaries, hierarchies).
- 🔍 **Hybrid Search** - Semantic and keyword search across the entire codebase.
- 📂 **VFS Navigation** - Browse and read files via token-efficient outlines and contexts.
- 📝 **Auto-Indexing** - Automatically keeps the code index fresh after edits.
- 📊 **Token Analytics** - Track token savings and repository indexing status.

## Installation

```bash
pip install gitmem-mcp
```

## Quick Start

### 1. Install GitMem MCP

```bash
pip install gitmem-mcp
```

### 2. Configure Your AI Client

Setting up GitMem MCP is a two-step process:

#### Step A: Register the Server (One-time)
Add GitMem MCP to your AI tool's global settings.

**Claude Desktop**
Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "gitmem_mcp": {
      "command": "gitmem",
      "args": ["start"],
      "env": {
        "MANHATTAN_API_URL": "https://www.themanhattanproject.ai"
      }
    }
  }
}
```

**Cursor**
Add to Cursor Settings > MCP:
- Name: `gitmem_mcp`
- Type: `command`
- Command: `gitmem start`
- Env:
  - `MANHATTAN_API_URL`: `https://www.themanhattanproject.ai`

**VS Code / GitHub Copilot**
Add to `mcp.json` or your MCP settings:
```json
{
  "servers": {
    "gitmem_mcp": {
      "command": "gitmem",
      "args": ["start"],
      "env": {
        "MANHATTAN_API_URL": "https://www.themanhattanproject.ai"
      }
    }
  }
}
```

#### Step B: Apply Project Rules (Per Project)
Run the setup command in your project root to ensure the agent follows the mandatory indexing policy.

```bash
# For Cursor
gitmem setup cursor

# For Claude
gitmem setup claude

# For Gemini (Antigravity)
gitmem setup gemini

# For GitHub Copilot
gitmem setup copilot

# For Windsurf
gitmem setup windsurf

# For all supported clients
gitmem setup all
```

### 3. Start Navigating!

Once configured, your AI agent can use GitMem MCP to understand your codebase efficiently.

#### Example Usage

**Searching for code:**
```
User: How does the authentication flow work?
AI: *calls search_codebase "authentication flow"*
    I found the authentication logic in `auth.py`. 
    Let me read the context for you...
```

**Understanding a file:**
```
User: Summarize the main functions in server.py
AI: *calls get_file_outline "src/server.py"*
    The server.py file contains:
    - `start_server()`: Initializes the FastMCP instance...
    - `api_usage()`: Returns usage statistics...
```

**Saving tokens:**
```
User: Read the implementation of the memory builder.
AI: *calls read_file_context "src/builder.py"*
    (Returns a compressed semantic skeleton, saving 70% tokens)
    The memory builder uses a two-phase ingestion process...
```

## Configuration Options

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `MANHATTAN_API_KEY` | Your API key (if using cloud embeddings) | - |
| `MANHATTAN_API_URL` | Custom API URL (optional) | Gradio Endpoint |
| `MANHATTAN_MEM_PATH` | Storage path for memory/index | `~/.gitmem-mcp/data` |
| `MANHATTAN_TIMEOUT` | Request timeout (seconds) | `120` |

## CLI Commands

```bash
# Start the MCP server (default)
gitmem start

# Set up client rules (Cursor, Claude, etc.)
gitmem setup [client]

# Show version
gitmem --version

# Show help
gitmem --help
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- 🌐 [Website](https://themanhattanproject.ai)
- 📖 [Documentation](https://themanhattanproject.ai/mcp-docs)
- 🐛 [Issues](https://github.com/agent-architects/gitmem-mcp/issues)
- 💬 [Discord](https://discord.gg/manhattan)
