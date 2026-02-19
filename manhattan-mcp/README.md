# Manhattan MCP (GitMem)

**Token-Efficient Codebase Navigation** - MCP Server for the Manhattan Memory System.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Manhattan MCP is a local [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that provides AI agents (Claude Desktop, Cursor, Windsurf, etc.) with a **Virtual File System (VFS)** backed by compressed, cached code context. It allows agents to understand large codebases while saving 50-80% on tokens.

## Features

- 🏗️ **GitMem Context** - Compressed semantic code skeletons (signatures, summaries, hierarchies).
- 🔍 **Hybrid Search** - Semantic and keyword search across the entire codebase.
- 📂 **VFS Navigation** - Browse and read files via token-efficient outlines and contexts.
- 📝 **Auto-Indexing** - Automatically keeps the code index fresh after edits.
- 📊 **Token Analytics** - Track token savings and repository indexing status.

## Installation

```bash
pip install manhattan-mcp
```

## Quick Start

### 1. Install Manhattan MCP

```bash
pip install manhattan-mcp
```

### 2. Configure Your AI Client

Setting up Manhattan MCP is a two-step process:

#### Step A: Register the Server (One-time)
Add Manhattan MCP to your AI tool's global settings.

**Claude Desktop**
Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "manhattan": {
      "command": "manhattan-mcp",
      "args": ["start"]
    }
  }
}
```

**Cursor**
Add to Cursor Settings > MCP:
- Name: `manhattan`
- Type: `command`
- Command: `manhattan-mcp`

**GitHub Copilot (VS Code)**
Add to your Copilot MCP settings:
```json
{
  "servers": {
    "manhattan": {
      "command": "manhattan-mcp",
      "args": ["start"]
    }
  }
}
```

#### Step B: Apply Project Rules (Per Project)
Run the setup command in your project root to ensure the agent follows the mandatory indexing policy.

```bash
# For Cursor
manhattan-mcp setup cursor

# For Claude
manhattan-mcp setup claude

# For Gemini (Antigravity)
manhattan-mcp setup gemini

# For GitHub Copilot
manhattan-mcp setup copilot

# For Windsurf
manhattan-mcp setup windsurf

# For all supported clients
manhattan-mcp setup all
```

### 3. Start Navigating!

Once configured, your AI agent can use Manhattan MCP to understand your codebase efficiently.

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
| `MANHATTAN_MEM_PATH` | Storage path for memory/index | `~/.manhattan-mcp/data` |
| `MANHATTAN_TIMEOUT` | Request timeout (seconds) | `120` |

## CLI Commands

```bash
# Start the MCP server (default)
manhattan-mcp start

# Set up client rules (Cursor, Claude, etc.)
manhattan-mcp setup [client]

# Show version
manhattan-mcp --version

# Show help
manhattan-mcp --help
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- 🌐 [Website](https://themanhattanproject.ai)
- 📖 [Documentation](https://themanhattanproject.ai/mcp-docs)
- 🐛 [Issues](https://github.com/agent-architects/manhattan-mcp/issues)
- 💬 [Discord](https://discord.gg/manhattan)
