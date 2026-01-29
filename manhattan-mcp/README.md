# Manhattan MCP

**Give AI Agents Persistent Memory** - MCP Server for the Manhattan Memory System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Manhattan MCP is a local [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that connects AI agents (Claude Desktop, Cursor, Windsurf, etc.) to the **Manhattan Memory System** - a cloud-based persistent memory for AI assistants.

## Features

- 🧠 **Persistent Memory** - Store and retrieve information across conversations
- 🔍 **Semantic Search** - Find relevant memories using natural language queries
- 🤖 **AI-Generated Answers** - Get comprehensive answers using memory context
- 👤 **Multi-Agent Support** - Create separate memory spaces for different use cases
- 📊 **Analytics** - Track memory usage and agent statistics
- 💾 **Export/Import** - Backup and restore memory data

## Installation

```bash
pip install manhattan-mcp
```

## Quick Start

### 1. Get Your API Key

Sign up at [https://themanhattanproject.ai](https://themanhattanproject.ai) to get your API key.

### 2. Set Environment Variable

```bash
export MANHATTAN_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```env
MANHATTAN_API_KEY=your-api-key-here
```

### 3. Configure Your AI Client

#### Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

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

#### Cursor

Add to your Cursor MCP settings (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "manhattan": {
      "command": "manhattan-mcp"
    }
  }
}
```

#### Windsurf

Add to your Windsurf configuration:

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

### 4. Start Using Memory!

Once configured, your AI agent will have access to 35+ memory tools:

- **`search_memory`** - Search for relevant memories
- **`add_memory_direct`** - Store new information
- **`get_context_answer`** - Get AI-generated answers with memory context
- **`session_start`** / **`session_end`** - Manage conversation sessions
- And many more!

## Available Tools

### Memory Operations
| Tool | Description |
|------|-------------|
| `search_memory` | Search memories using natural language |
| `add_memory_direct` | Store structured memories |
| `get_context_answer` | Get AI answers using memory context |
| `update_memory_entry` | Update existing memories |
| `delete_memory_entries` | Delete specific memories |

### Agent Management
| Tool | Description |
|------|-------------|
| `create_agent` | Create a new memory agent |
| `list_agents` | List all your agents |
| `get_agent` | Get agent details |
| `update_agent` | Update agent configuration |
| `delete_agent` | Permanently delete an agent |

### Session Management
| Tool | Description |
|------|-------------|
| `session_start` | Initialize a conversation |
| `session_end` | End session and sync memories |
| `pull_context` | Load relevant context |
| `push_memories` | Sync pending memories |

### AI Helpers
| Tool | Description |
|------|-------------|
| `auto_remember` | Automatically extract facts from messages |
| `should_remember` | Check if info is worth storing |
| `what_do_i_know` | Summary of known user info |

## Configuration Options

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `MANHATTAN_API_KEY` | Your API key (required) | - |
| `MANHATTAN_API_URL` | API endpoint URL | `https://themanhattanproject.ai/mcp` |
| `MANHATTAN_AGENT_ID` | Default agent ID | Enterprise default |
| `MANHATTAN_TIMEOUT` | Request timeout (seconds) | `120` |

## CLI Commands

```bash
# Start the MCP server (default)
manhattan-mcp start

# Show version
manhattan-mcp --version

# Show help
manhattan-mcp --help
```

## Example Usage

Once your AI agent is connected, it can use memory like this:

**Storing information:**
```
User: My name is Sarah and I prefer Python over JavaScript.
AI: *calls add_memory_direct to store this preference*
    Nice to meet you, Sarah! I've noted your preference for Python.
```

**Retrieving context:**
```
User: What programming language should I use for this project?
AI: *calls search_memory to find preferences*
    Based on your preference for Python, I'd recommend using it for this project!
```

## Development

```bash
# Clone the repository
git clone https://github.com/agent-architects/manhattan-mcp
cd manhattan-mcp

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- 🌐 [Website](https://themanhattanproject.ai)
- 📖 [Documentation](https://themanhattanproject.ai/mcp-docs)
- 🐛 [Issues](https://github.com/agent-architects/manhattan-mcp/issues)
- 💬 [Discord](https://discord.gg/manhattan)
