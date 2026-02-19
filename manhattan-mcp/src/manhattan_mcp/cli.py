"""
CLI entry point for Manhattan MCP Server.

Provides the `manhattan-mcp` command for starting the MCP server
and setting up client-specific rules files.
"""

import argparse
import os
import sys
from pathlib import Path

from manhattan_mcp import __version__


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="manhattan-mcp",
        description="Manhattan MCP Server - AI Memory for Claude Desktop, Cursor, and more"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"manhattan-mcp {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser(
        "start",
        help="Start the MCP server (default if no command given)"
    )
    start_parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode (default: stdio)"
    )
    
    # Setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Generate rules files so your AI client auto-uses Manhattan MCP tools"
    )
    setup_parser.add_argument(
        "client",
        choices=["cursor", "claude", "gemini", "copilot", "windsurf", "all"],
        help="Which client to set up (or 'all' for all clients)"
    )
    setup_parser.add_argument(
        "--dir",
        default=".",
        help="Project directory to create rules files in (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Default to 'start' if no command given
    if args.command is None:
        args.command = "start"
        args.transport = "stdio"
    
    if args.command == "start":
        start_server(args.transport)
    elif args.command == "setup":
        setup_client(args.client, args.dir)


def start_server(transport: str = "stdio"):
    """Start the MCP server."""
    from manhattan_mcp.server import mcp
    
    print(f"🧠 Starting Manhattan MCP Server v{__version__} (local mode)", file=sys.stderr)
    print(f"🚀 Transport: {transport}", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Start the server
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse")


def setup_client(client: str, project_dir: str = "."):
    """Generate rules files for a specific client (or all clients)."""
    from manhattan_mcp.rules_templates import CLIENT_RULES, SUPPORTED_CLIENTS
    
    project_path = Path(project_dir).resolve()
    
    if not project_path.is_dir():
        print(f"❌ Directory not found: {project_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine which clients to set up
    if client == "all":
        clients = SUPPORTED_CLIENTS
    else:
        clients = [client]
    
    print(f"🔧 Setting up Manhattan MCP for: {', '.join(clients)}", file=sys.stderr)
    print(f"📁 Project directory: {project_path}", file=sys.stderr)
    print("", file=sys.stderr)
    
    for client_name in clients:
        _create_rules_file(client_name, project_path)
    
    print("", file=sys.stderr)
    print("✅ Setup complete! Your AI client will now auto-use Manhattan MCP tools.", file=sys.stderr)
    print("💡 Make sure manhattan-mcp is configured as an MCP server in your client.", file=sys.stderr)


def _create_rules_file(client_name: str, project_path: Path):
    """Create or append a rules file for a specific client."""
    from manhattan_mcp.rules_templates import CLIENT_RULES
    
    config = CLIENT_RULES[client_name]
    file_path = project_path / config["path"]
    content = config["content"]
    description = config["description"]
    mode = config["mode"]
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if mode == "create":
        # Create (or overwrite) the file
        file_path.write_text(content, encoding="utf-8")
        print(f"  ✅ Created {description}", file=sys.stderr)
        print(f"     → {file_path}", file=sys.stderr)
    
    elif mode == "append":
        # Check if content already exists
        if file_path.exists():
            existing = file_path.read_text(encoding="utf-8")
            if "Manhattan MCP" in existing:
                print(f"  ⏭️  Skipped {description} (already configured)", file=sys.stderr)
                return
            # Append with separator
            with open(file_path, "a", encoding="utf-8") as f:
                f.write("\n" + content)
            print(f"  ✅ Appended to {description}", file=sys.stderr)
            print(f"     → {file_path}", file=sys.stderr)
        else:
            # Create new file
            file_path.write_text(content.lstrip("\n"), encoding="utf-8")
            print(f"  ✅ Created {description}", file=sys.stderr)
            print(f"     → {file_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
