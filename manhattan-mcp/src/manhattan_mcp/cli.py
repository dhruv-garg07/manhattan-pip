"""
CLI entry point for Manhattan MCP Server.

Provides the `manhattan-mcp` command for starting the MCP server.
"""

import argparse
import sys

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
    
    args = parser.parse_args()
    
    # Default to 'start' if no command given
    if args.command is None:
        args.command = "start"
        args.transport = "stdio"
    
    if args.command == "start":
        start_server(args.transport)


def start_server(transport: str = "stdio"):
    """Start the MCP server."""
    from manhattan_mcp.config import get_config
    from manhattan_mcp.server import mcp
    
    # Validate configuration
    try:
        config = get_config()
        config.validate()
    except ValueError as e:
        sys.exit(1)
    
    print(f"🧠 Starting Manhattan MCP Server v{__version__}", file=sys.stderr)
    print(f"📡 API URL: {config.api_url}", file=sys.stderr)
    print(f"🔑 API Key: {config.api_key[:8]}...{config.api_key[-4:]}", file=sys.stderr)
    print(f"🚀 Transport: {transport}", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Start the server
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse")


if __name__ == "__main__":
    main()
