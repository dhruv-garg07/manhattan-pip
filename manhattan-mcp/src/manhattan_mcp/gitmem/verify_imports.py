
try:
    from mcp.server.fastmcp import FastMCP, Context
    from mcp.types import UserMessage, SystemMessage
    print("Imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
