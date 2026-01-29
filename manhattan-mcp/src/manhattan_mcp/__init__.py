"""
Manhattan MCP - Local MCP Server for Manhattan Memory System

This package provides an MCP (Model Context Protocol) server that connects
AI agents (Claude Desktop, Cursor, etc.) to the Manhattan memory system.
"""

__version__ = "0.1.0"
__author__ = "Agent Architects Studio"

from manhattan_mcp.server import mcp

__all__ = ["mcp", "__version__"]
