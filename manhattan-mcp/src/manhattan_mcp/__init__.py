"""
Manhattan MCP - Local MCP Server for Manhattan Memory System

This package provides an MCP (Model Context Protocol) server that connects
AI agents (Claude Desktop, Cursor, etc.) to the Manhattan memory system.
"""

__version__ = "0.1.0"
__author__ = "Agent Architects Studio"

from manhattan_mcp.server import mcp
from manhattan_mcp.config import get_config, Config
from . import gitmem

__all__ = ["mcp", "get_config", "Config", "gitmem", "__version__"]
