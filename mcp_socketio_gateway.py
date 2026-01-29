"""
MCP Socket.IO Gateway - Remote MCP Server via WebSocket

This module exposes the MCP memory tools via Socket.IO, allowing AI agents
to connect remotely using only a URL (no client file needed).

Usage:
    AI agents connect via Socket.IO to the /mcp namespace and use:
    - mcp:get_tools - Get available tools and their schemas
    - mcp:call_tool - Execute a tool with arguments

Example Client:
    import socketio
    sio = socketio.Client()
    sio.connect("https://themanhattanproject.ai", namespaces=["/mcp"])
    
    tools = sio.call("mcp:get_tools", {"api_key": "your-key"}, namespace="/mcp")
    result = sio.call("mcp:call_tool", {
        "api_key": "your-key",
        "tool": "search_memory",
        "arguments": {"agent_id": "my-agent", "query": "user preferences"}
    }, namespace="/mcp")
"""

import os
import sys
import json
import asyncio
import functools
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask_socketio import SocketIO, emit, disconnect
from flask import request

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# Import key utilities for API key verification
from key_utils import hash_key

from flask import Blueprint, Response, request, stream_with_context, jsonify, redirect
import uuid
import threading
from time import time as get_time

# Use gevent-compatible queue to avoid blocking workers
try:
    from gevent.queue import Queue as GeventQueue
    GEVENT_AVAILABLE = True
except ImportError:
    from queue import Queue as GeventQueue
    GEVENT_AVAILABLE = False
    print("[MCP Gateway] Warning: gevent not available, using standard queue (may cause 502 errors)")

# Define Blueprint for SSE
mcp_bp = Blueprint('mcp_sse', __name__)

# Session configuration
SESSION_TTL_SECONDS = 3600  # 1 hour TTL for sessions
MAX_GLOBAL_SESSIONS = 100   # Maximum concurrent SSE sessions
MAX_USER_SESSIONS = 5       # Maximum sessions per user
HEARTBEAT_INTERVAL = 15     # Seconds between heartbeats (reduced from 30)
SESSION_CLEANUP_INTERVAL = 300  # Run cleanup every 5 minutes

# Store SSE sessions with metadata: session_id -> {queue, user_id, created_at, last_activity}
_sse_sessions: Dict[str, Dict[str, Any]] = {}
_session_lock = threading.Lock()
_last_cleanup_time = 0


def _cleanup_stale_sessions():
    """Remove sessions that have exceeded TTL."""
    global _last_cleanup_time
    current_time = get_time()
    
    # Only run cleanup periodically
    if current_time - _last_cleanup_time < SESSION_CLEANUP_INTERVAL:
        return
    
    _last_cleanup_time = current_time
    
    with _session_lock:
        stale_sessions = []
        for session_id, session_data in _sse_sessions.items():
            if current_time - session_data.get('created_at', 0) > SESSION_TTL_SECONDS:
                stale_sessions.append(session_id)
        
        for session_id in stale_sessions:
            print(f"[MCP SSE] Cleaning up stale session: {session_id}")
            del _sse_sessions[session_id]
        
        if stale_sessions:
            print(f"[MCP SSE] Cleaned up {len(stale_sessions)} stale sessions")


def _count_user_sessions(user_id: str) -> int:
    """Count active sessions for a user."""
    count = 0
    for session_data in _sse_sessions.values():
        if session_data.get('user_id') == user_id:
            count += 1
    return count


def _create_session(user_id: Optional[str] = None) -> Optional[str]:
    """Create a new session with limits enforcement."""
    _cleanup_stale_sessions()
    
    with _session_lock:
        # Check global limit
        if len(_sse_sessions) >= MAX_GLOBAL_SESSIONS:
            print(f"[MCP SSE] Global session limit reached ({MAX_GLOBAL_SESSIONS})")
            return None
        
        # Check per-user limit
        if user_id and _count_user_sessions(user_id) >= MAX_USER_SESSIONS:
            print(f"[MCP SSE] User session limit reached for {user_id} ({MAX_USER_SESSIONS})")
            return None
        
        session_id = str(uuid.uuid4())
        _sse_sessions[session_id] = {
            'queue': GeventQueue(),
            'user_id': user_id,
            'created_at': get_time(),
            'last_activity': get_time()
        }
        
        print(f"[MCP SSE] Created session: {session_id} (total: {len(_sse_sessions)})")
        return session_id


def _get_session_queue(session_id: str) -> Optional[GeventQueue]:
    """Get the queue for a session, updating last activity."""
    session_data = _sse_sessions.get(session_id)
    if session_data:
        session_data['last_activity'] = get_time()
        return session_data.get('queue')
    return None


def _delete_session(session_id: str):
    """Remove a session."""
    with _session_lock:
        if session_id in _sse_sessions:
            del _sse_sessions[session_id]
            print(f"[MCP SSE] Deleted session: {session_id} (remaining: {len(_sse_sessions)})")

# Supabase for API key validation (optional - falls back to dev mode if unavailable)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

_supabase = None
try:
    from supabase import create_client
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print("[MCP Gateway] Supabase client initialized for API key validation")
except ImportError as e:
    print(f"[MCP Gateway] Supabase not available: {e}. Using development mode for auth.")
except Exception as e:
    print(f"[MCP Gateway] Supabase init error: {e}. Using development mode for auth.")

# Import the MCP tools from mcp_memory_client
from mcp_memory_client import mcp

# Store connected clients
_connected_clients: Dict[str, Dict[str, Any]] = {}


def verify_api_key(api_key: Optional[str]) -> Dict[str, Any]:
    """
    Verify an API key against the database.
    
    Returns:
        Dict with 'ok': True/False and 'user_id' if valid
    """
    if not api_key:
        return {"ok": False, "error": "API key required"}
    
    if not _supabase:
        # Development mode - allow if key starts with 'sk-'
        if api_key.startswith("sk-"):
            return {"ok": True, "user_id": "dev-user", "mode": "development"}
        return {"ok": False, "error": "Database not configured"}
    
    try:
        # Hash the key and look it up
        hashed = hash_key(api_key)
        
        result = _supabase.table("api_keys").select("id, user_id, status, permissions").eq("hashed_key", hashed).execute()
        
        if not result.data:
            # Try legacy key column (some old keys stored hash there)
            result = _supabase.table("api_keys").select("id, user_id, status, permissions").eq("key", hashed).execute()
        
        if result.data and len(result.data) > 0:
            key_record = result.data[0]
            if key_record.get("status") == "active":
                return {
                    "ok": True,
                    "user_id": key_record.get("user_id"),
                    "permissions": key_record.get("permissions", {})
                }
            else:
                return {"ok": False, "error": "API key is not active"}
        
        return {"ok": False, "error": "Invalid API key"}
    
    except Exception as e:
        print(f"[MCP Gateway] API key verification error: {e}")
        return {"ok": False, "error": "Verification failed"}


def get_tools_schema() -> Dict[str, Any]:
    """
    Extract tool schemas from the FastMCP server.
    
    Returns a dictionary of tool names to their schemas.
    """
    tools = {}
    
    # Get tools from FastMCP's internal registry
    if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
        for name, tool in mcp._tool_manager._tools.items():
            tools[name] = {
                "name": name,
                "description": tool.description if hasattr(tool, 'description') else "",
                "parameters": tool.parameters if hasattr(tool, 'parameters') else {}
            }
    elif hasattr(mcp, 'list_tools'):
        # Try alternative method
        try:
            tool_list = asyncio.run(mcp.list_tools())
            for tool in tool_list:
                tools[tool.name] = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                }
        except Exception as e:
            print(f"[MCP Gateway] Error listing tools: {e}")
    
    # Fallback: manually define core tools if detection fails
    if not tools:
        tools = _get_fallback_tools_schema()
    
    return tools


def _get_fallback_tools_schema() -> Dict[str, Any]:
    """Fallback tool definitions if auto-detection fails."""
    return {
        "search_memory": {
            "name": "search_memory",
            "description": "Search memories using hybrid retrieval (semantic + keyword)",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Agent identifier"},
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Max results", "default": 5},
                    "enable_reflection": {"type": "boolean", "default": False}
                },
                "required": ["agent_id", "query"]
            }
        },
        "add_memory_direct": {
            "name": "add_memory_direct",
            "description": "Store memories directly without LLM processing",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "memories": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["agent_id", "memories"]
            }
        },
        "auto_remember": {
            "name": "auto_remember",
            "description": "Automatically extract and store facts from user message",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "user_message": {"type": "string"}
                },
                "required": ["agent_id", "user_message"]
            }
        },
        "get_context_answer": {
            "name": "get_context_answer",
            "description": "Get AI-generated answer using memory context",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "question": {"type": "string"}
                },
                "required": ["agent_id", "question"]
            }
        },
        "session_start": {
            "name": "session_start",
            "description": "Initialize a new conversation session",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "auto_pull_context": {"type": "boolean", "default": True}
                }
            }
        },
        "session_end": {
            "name": "session_end",
            "description": "End the current session and sync memories",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "conversation_summary": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "agent_stats": {
            "name": "agent_stats",
            "description": "Get comprehensive statistics for an agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"}
                },
                "required": ["agent_id"]
            }
        },
        "create_agent": {
            "name": "create_agent",
            "description": "Create a new agent in the system",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string"},
                    "agent_slug": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["agent_name", "agent_slug"]
            }
        },
        "list_agents": {
            "name": "list_agents",
            "description": "List all agents owned by the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"}
                }
            }
        }
    }


def _get_tool_function(tool_name: str):
    """Get the tool function by name from FastMCP or module."""
    tool_fn = None
    
    if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
        tool = mcp._tool_manager._tools.get(tool_name)
        if tool:
            tool_fn = tool.fn if hasattr(tool, 'fn') else tool
    
    if not tool_fn:
        # Try to get function from module globals
        import mcp_memory_client
        tool_fn = getattr(mcp_memory_client, tool_name, None)
    
    return tool_fn


def _execute_tool_sync(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Execute an MCP tool synchronously - gevent-safe version.
    
    This avoids asyncio event loops which conflict with gevent workers.
    Uses gevent.spawn with timeout for safe concurrent execution.
    """
    tool_fn = _get_tool_function(tool_name)
    
    if not tool_fn:
        return {"ok": False, "error": f"Tool '{tool_name}' not found"}
    
    try:
        # Check if it's an async function
        if asyncio.iscoroutinefunction(tool_fn):
            # For async functions, we need to run them in a new event loop
            # But we use gevent.spawn to not block the worker
            if GEVENT_AVAILABLE:
                import gevent
                from gevent import Timeout
                
                def run_async():
                    loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(loop)
                        return loop.run_until_complete(tool_fn(**arguments))
                    finally:
                        loop.close()
                
                # Spawn a greenlet with timeout
                greenlet = gevent.spawn(run_async)
                try:
                    with Timeout(90):  # 90 second timeout for tool execution
                        result = greenlet.get()
                except Timeout:
                    greenlet.kill()
                    return {"ok": False, "error": "Tool execution timed out (90s)"}
            else:
                # Fallback: run directly (may block)
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(tool_fn(**arguments))
                finally:
                    loop.close()
        else:
            # Sync function - execute directly
            result = tool_fn(**arguments)
        
        # Parse JSON result if it's a string
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"ok": True, "result": result}
        
        return result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


async def execute_tool_async(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Async wrapper that delegates to sync execution for gevent compatibility."""
    # In gevent environment, we use sync execution to avoid event loop conflicts
    return _execute_tool_sync(tool_name, arguments)


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Synchronous wrapper for tool execution - gevent-safe."""
    return _execute_tool_sync(tool_name, arguments)


# ============================================================================
# Standard MCP SSE Implementation (for "No Local File" usage)
# ============================================================================

# Primary MCP endpoint - handles both streamable HTTP and SSE
@mcp_bp.route("/mcp", methods=["POST", "GET", "DELETE"])
def handle_mcp_root():
    """
    Primary MCP endpoint supporting streamable-http transport.
    
    For POST requests: Direct JSON-RPC processing (streamable-http)
    For GET requests: Redirect to SSE endpoint
    """
    api_key = request.args.get("api_key") or request.headers.get("Authorization", "").replace("Bearer ", "")
    auth = verify_api_key(api_key)
    if not auth["ok"]:
        return jsonify({"error": "Unauthorized", "message": auth.get("error")}), 401
    
    if request.method == "GET":
        # Redirect to SSE endpoint for SSE transport
        return redirect(url_for('mcp_sse.handle_sse', api_key=api_key))
    
    if request.method == "POST":
        # Streamable HTTP - process JSON-RPC directly and return response
        try:
            message = request.json
            print(f"[MCP HTTP] Processing: {message.get('method', 'unknown')} (id: {message.get('id', 'none')})")
            
            # Process the JSON-RPC message directly (sync - no asyncio)
            response = _process_json_rpc_direct(message)
            print(f"[MCP HTTP] Response generated for: {message.get('method', 'unknown')}")
            return jsonify(response)
                
        except Exception as e:
            print(f"[MCP HTTP] Error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "jsonrpc": "2.0",
                "id": request.json.get("id") if request.json else None,
                "error": {"code": -32603, "message": str(e)}
            }), 500
    
    return jsonify({"error": "Method not allowed"}), 405


@mcp_bp.route("/mcp/sse", methods=["POST", "GET", "DELETE"])
def handle_sse():
    """
    MCP SSE Endpoint - supports both SSE and streamable-http transports.
    
    GET: Establishes SSE connection and streams responses
    POST: Handles streamable-http JSON-RPC requests directly
    DELETE: Cleans up sessions
    """
    api_key = request.args.get("api_key") or request.headers.get("Authorization", "").replace("Bearer ", "")
    auth = verify_api_key(api_key)
    if not auth["ok"]:
        return jsonify({"error": "Unauthorized", "message": auth.get("error")}), 401
    
    # Handle POST as streamable-http transport (JSON-RPC direct)
    if request.method == "POST":
        try:
            message = request.json
            print(f"[MCP HTTP] POST to /mcp/sse - Processing: {message.get('method', 'unknown')} (id: {message.get('id', 'none')})")
            
            # Process the JSON-RPC message directly (sync - no asyncio)
            response_data = _process_json_rpc_direct(message)
            print(f"[MCP HTTP] Response generated for: {message.get('method', 'unknown')}")
            return jsonify(response_data)
                
        except Exception as e:
            print(f"[MCP HTTP] Error processing POST: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "jsonrpc": "2.0",
                "id": request.json.get("id") if request.json else None,
                "error": {"code": -32603, "message": str(e)}
            }), 500
    
    # Handle DELETE - cleanup
    if request.method == "DELETE":
        return jsonify({"status": "ok", "message": "Session cleanup acknowledged"}), 200
    
    # Handle GET as SSE transport
    # Extract user_id from auth for session tracking
    user_id = auth.get('user_id')
    
    # Create session with limits enforcement
    session_id = _create_session(user_id)
    if not session_id:
        return jsonify({
            "error": "Session limit reached",
            "message": "Too many active connections. Please try again later."
        }), 429
    
    session_queue = _get_session_queue(session_id)
    
    def generate():
        # 1. Send the endpoint event telling client where to POST messages
        endpoint_url = url_for('mcp_sse.handle_messages', session_id=session_id, _external=True)
        print(f"[MCP SSE] Sending endpoint URL: {endpoint_url}")
        yield f"event: endpoint\ndata: {endpoint_url}\n\n"
        
        try:
            while True:
                try:
                    # Use shorter timeout with gevent-compatible queue
                    data = session_queue.get(timeout=HEARTBEAT_INTERVAL)
                    yield f"event: message\ndata: {json.dumps(data)}\n\n"
                except Exception:
                    # Send heartbeat to keep connection alive (works for both Empty and gevent timeout)
                    yield ": heartbeat\n\n"
                    continue
        except GeneratorExit:
            print(f"[MCP SSE] Session closed by client: {session_id}")
            _delete_session(session_id)
        except Exception as e:
            print(f"[MCP SSE] Error in stream: {e}")
            _delete_session(session_id)

    response = Response(stream_with_context(generate()), content_type="text/event-stream")
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


# SOCKET IO ROUTE
@mcp_bp.route("/mcp/messages", methods=["POST", "GET", "DELETE"])
def handle_messages():
    """
    Standard MCP Message Endpoint.
    Receives JSON-RPC messages and queues responses.
    """
    session_id = request.args.get("session_id")
    print(f"[MCP SSE] Message received for session: {session_id}, method: {request.method}")
    
    if not session_id:
        print("[MCP SSE] No session_id provided")
        return jsonify({"error": "session_id required"}), 400
        
    session_queue = _get_session_queue(session_id)
    if not session_queue:
        print(f"[MCP SSE] Session not found: {session_id}, active sessions: {list(_sse_sessions.keys())}")
        return jsonify({"error": "Session not found or expired"}), 404
    
    if request.method in ["GET", "DELETE"]:
        # Health check or cleanup
        return jsonify({"status": "ok", "session_id": session_id}), 200
    
    try:
        message = request.json
        print(f"[MCP SSE] Processing message: {message.get('method', 'unknown')} (id: {message.get('id', 'none')})")
        
        # Process synchronously - gevent-safe, no asyncio
        _process_json_rpc_sync(session_id, message)
        print(f"[MCP SSE] Message processed successfully for session: {session_id}")
        
        return "Accepted", 202
    except Exception as e:
        print(f"[MCP SSE] Error handling message: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@mcp_bp.route("/mcp/<tool_name>", methods=["POST", "GET", "DELETE"])
def handle_tool_rest(tool_name):
    """
    REST Endpoint for direct tool execution (used by mcp_memory_client.py).
    Matches POST /mcp/auto_remember, /mcp/search_memory, etc.
    """
    # Skip reserved routes
    if tool_name in ["sse", "messages", "search_tool"]: # Add any other reserved names
        return jsonify({"error": "Reserved endpoint"}), 404

    # Auth check
    auth_header = request.headers.get("Authorization")
    api_key = None
    if auth_header and auth_header.startswith("Bearer "):
        api_key = auth_header.split(" ")[1]
    
    # Fallback to query param or body
    if not api_key:
        api_key = request.args.get("api_key") or (request.json and request.json.get("api_key"))

    auth = verify_api_key(api_key)
    if not auth["ok"]:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        arguments = request.json or {}
        # Remove auth args if present to avoid passing to tool
        if "api_key" in arguments:
            del arguments["api_key"]
            
        print(f"[MCP REST] Executing tool: {tool_name}")
        
        # Execute tool
        result = execute_tool(tool_name, arguments)
        
        # If result is already a dict/json string, ensure it's returned as JSON
        if isinstance(result, str):
            try:
                # Try to parse if it looks like JSON
                json_result = json.loads(result)
                return jsonify(json_result)
            except:
                return jsonify({"result": result})
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[MCP REST] Error executing {tool_name}: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


def _process_json_rpc_direct(message: Dict[str, Any]) -> Dict[str, Any]:
    """Process incoming JSON-RPC message and return response directly (for streamable-http). Synchronous for gevent compatibility."""
    if not isinstance(message, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}}
        
    msg_type = message.get("method")
    msg_id = message.get("id")
    
    try:
        # Initialize
        if msg_type == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "manhattan-memory",
                        "version": "1.0.0"
                    }
                }
            }
            
        # List Tools
        elif msg_type == "tools/list":
            tools_schema = get_tools_schema()
            tool_list = []
            for name, schema in tools_schema.items():
                tool_list.append({
                    "name": name,
                    "description": schema.get("description", ""),
                    "inputSchema": schema.get("parameters", {})
                })
            
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": tool_list
                }
            }
            
        # Call Tool
        elif msg_type == "tools/call":
            params = message.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            
            # Use synchronous execution for gevent compatibility
            result = execute_tool(name, args)
            
            # Format result for MCP (content array)
            if isinstance(result, dict) and not result.get("ok", True) and "error" in result:
                content = [{"type": "text", "text": f"Error: {result['error']}"}]
                is_error = True
            else:
                content = [{"type": "text", "text": json.dumps(result, indent=2)}]
                is_error = False
                
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": content,
                    "isError": is_error
                }
            }
            
        # Ping / Notifications (respond with empty result)
        elif msg_type == "notifications/initialized":
            # No response needed for notifications
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
            
        else:
            # Unknown method
            print(f"[MCP HTTP] Unknown method: {msg_type}")
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {msg_type}"
                }
            }
                
    except Exception as e:
        print(f"[MCP HTTP] Execution error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }


def _process_json_rpc_sync(session_id: str, message: Dict[str, Any]):
    """Process incoming JSON-RPC message and queue response. Synchronous for gevent."""
    if not isinstance(message, dict):
        return
        
    msg_type = message.get("method")
    msg_id = message.get("id")
    
    response = None
    
    try:
        # Initialize
        if msg_type == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "manhattan-memory-sse",
                        "version": "1.0.0"
                    }
                }
            }
            
        # List Tools
        elif msg_type == "tools/list":
            tools_schema = get_tools_schema()
            tool_list = []
            for name, schema in tools_schema.items():
                tool_list.append({
                    "name": name,
                    "description": schema.get("description", ""),
                    "inputSchema": schema.get("parameters", {})
                })
            
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": tool_list
                }
            }
            
        # Call Tool
        elif msg_type == "tools/call":
            params = message.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            
            # Use synchronous tool execution
            result = execute_tool(name, args)
            
            # Format result for MCP (content array)
            if isinstance(result, dict) and not result.get("ok", True) and "error" in result:
                content = [{"type": "text", "text": f"Error: {result['error']}"}]
                is_error = True
            else:
                content = [{"type": "text", "text": json.dumps(result, indent=2)}]
                is_error = False
                
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": content,
                    "isError": is_error
                }
            }
            
        # Ping / Notifications (ignore or ack)
        elif msg_type == "notifications/initialized":
            # Client confirming init
            return
            
        else:
            # Unknown method
            print(f"[MCP SSE] Unknown method: {msg_type}")
            # Optional: send error back if it's a request (has id)
            if msg_id is not None:
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    }
                }
                
    except Exception as e:
        print(f"[MCP SSE] Execution error: {e}")
        if msg_id is not None:
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }

    # Send response if generated
    if response:
        session_queue = _get_session_queue(session_id)
        if session_queue:
            session_queue.put(response)


# Import url_for needs to be inside request context or imported
from flask import url_for


def init_mcp_socketio(socketio: SocketIO):
    """
    Initialize Socket.IO event handlers for MCP.
    Call this from your main Flask app after creating SocketIO instance.
    """
    
    @socketio.on("connect", namespace="/mcp")
    def handle_connect():
        """Handle new WebSocket connection."""
        client_id = request.sid
        print(f"[MCP Gateway] Client connected: {client_id}")
        
        _connected_clients[client_id] = {
            "connected_at": datetime.now().isoformat(),
            "authenticated": False
        }
        
        emit("connection_established", {
            "status": "connected",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "message": "Welcome to Manhattan MCP Gateway. Use mcp:get_tools to discover available tools."
        })
    
    @socketio.on("disconnect", namespace="/mcp")
    def handle_disconnect():
        """Handle WebSocket disconnection."""
        client_id = request.sid
        print(f"[MCP Gateway] Client disconnected: {client_id}")
        
        if client_id in _connected_clients:
            del _connected_clients[client_id]
    
    @socketio.on("mcp:get_tools", namespace="/mcp")
    def handle_get_tools(data):
        """
        Tool discovery endpoint.
        
        Request: {"api_key": "sk-xxx"}
        Response: {"ok": true, "tools": {...}}
        """
        api_key = data.get("api_key") if data else None
        
        # Verify API key
        auth = verify_api_key(api_key)
        if not auth.get("ok"):
            return {"ok": False, "error": auth.get("error", "Authentication failed")}
        
        # Mark client as authenticated
        client_id = request.sid
        if client_id in _connected_clients:
            _connected_clients[client_id]["authenticated"] = True
            _connected_clients[client_id]["user_id"] = auth.get("user_id")
        
        # Return tool schemas
        tools = get_tools_schema()
        return {
            "ok": True,
            "tools": tools,
            "tool_count": len(tools),
            "message": "Use mcp:call_tool to execute a tool"
        }
    
    @socketio.on("mcp:call_tool", namespace="/mcp")
    def handle_call_tool(data):
        """
        Tool execution endpoint.
        
        Request: {
            "api_key": "sk-xxx",
            "tool": "search_memory",
            "arguments": {"agent_id": "...", "query": "..."}
        }
        Response: Tool result or error
        """
        if not data:
            return {"ok": False, "error": "No data provided"}
        
        api_key = data.get("api_key")
        tool_name = data.get("tool")
        arguments = data.get("arguments", {})
        
        # Verify API key
        auth = verify_api_key(api_key)
        if not auth.get("ok"):
            return {"ok": False, "error": auth.get("error", "Authentication failed")}
        
        # Validate tool name
        if not tool_name:
            return {"ok": False, "error": "Tool name required"}
        
        # Execute the tool
        try:
            result = execute_tool(tool_name, arguments)
            return result
        except Exception as e:
            print(f"[MCP Gateway] Tool execution error: {e}")
            return {"ok": False, "error": str(e)}
    
    @socketio.on("mcp:ping", namespace="/mcp")
    def handle_ping():
        """Health check ping/pong."""
        return {
            "ok": True,
            "pong": True,
            "timestamp": datetime.now().isoformat()
        }
    
    @socketio.on("mcp:get_instructions", namespace="/mcp")
    def handle_get_instructions(data=None):
        """
        Get MCP instructions for AI agents.
        
        Response: Instructions for how to use the memory system
        """
        api_key = data.get("api_key") if data else None
        
        auth = verify_api_key(api_key)
        if not auth.get("ok"):
            return {"ok": False, "error": auth.get("error", "Authentication failed")}
        
        return {
            "ok": True,
            "instructions": """
Manhattan Memory MCP - Remote AI Agent Instructions

This is a PERSISTENT MEMORY SYSTEM for storing and retrieving information.

MANDATORY WORKFLOW:
1. Call session_start at conversation beginning
2. Call search_memory BEFORE answering user questions
3. Call add_memory_direct when user shares new information
4. Call auto_remember after every user message
5. Call session_end when conversation ends

MEMORY TRIGGERS - ALWAYS STORE:
- User's name, preferences, interests
- Important dates, deadlines, events
- Technical details, project information
- Personal context shared by user
- Decisions, agreements, action items

CONNECTION EXAMPLE:
    import socketio
    sio = socketio.Client()
    sio.connect("https://themanhattanproject.ai", namespaces=["/mcp"])
    result = sio.call("mcp:call_tool", {
        "api_key": "your-key",
        "tool": "search_memory",
        "arguments": {"agent_id": "your-agent", "query": "user info"}
    }, namespace="/mcp")
""",
            "default_agent_id": "84aab1f8-3ea9-4c6a-aa3c-cd8eaa274a5e"
        }
    
    print("[MCP Gateway] Socket.IO handlers initialized on /mcp namespace")
    return socketio
