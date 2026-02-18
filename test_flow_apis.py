
import asyncio
import json
import os
import sys
from typing import List, Dict, Any

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "manhattan-mcp/src")))

# Mocking the LocalAPI and CodingAPI if needed, but we should use real ones if available
try:
    from manhattan_mcp.gitmem_coding.coding_api import CodingAPI
except ImportError:
    print("Error: Could not import CodingAPI. Check paths.")
    sys.exit(1)

# Test Configurations
AGENT_ID = "test_flow_agent_001"
# Use absolute path for index.py
FILE_PATH = os.path.abspath("index.py")
ROOT_PATH = os.path.abspath("./test_gitmem_v2")

def create_manual_chunks() -> List[Dict[str, Any]]:
    """Manually created chunks for index.py based on its content."""
    return [
        {
            "name": "gevent_patching",
            "type": "block",
            "content": "from gevent import monkey\nmonkey.patch_all()",
            "summary": "Critical startup logic that applies gevent monkey patching to ensure recursive patching of standard library modules.",
            "keywords": ["gevent", "monkey patch", "startup", "concurrency"],
            "start_line": 1,
            "end_line": 15
        },
        {
            "name": "app_initialization",
            "type": "block",
            "content": "app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)",
            "summary": "Initializes the Flask application with specific static and template directories.",
            "keywords": ["Flask", "initialization", "app", "static", "templates"],
            "start_line": 79,
            "end_line": 83
        },
        {
            "name": "SocketIO_init",
            "type": "block",
            "content": "socketio = SocketIO(app, cors_allowed_origins=\"*\", async_mode='gevent', ping_timeout=60, ping_interval=25)",
            "summary": "Initializes Flask-SocketIO with gevent async mode for real-time communication.",
            "keywords": ["SocketIO", "gevent", "real-time", "websocket"],
            "start_line": 85,
            "end_line": 95
        },
        {
            "name": "login_route",
            "type": "function",
            "content": "@app.route('/login', methods=['POST'])\ndef login():...",
            "summary": "Handles user login by verifying credentials against Supabase and managing sessions.",
            "keywords": ["login", "auth", "Supabase", "session", "POST"],
            "start_line": 447,
            "end_line": 480
        },
        {
            "name": "google_auth",
            "type": "function",
            "content": "@app.route(\"/login/google\")\ndef login_google():...",
            "summary": "Initiates Google OAuth flow via Supabase authorize endpoint.",
            "keywords": ["google", "oauth", "auth", "Supabase", "redirect"],
            "start_line": 481,
            "end_line": 486
        },
        {
            "name": "memory_upload",
            "type": "function",
            "content": "@app.route('/memory', methods=['GET', 'POST'])\n@login_required\ndef memory():...",
            "summary": "Handles memory uploads including text and multiple file types (PDF, Doc, Images) for RAG integration.",
            "keywords": ["memory", "upload", "RAG", "file", "text"],
            "start_line": 1507,
            "end_line": 1600
        }
    ]

async def run_test():
    print(f"--- Starting Flow API Test ---")
    print(f"Agent ID: {AGENT_ID}")
    print(f"Storage Path: {ROOT_PATH}")
    
    # Initialize API
    api = CodingAPI(root_path=ROOT_PATH)
    
    # Clean up previous test data if any
    if os.path.exists(ROOT_PATH):
        import shutil
        shutil.rmtree(ROOT_PATH)
    os.makedirs(ROOT_PATH)
    
    # 1. Create Flow
    print("\n1. Calling create_mem for index.py...")
    chunks = create_manual_chunks()
    result = api.create_mem(AGENT_ID, FILE_PATH, chunks)
    print(f"   Success: {result.get('status') == 'OK'}")
    print(f"   Context ID: {result.get('context_id')}")
    
    # 2. Test Complex Queries
    queries = [
        "How is the gevent monkey patching handled at startup?",
        "Where is Google OAuth authentication implemented?",
        "How are files uploaded to the memory system for RAG?",
        "What is the real-time communication setup for the app?"
    ]
    
    print("\n2. Testing get_mem with complex queries...")
    for query in queries:
        print(f"\n   Query: '{query}'")
        search_res = api.get_mem(AGENT_ID, query)
        top_results = search_res.get("results", [])
        
        if top_results:
            top = top_results[0]
            chunk = top['chunk']
            print(f"   Found match: {chunk['name']} (Type: {chunk['type']})")
            print(f"   Score: {top['score']:.4f} (Vector: {top['vector_score']:.4f}, Keyword: {top['keyword_score']:.4f})")
            print(f"   Summary: {chunk['summary'][:100]}...")
            
            # Verify if the match is correct (simple name check)
            expected_map = {
                "gevent": "gevent_patching",
                "Google": "google_auth",
                "upload": "memory_upload",
                "real-time": "SocketIO_init"
            }
            matched_correctly = False
            for k, v in expected_map.items():
                if k in query and v == chunk['name']:
                    matched_correctly = True
                    break
            
            if matched_correctly:
                print("   ✅ MATCH CORRECT")
            else:
                print("   ❌ MATCH INCORRECT (Expected specialized chunk)")
        else:
            print("   No results found.")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
