"""
Direct Test: create_mem & get_mem from CodingAPI
===================================================
Tests the CodingAPI functions directly (no MCP layer) by:
1. Storing large coding context from index.py as semantic chunks
2. Retrieving them with various queries via get_mem
3. Running 5 iterative test loops, each with improved queries/validation
4. AI-style verification of retrieval relevance
"""

import sys
import os
import json
import time
import shutil
import io

# Add src to path so we can import the gitmem_coding module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "manhattan_mcp"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

# ============================================================================
# Test Configuration
# ============================================================================
AGENT_ID = "test_flow_agent"
FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index.py"))
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), ".test_gitmem_coding")
LOG_FILE = os.path.join(os.path.dirname(__file__), "test_flow_results.log")

# Use a log file for clean output
log = None

def log_print(*args):
    """Print to both console and log file."""
    msg = " ".join(str(a) for a in args)
    print(msg)
    if log:
        log.write(msg + "\n")
        log.flush()

# ============================================================================
# Large Coding Context Chunks from index.py (semantic decomposition)
# ============================================================================
INDEX_PY_CHUNKS = [
    {
        "name": "FlaskApp_init",
        "type": "module",
        "content": "app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)\napp.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')",
        "summary": "Flask application initialization with static and template directory configuration. Sets up the main Flask app instance with secret key from environment variables. Uses gevent monkey patching at startup for async worker compatibility.",
        "keywords": ["flask", "app", "initialization", "static", "templates", "secret_key", "gevent"],
        "start_line": 1,
        "end_line": 82
    },
    {
        "name": "SocketIO_MCP_setup",
        "type": "block",
        "content": "socketio = SocketIO(app, cors_allowed_origins='*', async_mode='gevent')\ninit_websocket(socketio)\ninit_mcp_socketio(socketio)",
        "summary": "Flask-SocketIO initialization with gevent async mode for real-time WebSocket updates. Integrates GitMem WebSocket handlers and MCP Socket.IO Gateway. MCP SSE Blueprint registered at /mcp for server-sent events transport.",
        "keywords": ["socketio", "websocket", "mcp", "sse", "gateway", "gevent", "real-time", "gitmem"],
        "start_line": 84,
        "end_line": 124
    },
    {
        "name": "User",
        "type": "class",
        "content": "class User:\n    def __init__(self, user_id=None, email=None): ...\n    @property is_authenticated, is_active, is_anonymous\n    def get_id(self): ...",
        "summary": "User model class for Flask-Login integration. Stores user_id and email. Properties: is_authenticated (True if user_id exists), is_active (always True), is_anonymous (True if no user_id). get_id returns string user_id.",
        "keywords": ["user", "model", "flask-login", "authentication", "user_id", "email", "session"],
        "start_line": 184,
        "end_line": 210
    },
    {
        "name": "keep_alive_task",
        "type": "function",
        "content": "def keep_alive_task():\n    def ping_website(): ...\n    keep_alive_thread = threading.Thread(target=ping_website, daemon=True)",
        "summary": "Background keep-alive task that pings the website every 5 minutes (300s interval) to prevent Render from sleeping. Uses a daemon thread so it doesn't block shutdown. Pings /ping endpoint.",
        "keywords": ["keep-alive", "background", "thread", "daemon", "ping", "render", "timer", "health"],
        "start_line": 158,
        "end_line": 181
    },
    {
        "name": "explore",
        "type": "function",
        "content": "@app.route('/explore')\ndef explore():\n    filters = SearchFilters(...)\n    agents = asyncio.run(agent_service.fetch_agents(filters))",
        "summary": "Explore page route handler. Accepts URL query parameters: search, category, model, status, sort_by, modalities, capabilities. Creates SearchFilters object and fetches agents.",
        "keywords": ["explore", "agents", "search", "filter", "category", "model", "route", "marketplace"],
        "start_line": 236,
        "end_line": 273
    },
    {
        "name": "agent_detail",
        "type": "function",
        "content": "@app.route('/agent/<agent_id>')\ndef agent_detail(agent_id):\n    agent = asyncio.run(agent_service.get_agent_by_id(agent_id))",
        "summary": "Agent detail page route. Takes agent_id URL parameter, fetches full agent data. Redirects to explore page if not found.",
        "keywords": ["agent", "detail", "route", "get_agent_by_id", "agent_id", "page"],
        "start_line": 274,
        "end_line": 289
    },
    {
        "name": "login",
        "type": "function",
        "content": "@app.route('/login', methods=['POST'])\ndef login():\n    auth = supabase.auth.sign_in_with_password({...})",
        "summary": "POST login handler. Cleans email, validates credentials, signs in via Supabase Auth sign_in_with_password. Stores session tokens. Creates User object and calls login_user().",
        "keywords": ["login", "authentication", "supabase", "password", "token", "session", "flask-login"],
        "start_line": 447,
        "end_line": 479
    },
    {
        "name": "login_google",
        "type": "function",
        "content": "@app.route('/login/google')\ndef login_google():\n    oauth_url = f'{SUPABASE_URL}/auth/v1/authorize?provider=google'",
        "summary": "Google OAuth login initiation. Generates Supabase OAuth authorization URL with google provider. Redirects user to Google consent screen.",
        "keywords": ["google", "oauth", "login", "supabase", "authorization", "redirect"],
        "start_line": 481,
        "end_line": 485
    },
    {
        "name": "auth_callback",
        "type": "function",
        "content": "@app.route('/auth/callback', methods=['GET', 'POST'])\ndef auth_callback():",
        "summary": "OAuth callback handler for Google login. GET serves auth_callback.html with JS to extract tokens. POST validates tokens via supabase.auth.get_user(), creates/upserts profile, stores tokens in session, logs in via Flask-Login.",
        "keywords": ["oauth", "callback", "google", "token", "validation", "profile", "upsert", "supabase"],
        "start_line": 487,
        "end_line": 547
    },
    {
        "name": "github_verify",
        "type": "function",
        "content": "@app.route('/auth/github/verify', methods=['POST'])\ndef github_verify():",
        "summary": "GitHub OAuth verification endpoint. Validates access_token against Supabase Auth, handles profile creation with unique username deduplication, updates github_url.",
        "keywords": ["github", "oauth", "verify", "profile", "username", "unique", "supabase", "authentication"],
        "start_line": 560,
        "end_line": 649
    },
    {
        "name": "register",
        "type": "function",
        "content": "@app.route('/register', methods=['POST'])\ndef register():\n    auth = supabase.auth.sign_up({...})",
        "summary": "User registration handler. Validates all required fields, passwords match, password >= 8 chars, username unique. Signs up via Supabase. Inserts profile with role-specific fields.",
        "keywords": ["register", "signup", "supabase", "validation", "profile", "password", "email", "confirmation"],
        "start_line": 658,
        "end_line": 755
    },
    {
        "name": "submit_agent",
        "type": "function",
        "content": "@app.route('/submit', methods=['POST'])\n@login_required\ndef submit_agent():",
        "summary": "Agent submission handler (POST, login required). Processes form data including headers, authentication, io_schema/out_schema, tags. Creates agent via agent_service.create_agent().",
        "keywords": ["submit", "agent", "create", "form", "headers", "schema", "authentication", "creator_studio"],
        "start_line": 334,
        "end_line": 411
    },
    {
        "name": "run_agent",
        "type": "function",
        "content": "def run_agent(user_input, agent_data):\n    response = requests.post(url, json=user_input['body'])",
        "summary": "Executes an agent by sending POST request to agent's base_url + run_path. Parses agent_data, builds URL, sends JSON payload. Returns response text or error.",
        "keywords": ["run", "agent", "execute", "post", "request", "base_url", "run_path", "payload"],
        "start_line": 974,
        "end_line": 1010
    },
    {
        "name": "dashboard",
        "type": "function",
        "content": "@app.route('/dashboard')\n@login_required\ndef dashboard():",
        "summary": "Dashboard page (login required). Fetches user profile from Supabase profiles table with fallback if not found. Renders dashboard.html.",
        "keywords": ["dashboard", "profile", "supabase", "login_required", "user", "fallback"],
        "start_line": 420,
        "end_line": 441
    },
    {
        "name": "edit_profile",
        "type": "function",
        "content": "@app.route('/profile/edit', methods=['GET', 'POST'])\n@login_required\ndef edit_profile():",
        "summary": "Profile editing handler (GET/POST, login required). Validates username uniqueness, updates profile with role-specific fields.",
        "keywords": ["edit", "profile", "update", "username", "validation", "supabase", "form"],
        "start_line": 846,
        "end_line": 912
    },
    {
        "name": "create_api_key",
        "type": "function",
        "content": "@app.route('/api/keys', methods=['POST'])\n@login_required\ndef create_api_key():",
        "summary": "API key creation endpoint. Generates secure key, hashes before storage, sets permissions/limits/expiration. Falls back to legacy column if needed.",
        "keywords": ["api_key", "create", "hash", "security", "permissions", "limits", "expiration", "supabase"],
        "start_line": 1077,
        "end_line": 1168
    },
    {
        "name": "join_waitlist",
        "type": "function",
        "content": "@app.route('/join-waitlist', methods=['POST'])\ndef join_waitlist():",
        "summary": "Waitlist signup endpoint. Validates email, checks duplicates, sends async welcome email with HTML template, inserts to Supabase.",
        "keywords": ["waitlist", "signup", "email", "validation", "supabase", "async", "count"],
        "start_line": 1208,
        "end_line": 1342
    },
    {
        "name": "join_gitmem_waitlist",
        "type": "function",
        "content": "@app.route('/join-gitmem-waitlist', methods=['POST'])\ndef join_gitmem_waitlist():",
        "summary": "GitMem-specific waitlist signup with extended data: email, name, tools, stack, goals, setup, open_to_feedback.",
        "keywords": ["gitmem", "waitlist", "signup", "tools", "stack", "goals", "feedback", "email"],
        "start_line": 1366,
        "end_line": 1505
    },
    {
        "name": "memory",
        "type": "function",
        "content": "@app.route('/memory', methods=['GET', 'POST'])\n@login_required\ndef memory():",
        "summary": "Memory page handler. Processes multiple file uploads with extension validation, saves to RAG DB. Also handles plain text memory. Cleans up temp files.",
        "keywords": ["memory", "upload", "file", "rag", "database", "controller", "text", "multifile"],
        "start_line": 1507,
        "end_line": 1607
    },
    {
        "name": "Supabase_config",
        "type": "block",
        "content": "SUPABASE_URL = os.environ.get('SUPABASE_URL')\nsupabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)\nsupabase_backend: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)",
        "summary": "Supabase client configuration. Two clients: anon key for user-facing (RLS), service role key for admin (bypasses RLS).",
        "keywords": ["supabase", "config", "client", "anon_key", "service_role", "rls", "environment"],
        "start_line": 146,
        "end_line": 151
    },
    {
        "name": "homepage",
        "type": "function",
        "content": "@app.route('/')\ndef homepage():\n    return render_template('homepage.html', user=current_user)",
        "summary": "Root route handler. Renders homepage.html template as the main landing page for the AI Agent Marketplace.",
        "keywords": ["homepage", "root", "route", "landing", "template", "marketplace"],
        "start_line": 1653,
        "end_line": 1656
    },
    {
        "name": "error_handlers",
        "type": "block",
        "content": "@app.errorhandler(404)\ndef not_found(error): ...\n@app.errorhandler(500)\ndef internal_error(error): ...",
        "summary": "Custom error handlers for 404 Not Found and 500 Internal Server Error. Each renders a dedicated HTML template.",
        "keywords": ["error", "handler", "404", "500", "not_found", "internal_error", "template"],
        "start_line": 1170,
        "end_line": 1178
    },
    {
        "name": "parse_to_dict",
        "type": "function",
        "content": "def parse_to_dict(raw: str):\n    pattern = re.compile(r'...')\n    result[key] = ast.literal_eval(value)",
        "summary": "Utility function to parse raw string into dictionary using regex pattern matching. Uses ast.literal_eval for safe evaluation.",
        "keywords": ["parse", "dict", "regex", "literal_eval", "utility", "string", "conversion"],
        "start_line": 955,
        "end_line": 970
    },
]


def ai_verify_relevance(query, results, expected_names):
    """AI-style verification: Check if retrieved results are relevant.
    
    Results from CodingHybridRetriever are in format:
    {"results": [{"file_path": ..., "chunk": {"name": ..., ...}, "score": ...}]}
    """
    if "error" in str(results.get("status", "")):
        return {"pass": False, "reason": f"Error in results: {results}", "score": 0, "found": [], "missed": expected_names, "unexpected": [], "total_retrieved": 0, "all_retrieved_names": [], "scores": []}
    
    retrieved_chunks = results.get("results", results.get("chunks", []))
    if isinstance(retrieved_chunks, list):
        # Name is nested inside "chunk" dict
        retrieved_names = []
        scores = []
        for c in retrieved_chunks:
            chunk_data = c.get("chunk", c)  # Try nested chunk first, fallback to top-level
            retrieved_names.append(chunk_data.get("name", ""))
            scores.append(round(c.get("score", 0), 4))
    else:
        retrieved_names = []
        scores = []
    
    found = [n for n in expected_names if n in retrieved_names]
    missed = [n for n in expected_names if n not in retrieved_names]
    
    score = len(found) / max(len(expected_names), 1) * 100
    
    return {
        "pass": score >= 40,
        "score": round(score, 1),
        "found": found,
        "missed": missed,
        "total_retrieved": len(retrieved_names),
        "all_retrieved_names": retrieved_names,
        "scores": scores
    }


def run_test_iteration(coding_api, iteration, log_lines):
    """Run a single test iteration."""
    
    log_lines.append(f"\n{'='*70}")
    log_lines.append(f"  TEST ITERATION {iteration}/5")
    log_lines.append(f"{'='*70}")
    
    test_cases = {
        1: [
            ("authentication login", ["login", "auth_callback", "github_verify"], "Basic auth search"),
            ("Flask app initialization", ["FlaskApp_init", "SocketIO_MCP_setup"], "App setup search"),
            ("agent detail explore", ["explore", "agent_detail"], "Agent pages search"),
            ("waitlist signup", ["join_waitlist", "join_gitmem_waitlist"], "Waitlist search"),
            ("user profile dashboard", ["dashboard", "edit_profile", "User"], "Profile search"),
        ],
        2: [
            ("OAuth callback Google login token", ["auth_callback", "login_google"], "OAuth flow"),
            ("file upload RAG memory processing", ["memory"], "Memory upload"),
            ("API key creation hashing security", ["create_api_key"], "API key security"),
            ("background thread keep alive ping", ["keep_alive_task"], "Keep-alive"),
            ("agent submission headers schema", ["submit_agent", "run_agent"], "Agent creation"),
        ],
        3: [
            ("prevent cloud server from sleeping", ["keep_alive_task"], "Conceptual keep-alive"),
            ("new user registers email password", ["register", "login"], "Registration"),
            ("Supabase clients configured roles", ["Supabase_config"], "Config search"),
            ("GitHub username deduplication profile", ["github_verify"], "GitHub username"),
            ("error handling 404 500 page", ["error_handlers"], "Error handlers"),
        ],
        4: [
            ("routes require login authentication", ["dashboard", "edit_profile", "submit_agent", "create_api_key", "memory"], "Auth routes"),
            ("OAuth social login providers", ["login_google", "auth_callback", "github_verify", "login"], "Auth providers"),
            ("Supabase database operations tables", ["register", "dashboard", "login", "create_api_key", "Supabase_config"], "DB operations"),
            ("WebSocket real-time features", ["SocketIO_MCP_setup"], "Real-time"),
            ("parse string utility regex", ["parse_to_dict"], "Utility function"),
        ],
        5: [
            ("authentication flow registration OAuth session", ["register", "login", "login_google", "auth_callback", "github_verify", "User"], "Full auth pipeline"),
            ("marketplace agent lifecycle submission execution", ["submit_agent", "run_agent", "agent_detail", "explore"], "Agent lifecycle"),
            ("email notification waitlist signups", ["join_waitlist", "join_gitmem_waitlist"], "Email notifications"),
            ("security password validation API key hashing token", ["register", "create_api_key", "login"], "Security features"),
            ("homepage landing page route", ["homepage"], "Main route"),
        ],
    }
    
    cases = test_cases.get(iteration, test_cases[1])
    passed = 0
    failed = 0
    total_score = 0
    
    for i, (query, expected_names, description) in enumerate(cases, 1):
        log_lines.append(f"\n  Test {iteration}.{i}: {description}")
        log_lines.append(f"  Query: \"{query}\"")
        log_lines.append(f"  Expected: {expected_names}")
        
        start = time.time()
        result = coding_api.get_mem(AGENT_ID, query)
        elapsed = time.time() - start
        
        verdict = ai_verify_relevance(query, result, expected_names)
        
        status = "PASS" if verdict["pass"] else "FAIL"
        log_lines.append(f"  {status} | Score: {verdict['score']}% | Time: {elapsed:.3f}s")
        log_lines.append(f"  Retrieved: {verdict['all_retrieved_names']}")
        log_lines.append(f"  Scores: {verdict.get('scores', [])}")
        log_lines.append(f"  Found: {verdict['found']}")
        if verdict['missed']:
            log_lines.append(f"  Missed: {verdict['missed']}")
        
        if verdict["pass"]:
            passed += 1
        else:
            failed += 1
        total_score += verdict["score"]
    
    avg_score = total_score / len(cases) if cases else 0
    log_lines.append(f"\n  ITERATION {iteration} RESULTS: {passed} passed, {failed} failed, Avg Score: {avg_score:.1f}%")
    
    return {"passed": passed, "failed": failed, "avg_score": avg_score}


def main():
    global log
    log_lines = []
    
    log_lines.append("=" * 70)
    log_lines.append("  DIRECT TEST: create_mem & get_mem from CodingAPI")
    log_lines.append("  Testing with large coding context from index.py")
    log_lines.append("=" * 70)
    
    # Clean up previous test data
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
        log_lines.append(f"\nCleaned up previous test data at {TEST_DATA_DIR}")
    
    # Initialize CodingAPI
    log_lines.append(f"\nInitializing CodingAPI at: {TEST_DATA_DIR}")
    coding_api = CodingAPI(root_path=TEST_DATA_DIR)
    log_lines.append("CodingAPI initialized successfully")
    
    # STEP 1: Create Flow
    log_lines.append(f"\n{'='*70}")
    log_lines.append(f"  STEP 1: create_mem - Storing {len(INDEX_PY_CHUNKS)} semantic chunks")
    log_lines.append(f"{'='*70}")
    
    start = time.time()
    create_result = coding_api.create_mem(
        agent_id=AGENT_ID,
        file_path=FILE_PATH,
        chunks=INDEX_PY_CHUNKS
    )
    create_time = time.time() - start
    
    log_lines.append(f"\ncreate_mem result: {json.dumps(create_result, indent=2)}")
    log_lines.append(f"Time: {create_time:.3f}s")
    
    status = create_result.get("status", "")
    if "error" in str(status).lower():
        log_lines.append("CRITICAL: create_mem failed! Aborting tests.")
        # Write log and exit
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print("\n".join(log_lines))
        return
    
    chunks_stored = create_result.get("chunks_processed", create_result.get("total_chunks", 0))
    log_lines.append(f"Successfully stored {chunks_stored} chunks")
    
    # STEP 2: Verify with list_mems
    log_lines.append(f"\n{'='*70}")
    log_lines.append("  STEP 2: Verify via list_mems")
    log_lines.append(f"{'='*70}")
    
    list_result = coding_api.list_mems(AGENT_ID)
    log_lines.append(f"Stored flows: {json.dumps(list_result, indent=2)}")
    
    # STEP 3: Run 5 test iterations
    overall_results = []
    for iteration in range(1, 6):
        result = run_test_iteration(coding_api, iteration, log_lines)
        overall_results.append(result)
    
    # FINAL SUMMARY
    log_lines.append(f"\n\n{'='*70}")
    log_lines.append("  FINAL TEST SUMMARY - 5 ITERATIONS")
    log_lines.append(f"{'='*70}")
    
    total_passed = sum(r["passed"] for r in overall_results)
    total_failed = sum(r["failed"] for r in overall_results)
    total_tests = total_passed + total_failed
    
    for i, r in enumerate(overall_results, 1):
        iter_total = r["passed"] + r["failed"]
        log_lines.append(f"  Iteration {i}: {r['passed']}/{iter_total} passed | Avg Score: {r['avg_score']:.1f}%")
    
    overall_score = sum(r["avg_score"] for r in overall_results) / len(overall_results)
    log_lines.append(f"\n  Overall: {total_passed}/{total_tests} tests passed")
    log_lines.append(f"  Overall Average Relevance Score: {overall_score:.1f}%")
    
    # AI Verdict
    if overall_score >= 60:
        log_lines.append("\n  AI VERDICT: EXCELLENT - Code Flow system is working well!")
    elif overall_score >= 40:
        log_lines.append("\n  AI VERDICT: ACCEPTABLE - Some retrieval gaps detected.")
    else:
        log_lines.append("\n  AI VERDICT: NEEDS IMPROVEMENT - Retrieval accuracy is low.")
    
    log_lines.append(f"\n  Test data at: {TEST_DATA_DIR}")
    
    # Write all output
    full_output = "\n".join(log_lines)
    
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(full_output)
    
    print(full_output)
    print(f"\nFull results saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
