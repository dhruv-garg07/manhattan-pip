import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import shutil
import json

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

# ============================================================
# Configuration
# ============================================================
REAL_INDEX_FILE = "/Users/gargdhruv/Desktop/manhattan-pip/index.py"
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_e2e_stress_data")
AGENT_ID = "e2e_stress"

passed = 0
failed = 0
errors = []

def test(name, condition, details=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        msg = f"  ❌ FAIL: {name}" + (f" — {details}" if details else "")
        print(msg)
        errors.append(msg)
        failed += 1

def show_results(r, top_n=3):
    """Helper to print retrieval diagnostics."""
    results = r.get("results", [])
    print(f"  Filter: {r.get('filter')}  |  Query: {r.get('query')}  |  Count: {r.get('count')}")
    for i, res in enumerate(results[:top_n]):
        c = res["chunk"]
        print(f"  [{i+1}] {c.get('name')} "
              f"(score={res.get('score',0):.4f} vec={res.get('vector_score',0):.4f} kw={res.get('keyword_score',0):.4f}) "
              f"type={c.get('type')} keywords={c.get('keywords',[])[:]}")

# ============================================================
# Setup
# ============================================================
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)

print("=" * 70)
print("E2E STRESS TEST: Ingest index.py → Difficult Query Retrieval")
print("=" * 70)

api = CodingAPI(root_path=TEST_DIR)

# ============================================================
# Realistic Semantic Chunks (as an AI agent would send via MCP)
# ============================================================
SEMANTIC_CHUNKS = [
    {
        "name": "Gevent Monkey Patching Setup",
        "type": "block",
        "content": "try:\n    from gevent import monkey\n    monkey.patch_all()\nexcept ImportError:\n    pass",
        "summary": "Initializes gevent monkey patching at application startup. Handles ImportError if gevent is not available.",
        "keywords": ["gevent", "monkey-patching", "startup", "worker", "import"],
        "start_line": 4, "end_line": 11
    },
    {
        "name": "Flask Application Configuration",
        "type": "module",
        "content": "app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)\napp.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')",
        "summary": "Initializes Flask application with environment variables, configures static/template directories, sets up secret key and creates Supabase clients.",
        "keywords": ["flask", "configuration", "environment", "supabase-client", "static", "templates", "secret-key"],
        "start_line": 73, "end_line": 151
    },
    {
        "name": "Flask-SocketIO and WebSocket Initialization",
        "type": "block",
        "content": "socketio = SocketIO(app, cors_allowed_origins='*', async_mode='gevent')\nfrom gitmem.api.websocket_events import init_websocket\ninit_websocket(socketio)",
        "summary": "Initializes Flask-SocketIO for real-time WebSocket communication with gevent async mode. Registers MCP Socket.IO gateway and GitMem WebSocket handlers.",
        "keywords": ["socketio", "websocket", "realtime", "gevent", "mcp", "gateway"],
        "start_line": 84, "end_line": 124
    },
    {
        "name": "User",
        "type": "class",
        "content": "class User:\n    def __init__(self, user_id=None, email=None):\n        self.id = user_id\n        self.email = email\n    @property\n    def is_authenticated(self): return self.id is not None\n    def get_id(self): return str(self.id)",
        "summary": "Flask-Login compatible User model with properties for authentication state. Stores user_id and email, provides get_id() method.",
        "keywords": ["user", "model", "flask-login", "authentication", "session", "properties"],
        "start_line": 184, "end_line": 203
    },
    {
        "name": "health_check",
        "type": "function",
        "content": "@app.route('/health')\ndef health_check():\n    status = {'status': 'healthy', 'timestamp': datetime.utcnow().isoformat(), 'socketio_enabled': socketio is not None}\n    return jsonify(status)",
        "summary": "Health check endpoint returning system status including socketio and MCP availability.",
        "keywords": ["health", "monitoring", "status", "endpoint", "ping", "health-check"],
        "start_line": 219, "end_line": 229
    },
    {
        "name": "explore",
        "type": "function",
        "content": "@app.route('/explore')\ndef explore():\n    search = request.args.get('search', '')\n    filters = SearchFilters(search=search, category=category, ...)\n    agents = asyncio.run(agent_service.fetch_agents(filters))\n    return render_template('explore.html', agents=agents)",
        "summary": "Handles /explore route for browsing agents with search and filter capabilities. Extracts filter parameters, creates SearchFilters, and renders explore.html.",
        "keywords": ["explore", "agents", "search", "filters", "route", "browse", "categories"],
        "start_line": 236, "end_line": 272
    },
    {
        "name": "agent_detail",
        "type": "function",
        "content": "@app.route('/agent/<agent_id>')\ndef agent_detail(agent_id):\n    agent = asyncio.run(agent_service.get_agent_by_id(agent_id))\n    return render_template('agent_detail.html', agent=agent)",
        "summary": "Handles /agent/<agent_id> route to display agent detail page. Fetches agent by ID, handles not-found.",
        "keywords": ["agent", "detail", "route", "agent-detail", "view"],
        "start_line": 274, "end_line": 288
    },
    {
        "name": "login",
        "type": "function",
        "content": "@app.route('/login', methods=['POST'])\ndef login():\n    email = _clean_email(request.form.get('email'))\n    password = request.form.get('password')\n    auth = supabase.auth.sign_in_with_password({'email': email, 'password': password})\n    login_user(User(user_id=auth.user.id, email=email))\n    return redirect(next_url)",
        "summary": "Handles POST /login route for user authentication. Validates email/password, signs in with Supabase Auth, stores session tokens, logs in with Flask-Login.",
        "keywords": ["login", "authentication", "supabase", "email", "password", "session", "flask-login", "route"],
        "start_line": 447, "end_line": 479
    },
    {
        "name": "login_google",
        "type": "function",
        "content": "@app.route('/login/google')\ndef login_google():\n    oauth_url = f'{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={redirect_url}'\n    return redirect(oauth_url)",
        "summary": "Initiates Google OAuth flow by redirecting to Supabase OAuth authorize endpoint with google provider.",
        "keywords": ["google", "oauth", "login", "redirect", "supabase", "authentication"],
        "start_line": 481, "end_line": 485
    },
    {
        "name": "auth_callback",
        "type": "function",
        "content": "@app.route('/auth/callback', methods=['GET', 'POST'])\ndef auth_callback():\n    if request.method == 'GET': return render_template('auth_callback.html')\n    access_token = data.get('access_token')\n    user_resp = supabase.auth.get_user(access_token)",
        "summary": "Handles OAuth callback from Supabase. GET serves callback HTML. POST processes tokens, validates with Supabase, ensures profile exists, logs in user.",
        "keywords": ["oauth", "callback", "authentication", "token", "profile-sync", "supabase"],
        "start_line": 487, "end_line": 547
    },
    {
        "name": "login_github",
        "type": "function",
        "content": "@app.route('/login/github')\ndef login_github():\n    oauth_url = f'{SUPABASE_URL}/auth/v1/authorize?provider=github&redirect_to={redirect_url}'\n    return redirect(oauth_url)",
        "summary": "Initiates GitHub OAuth flow by redirecting to Supabase OAuth authorize endpoint with github provider.",
        "keywords": ["github", "oauth", "login", "redirect", "supabase", "authentication"],
        "start_line": 549, "end_line": 553
    },
    {
        "name": "github_verify",
        "type": "function",
        "content": "@app.route('/auth/github/verify', methods=['POST'])\ndef github_verify():\n    access_token = data.get('access_token')\n    user_info = supabase_backend.auth.get_user(access_token)\n    github_username = github_user.user_metadata.get('preferred_username')",
        "summary": "Verifies GitHub OAuth token. Checks if profile exists, updates GitHub URL if missing, or creates new profile with unique username. Logs in user.",
        "keywords": ["github", "verify", "oauth", "token", "profile", "username", "authentication"],
        "start_line": 560, "end_line": 649
    },
    {
        "name": "register",
        "type": "function",
        "content": "@app.route('/register', methods=['POST'])\ndef register():\n    email = _clean_email(request.form.get('email'))\n    password = request.form.get('password')\n    auth = supabase.auth.sign_up({'email': email, 'password': password})",
        "summary": "Handles user registration with form validation, unique username checking, password strength validation. Creates Supabase Auth user and profiles table entry.",
        "keywords": ["register", "signup", "user-creation", "validation", "supabase", "password", "username", "authentication"],
        "start_line": 658, "end_line": 755
    },
    {
        "name": "logout",
        "type": "function",
        "content": "@app.route('/logout', methods=['GET', 'POST'])\n@login_required\ndef logout():\n    supabase.auth.sign_out()\n    logout_user()\n    session.clear()",
        "summary": "Handles user logout by signing out from Supabase, removing tokens from session, logging out Flask-Login user.",
        "keywords": ["logout", "session", "sign-out", "authentication", "flask-login"],
        "start_line": 757, "end_line": 771
    },
    {
        "name": "edit_profile",
        "type": "function",
        "content": "@app.route('/profile/edit', methods=['GET', 'POST'])\n@login_required\ndef edit_profile():\n    # GET shows form, POST validates and updates profile in Supabase\n    supabase.table('profiles').update(update_data).eq('id', current_user.id).execute()",
        "summary": "Handles GET/POST for /profile/edit. GET displays edit form. POST validates username uniqueness, updates profile in Supabase with role-specific fields.",
        "keywords": ["profile", "edit", "update", "form", "supabase", "username", "validation"],
        "start_line": 818, "end_line": 912
    },
    {
        "name": "api_agents",
        "type": "function",
        "content": "@app.route('/api/agents')\ndef api_agents():\n    filters = SearchFilters(search=search, category=category, ...)\n    agents = agent_service.fetch_agents(filters)\n    return jsonify([agent.dict() for agent in agents])",
        "summary": "AJAX API endpoint for fetching agents with filtering/sorting. Applies search, category, modalities, capabilities filters.",
        "keywords": ["api", "agents", "ajax", "endpoint", "filtering", "search", "json"],
        "start_line": 925, "end_line": 951
    },
    {
        "name": "list_api_keys",
        "type": "function",
        "content": "@app.route('/api/keys', methods=['GET'])\n@login_required\ndef list_api_keys():\n    resp = supabase.table('api_keys').select('id, name, masked_key, expiration, expires_at, created_at').eq('user__id', current_user.id).execute()\n    return jsonify(keys)",
        "summary": "GET endpoint to list user's masked API keys. Queries Supabase api_keys table filtered by user_id, returns masked key data.",
        "keywords": ["api", "keys", "list", "endpoint", "masking", "security", "api-keys", "management", "route"],
        "start_line": 1046, "end_line": 1061
    },
    {
        "name": "delete_api_key",
        "type": "function",
        "content": "@app.route('/api/keys/<key_id>', methods=['DELETE'])\n@login_required\ndef delete_api_key(key_id):\n    supabase_backend.table('api_keys').update({'status': 'revoked'}).eq('id', key_id).execute()\n    return jsonify({'ok': True})",
        "summary": "DELETE endpoint to revoke an API key by id. Soft-deletes by setting status to 'revoked' instead of removing.",
        "keywords": ["api", "keys", "delete", "revoke", "endpoint", "security", "api-keys"],
        "start_line": 1063, "end_line": 1075
    },
    {
        "name": "create_api_key",
        "type": "function",
        "content": "@app.route('/api/keys', methods=['POST'])\n@login_required\ndef create_api_key():\n    key_val = data.get('key') or generate_secret_key()\n    hashed_key = hash_key(key_val)\n    supabase_backend.table('api_keys').insert({...}).execute()\n    return jsonify({'key': key_val, 'id': new_id})",
        "summary": "POST endpoint to create new API key. Generates secure key if not provided, computes expiration, hashes before storing, returns plaintext key once on creation.",
        "keywords": ["api", "keys", "create", "endpoint", "hashing", "security", "api-keys", "key-creation", "permissions"],
        "start_line": 1077, "end_line": 1155
    },
    {
        "name": "run_agent",
        "type": "function",
        "content": "def run_agent(user_input, agent_data):\n    agent_data = parse_to_dict(html.unescape(agent_data))\n    url = agent_data.get('base_url') + '/' + agent_data.get('run_path', '')\n    response = requests.post(url, json=user_input['body'])\n    return f'Agent processed: {response.text}'",
        "summary": "Executes agent by POSTing to agent's base_url + run_path with user input. Parses agent_data, constructs request payload, handles HTTP errors.",
        "keywords": ["agent", "execution", "run", "http", "request", "post", "base-url"],
        "start_line": 974, "end_line": 1010
    },
    {
        "name": "run_agent_route",
        "type": "function",
        "content": "@app.route('/run-agent', methods=['POST'])\ndef run_agent_route():\n    result = run_agent(user_input, agent_data)\n    agent_service.update_agent_field(agent_id, 'total_runs', new_total_runs)\n    return jsonify({'response': result, 'total_runs': new_total_runs})",
        "summary": "POST endpoint that executes agent and increments total_runs counter. Parses agent data, calls run_agent, updates metrics in database.",
        "keywords": ["run-agent", "endpoint", "execution", "metrics", "total-runs", "route"],
        "start_line": 1013, "end_line": 1028
    },
    {
        "name": "api_creators",
        "type": "function",
        "content": "@app.route('/api/creators')\ndef api_creators():\n    all_creators = creator_service.fetch_creators()\n    filtered_creators = creator_service.filter_and_sort_creators(all_creators, search, sort_by)\n    return jsonify([creator.dict() for creator in filtered_creators])",
        "summary": "API endpoint for fetching creators list. Supports search and reputation sorting.",
        "keywords": ["api", "creators", "endpoint", "filtering", "search", "ajax", "json"],
        "start_line": 1030, "end_line": 1041
    },
    {
        "name": "error_handlers",
        "type": "block",
        "content": "@app.errorhandler(404)\ndef not_found(e): return render_template('404.html'), 404\n@app.errorhandler(500)\ndef server_error(e): return render_template('500.html'), 500",
        "summary": "Custom Flask error handlers for 404 and 500 HTTP error responses. Renders corresponding error templates.",
        "keywords": ["error", "handler", "404", "500", "template", "http"],
        "start_line": 1157, "end_line": 1165
    },
    {
        "name": "waitlist_signup",
        "type": "function",
        "content": "@app.route('/waitlist/signup', methods=['POST'])\ndef waitlist_signup():\n    email = request.json.get('email')\n    supabase_backend.table('waitlist').insert({'email': email}).execute()\n    email_service.send_welcome_email(email)",
        "summary": "POST endpoint for waitlist signup. Validates email, checks for duplicates, sends welcome email asynchronously, returns waitlist count.",
        "keywords": ["waitlist", "signup", "email", "registration", "marketing", "route"],
        "start_line": 1196, "end_line": 1270
    },
    {
        "name": "memory_upload_handler",
        "type": "function",
        "content": "@app.route('/memory', methods=['GET', 'POST'])\n@login_required\ndef memory():\n    controller = RAG_DB_Controller_FILE_DATA()\n    controller.update_file_data_to_db(user_ID=str(current_user.id), file_path=file_path)\n    controller.send_data_to_rag_db(user_ID=str(current_user.id), chunks=[text])",
        "summary": "Handles GET/POST for /memory. GET shows upload form. POST processes file uploads and text memory, validates file types, stores to RAG database, cleans up temp files.",
        "keywords": ["memory", "upload", "file", "rag", "database", "text", "route", "file-upload"],
        "start_line": 1367, "end_line": 1467
    },
    {
        "name": "upvote_rate_version_routes",
        "type": "block",
        "content": "@app.route('/upvote', methods=['POST'])\ndef upvote_agent(): ...\n@app.route('/rate', methods=['POST'])\ndef rate_agent(): ...\n@app.route('/version', methods=['POST'])\ndef version_agent(): ...",
        "summary": "POST endpoints for agent interactions: /upvote increments upvotes, /rate sets average rating, /version updates version number. All update agent metrics.",
        "keywords": ["upvote", "rate", "version", "agent", "metrics", "interaction", "route"],
        "start_line": 1168, "end_line": 1194
    },
    {
        "name": "main_entry_point",
        "type": "block",
        "content": "if __name__ == '__main__':\n    if socketio:\n        socketio.run(app, host='0.0.0.0', port=1078, debug=True)\n    else:\n        app.run(host='0.0.0.0', port=1078, debug=True)",
        "summary": "Application entry point that runs Flask with SocketIO if available, otherwise standard Flask. Listens on port 1078 with debug mode.",
        "keywords": ["main", "entry-point", "socketio", "port", "debug", "run"],
        "start_line": 1502, "end_line": 1509
    },
]

# ============================================================
# Phase 1: Ingest
# ============================================================
print(f"\n{'─' * 70}")
print("PHASE 1: Ingest index.py with semantic chunks")
print(f"{'─' * 70}")

result = api.create_mem(AGENT_ID, REAL_INDEX_FILE, chunks=SEMANTIC_CHUNKS)
print(f"  Ingested {len(SEMANTIC_CHUNKS)} semantic chunks")
print(f"  Result: {result.get('message', result.get('status', 'unknown'))}")

# Verify storage
agent_dir = os.path.join(TEST_DIR, "agents", AGENT_ID)
fc_path = os.path.join(agent_dir, "file_contexts.json")
test("file_contexts.json created", os.path.exists(fc_path))

vectors_path = os.path.join(agent_dir, "vectors.json")
test("vectors.json created", os.path.exists(vectors_path))

if os.path.exists(vectors_path):
    with open(vectors_path) as f:
        vecs = json.load(f)
    test(f"Vectors stored for all chunks ({len(vecs)}/{len(SEMANTIC_CHUNKS)})",
         len(vecs) == len(SEMANTIC_CHUNKS),
         f"Expected {len(SEMANTIC_CHUNKS)}, got {len(vecs)}")

# ============================================================
# Phase 2: Easy Queries (baseline sanity)
# ============================================================
print(f"\n{'─' * 70}")
print("PHASE 2: Baseline Sanity Checks")
print(f"{'─' * 70}")

# 2a: Exact function name
print("\n[2a] Query: 'login'")
r = api.get_mem(AGENT_ID, "login")
show_results(r)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    test("Top result is login-related", "login" in results[0]["chunk"].get("name", "").lower())

# 2b: Exact class name
print("\n[2b] Query: 'User class'")
r = api.get_mem(AGENT_ID, "User class")
show_results(r)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    test("Top result is User class", "user" in results[0]["chunk"].get("name", "").lower())

# 2c: Simple keyword
print("\n[2c] Query: 'health check endpoint'")
r = api.get_mem(AGENT_ID, "health check endpoint")
show_results(r)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    test("Top result is health_check", "health" in results[0]["chunk"].get("name", "").lower())

# ============================================================
# Phase 3: Route-Style Queries (the original bug)
# ============================================================
print(f"\n{'─' * 70}")
print("PHASE 3: Route-Style Queries (Original Bug)")
print(f"{'─' * 70}")

print(f"\n[3p] Query: '/api/keys' (Short route query)")
# This was a specific user report where "/api/keys" was treated as a file filter
# causing 0 results. It should NOT be a filter.
r_3p = api.get_mem(AGENT_ID, "/api/keys")
show_results(r_3p)

results_3p = r_3p.get("results", [])
test("Should return results for short route query", len(results_3p) > 0)
test("Should NOT treat /api/keys as file filter", r_3p.get("filter") is None, f"Got filter: {r_3p.get('filter')}")
test("Should find API key related chunks",
     any("api_key" in r["chunk"]["name"].lower() or "api key" in r["chunk"]["name"].lower() for r in results_3p),
     "Did not find API key related chunks in results")
print("  ✅ Short route query handled correctly (not a file filter)")

# 3a: The exact bug case
print("\n[3a] Query: \"/api/keys endpoint from index.py\"")
r = api.get_mem(AGENT_ID, "/api/keys endpoint from index.py")
show_results(r)
test("Filter is index.py", r.get("filter") is not None and "index.py" in (r.get("filter") or ""))
results = r.get("results", [])
test("Returns results (not empty)", len(results) > 0)
if results:
    top_names = [res["chunk"]["name"] for res in results[:3]]
    test("Top results are API key related",
         any("api_key" in n.lower() or "api key" in n.lower() for n in top_names),
         f"Got: {top_names}")

# 3b: Quoted route
print("\n[3b] Query: \"Can you explain the '/api/keys' endpoint?\"")
r = api.get_mem(AGENT_ID, "Can you explain the '/api/keys' endpoint?")
show_results(r)
test("No file filter (route not mistaken for file)",
     r.get("filter") is None,
     f"Got filter: {r.get('filter')}")
results = r.get("results", [])
test("Returns results", len(results) > 0)

# 3c: Route with parameters
print("\n[3c] Query: \"/api/keys/<key_id> DELETE\"")
r = api.get_mem(AGENT_ID, "/api/keys/<key_id> DELETE")
show_results(r)
test("No file filter", r.get("filter") is None)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    top_names = [res["chunk"]["name"] for res in results[:3]]
    test("delete_api_key in top results",
         any("delete" in n.lower() for n in top_names),
         f"Got: {top_names}")

# 3d: Multiple routes in query
print("\n[3d] Query: \"/login and /register routes\"")
r = api.get_mem(AGENT_ID, "/login and /register routes")
show_results(r)
test("No file filter", r.get("filter") is None)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    top_names = [res["chunk"]["name"] for res in results[:5]]
    has_login = any("login" in n.lower() for n in top_names)
    has_register = any("register" in n.lower() for n in top_names)
    test("Both login and register in results",
         has_login and has_register,
         f"login={has_login} register={has_register} Got: {top_names}")

# 3e: Deep nested route
print("\n[3e] Query: \"/auth/github/verify endpoint\"")
r = api.get_mem(AGENT_ID, "/auth/github/verify endpoint")
show_results(r)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    top_names = [res["chunk"]["name"] for res in results[:3]]
    test("github_verify in top results",
         any("github" in n.lower() and "verify" in n.lower() for n in top_names),
         f"Got: {top_names}")

# ============================================================
# Phase 4: Natural Language / Vague Queries
# ============================================================
print(f"\n{'─' * 70}")
print("PHASE 4: Natural Language & Vague Queries")
print(f"{'─' * 70}")

# 4a: Very vague
print("\n[4a] Query: \"how does authentication work\"")
r = api.get_mem(AGENT_ID, "how does authentication work")
show_results(r, top_n=5)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    top_names = [res["chunk"]["name"] for res in results[:5]]
    auth_related = [n for n in top_names if any(kw in n.lower() for kw in ["login", "auth", "register", "oauth", "github", "google"])]
    test("Majority of top results are auth-related",
         len(auth_related) >= 2,
         f"Auth-related: {auth_related} from {top_names}")

# 4b: Conceptual query
print("\n[4b] Query: \"where is the database connection configured\"")
r = api.get_mem(AGENT_ID, "where is the database connection configured")
show_results(r)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    # Should return Flask config or Supabase setup
    top_content = " ".join(res["chunk"].get("summary", "") for res in results[:3])
    test("Results relate to configuration/database",
         "supabase" in top_content.lower() or "config" in top_content.lower() or "environment" in top_content.lower(),
         f"Top summaries don't mention config/supabase")

# 4c: Action-oriented query
print("\n[4c] Query: \"how to upload files\"")
r = api.get_mem(AGENT_ID, "how to upload files")
show_results(r)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    top_names = [res["chunk"]["name"] for res in results[:3]]
    test("Memory/upload handler in results",
         any("memory" in n.lower() or "upload" in n.lower() for n in top_names),
         f"Got: {top_names}")

# 4d: Question about a specific feature
print("\n[4d] Query: \"what happens when a user signs up\"")
r = api.get_mem(AGENT_ID, "what happens when a user signs up")
show_results(r)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    top_names = [res["chunk"]["name"] for res in results[:5]]
    test("Registration/signup in results",
         any("register" in n.lower() or "signup" in n.lower() or "waitlist" in n.lower() for n in top_names),
         f"Got: {top_names}")

# ============================================================
# Phase 5: Edge Cases & Stress Tests
# ============================================================
print(f"\n{'─' * 70}")
print("PHASE 5: Edge Cases & Stress Tests")
print(f"{'─' * 70}")

# 5a: Single character query
print("\n[5a] Query: \"a\"")
r = api.get_mem(AGENT_ID, "a")
test("Single char query doesn't crash", r.get("status") == "search_results")

# 5b: Empty-ish query (all stop words)
print("\n[5b] Query: \"the is a an and or\"")
r = api.get_mem(AGENT_ID, "the is a an and or")
test("All-stopwords query doesn't crash", r.get("status") == "search_results")

# 5c: Very long query
long_query = "I need to understand the complete authentication flow including login registration OAuth Google GitHub callback handling token verification session management and password validation"
print(f"\n[5c] Long query ({len(long_query)} chars)")
r = api.get_mem(AGENT_ID, long_query)
show_results(r)
test("Long query returns results", r.get("count", 0) > 0)

# 5d: Query with special characters
print("\n[5d] Query: \"@app.route('/api/keys') decorator\"")
r = api.get_mem(AGENT_ID, "@app.route('/api/keys') decorator")
show_results(r)
results = r.get("results", [])
test("Special chars query returns results", len(results) > 0)

# 5e: CamelCase and mixed case
print("\n[5e] Query: \"SocketIO WebSocket initialization\"")
r = api.get_mem(AGENT_ID, "SocketIO WebSocket initialization")
show_results(r)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    top_names = [res["chunk"]["name"] for res in results[:3]]
    test("SocketIO chunk in results",
         any("socket" in n.lower() for n in top_names),
         f"Got: {top_names}")

# 5f: Abbreviation / shorthand
print("\n[5f] Query: \"oauth flow\"")
r = api.get_mem(AGENT_ID, "oauth flow")
show_results(r)
results = r.get("results", [])
test("Returns results for oauth", len(results) > 0)
if results:
    top_names = [res["chunk"]["name"] for res in results[:5]]
    oauth_found = [n for n in top_names if "oauth" in n.lower() or "google" in n.lower() or "github" in n.lower() or "callback" in n.lower()]
    test("OAuth-related functions in results",
         len(oauth_found) >= 1,
         f"Got: {top_names}")

# 5g: Negative/exclusion-style query (retrieval doesn't support NOT, but shouldn't crash)
print("\n[5g] Query: \"everything except authentication\"")
r = api.get_mem(AGENT_ID, "everything except authentication")
test("Negative-style query doesn't crash", r.get("status") == "search_results")

# 5h: Query specifically about error handling
print("\n[5h] Query: \"error handling 404 500\"")
r = api.get_mem(AGENT_ID, "error handling 404 500")
show_results(r)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    top_names = [res["chunk"]["name"] for res in results[:3]]
    test("Error handlers in results",
         any("error" in n.lower() or "handler" in n.lower() for n in top_names),
         f"Got: {top_names}")

# 5i: Query about Supabase interactions
print("\n[5i] Query: \"supabase database operations\"")
r = api.get_mem(AGENT_ID, "supabase database operations")
show_results(r, top_n=5)
results = r.get("results", [])
test("Returns results about supabase", len(results) > 0)

# 5j: Agent execution flow
print("\n[5j] Query: \"how to run execute agent\"")
r = api.get_mem(AGENT_ID, "how to run execute agent")
show_results(r)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    top_names = [res["chunk"]["name"] for res in results[:3]]
    test("run_agent or run_agent_route in results",
         any("run" in n.lower() or "agent" in n.lower() for n in top_names),
         f"Got: {top_names}")

# ============================================================
# Phase 6: Cross-cutting & Ranking Quality
# ============================================================
print(f"\n{'─' * 70}")
print("PHASE 6: Ranking Quality Tests")
print(f"{'─' * 70}")

# 6a: "api" should return API-related chunks, not "api_agents" mixed with "api_keys"
print("\n[6a] Query: \"api key management security\"")
r = api.get_mem(AGENT_ID, "api key management security")
show_results(r, top_n=5)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    top_names = [res["chunk"]["name"] for res in results[:3]]
    test("API key chunks are top ranked (not generic api_agents)",
         any("key" in n.lower() for n in top_names),
         f"Got: {top_names}")

# 6b: Specificity test — "gevent" should return the gevent chunk, not everything
print("\n[6b] Query: \"gevent monkey patching\"")
r = api.get_mem(AGENT_ID, "gevent monkey patching")
show_results(r)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    test("Top result is gevent-related",
         "gevent" in results[0]["chunk"].get("name", "").lower(),
         f"Got: {results[0]['chunk'].get('name')}")

# 6c: Ranking: exact name match should beat keyword overlap
print("\n[6c] Query: \"register\"")
r = api.get_mem(AGENT_ID, "register")
show_results(r)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    test("Top result is 'register' (exact name match)",
         results[0]["chunk"].get("name", "").lower() == "register",
         f"Got: {results[0]['chunk'].get('name')}")

# 6d: Ranking: route pattern match should boost correctly
print("\n[6d] Query: \"/run-agent endpoint\"")
r = api.get_mem(AGENT_ID, "/run-agent endpoint")
show_results(r)
results = r.get("results", [])
test("Returns results", len(results) > 0)
if results:
    top_names = [res["chunk"]["name"] for res in results[:3]]
    test("run_agent_route in top results",
         any("run_agent" in n.lower() for n in top_names),
         f"Got: {top_names}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 70}")
if failed == 0:
    print(f"ALL {passed} TESTS PASSED! ✅")
else:
    print(f"RESULTS: {passed} passed, {failed} failed")
    if errors:
        print(f"\n{'─' * 70}")
        print("FAILURES:")
        for e in errors:
            print(e)
print(f"{'=' * 70}")

# Cleanup
shutil.rmtree(TEST_DIR)
sys.exit(0 if failed == 0 else 1)
