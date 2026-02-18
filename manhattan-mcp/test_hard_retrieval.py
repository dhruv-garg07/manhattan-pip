"""
HARD Retrieval Test Suite for CodingAPI create_mem & get_mem
================================================================
Designed to expose weaknesses in:
  - Synonym/paraphrase handling
  - Negation & exclusion queries
  - Cross-cutting concern retrieval
  - Abstract/conceptual queries
  - Multi-hop reasoning queries
  - Edge cases (single word, very long, typos)
  - Type-specific queries (find all classes, all functions)
  - Return value / parameter queries
  - Relationship/dependency queries

All calls go through CodingAPI directly — no MCP.
"""

import sys
import os
import json
import time
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "manhattan_mcp"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

# ============================================================================
# Config
# ============================================================================
AGENT_ID = "hard_test_agent"
FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index.py"))
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), ".hard_test_gitmem")
LOG_FILE = os.path.join(os.path.dirname(__file__), "hard_test_results.log")

# ============================================================================
# Rich, detailed chunks — testing if nuanced content is retrievable
# ============================================================================
HARD_CHUNKS = [
    {
        "name": "FlaskApp_init",
        "type": "module",
        "content": "from gevent import monkey; monkey.patch_all()\napp = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)\napp.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')\napp.config['SESSION_COOKIE_SECURE'] = True",
        "summary": "Flask application initialization with gevent monkey patching for async compatibility. Configures static/template directories, secret key from environment variables, and secure session cookies. Entry point of the entire web server.",
        "keywords": ["flask", "app", "initialization", "gevent", "monkey_patch", "secret_key", "session_cookie", "secure"],
        "start_line": 1, "end_line": 82
    },
    {
        "name": "SocketIO_MCP_setup",
        "type": "block",
        "content": "socketio = SocketIO(app, cors_allowed_origins='*', async_mode='gevent')\ninit_websocket(socketio)\ninit_mcp_socketio(socketio)\nmcp_sse_bp = Blueprint('mcp_sse', __name__)\napp.register_blueprint(mcp_sse_bp, url_prefix='/mcp')",
        "summary": "Flask-SocketIO initialization with CORS allowed for all origins and gevent async mode. Registers MCP SSE Blueprint at /mcp prefix for server-sent events transport. Background thread used for non-blocking MCP gateway initialization.",
        "keywords": ["socketio", "websocket", "mcp", "sse", "blueprint", "cors", "gevent", "real-time", "gateway"],
        "start_line": 84, "end_line": 124
    },
    {
        "name": "Supabase_config",
        "type": "block",
        "content": "SUPABASE_URL = os.environ.get('SUPABASE_URL')\nSUPABASE_ANON_KEY = os.environ.get('SUPABASE_ANON_KEY')\nSUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')\nsupabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)\nsupabase_backend = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)",
        "summary": "Supabase dual-client configuration. 'supabase' uses anon key for user-facing operations respecting Row Level Security (RLS). 'supabase_backend' uses service role key for admin operations that bypass RLS. All credential loaded from .env via python-dotenv.",
        "keywords": ["supabase", "config", "client", "anon_key", "service_role", "rls", "environment", "dotenv", "credentials", "database"],
        "start_line": 146, "end_line": 151
    },
    {
        "name": "keep_alive_task",
        "type": "function",
        "content": "def keep_alive_task():\n    def ping_website():\n        while True:\n            requests.get('https://themanhattanproject.ai/ping')\n            time.sleep(300)\n    threading.Thread(target=ping_website, daemon=True).start()",
        "summary": "Background daemon thread that pings https://themanhattanproject.ai/ping every 300 seconds (5 minutes) to prevent Render cloud hosting from putting the server to sleep due to inactivity. Uses threading.Thread with daemon=True so it doesn't block graceful shutdown.",
        "keywords": ["keep-alive", "background", "thread", "daemon", "ping", "render", "sleep", "health", "timer", "inactivity"],
        "start_line": 158, "end_line": 181
    },
    {
        "name": "User",
        "type": "class",
        "content": "class User:\n    def __init__(self, user_id=None, email=None):\n        self.id = user_id\n        self.email = email\n    @property\n    def is_authenticated(self): return self.id is not None\n    @property\n    def is_active(self): return True\n    @property\n    def is_anonymous(self): return self.id is None\n    def get_id(self): return str(self.id)",
        "summary": "User model class implementing Flask-Login interface. Stores user_id and email. Properties: is_authenticated returns True when user_id exists, is_active always True, is_anonymous True when no user_id. get_id returns string user_id for session serialization. Used by @login_manager.user_loader to reconstruct user from session.",
        "keywords": ["user", "model", "flask-login", "authentication", "user_id", "email", "session", "is_authenticated", "is_anonymous", "login_manager"],
        "start_line": 184, "end_line": 210
    },
    {
        "name": "explore",
        "type": "function",
        "content": "@app.route('/explore')\ndef explore():\n    filters = SearchFilters(\n        search=request.args.get('search'),\n        category=request.args.get('category'),\n        model=request.args.get('model'),\n        status=request.args.get('status'),\n        sort_by=request.args.get('sort_by', 'trending')\n    )\n    agents = asyncio.run(agent_service.fetch_agents(filters))\n    return render_template('explore.html', agents=agents)",
        "summary": "Explore/marketplace page route. Accepts URL query parameters: search, category, model, status, sort_by (default: trending), modalities, capabilities. Creates SearchFilters object and asynchronously fetches agents via agent_service.fetch_agents(). Renders explore.html template with agents list and current filter state for the sidebar.",
        "keywords": ["explore", "agents", "search", "filter", "category", "model", "route", "marketplace", "trending", "fetch_agents", "SearchFilters"],
        "start_line": 236, "end_line": 273
    },
    {
        "name": "agent_detail",
        "type": "function",
        "content": "@app.route('/agent/<agent_id>')\ndef agent_detail(agent_id):\n    agent = asyncio.run(agent_service.get_agent_by_id(agent_id))\n    if not agent:\n        flash('Agent not found', 'error')\n        return redirect(url_for('explore'))\n    return render_template('agent_detail.html', agent=agent)",
        "summary": "Agent detail page showing full information about a specific agent. Takes agent_id as URL path parameter. Fetches agent data asynchronously via agent_service.get_agent_by_id(). Redirects to explore page with error flash message if agent not found. Renders agent_detail.html with complete agent data object.",
        "keywords": ["agent", "detail", "route", "get_agent_by_id", "agent_id", "page", "redirect", "flash", "not_found"],
        "start_line": 274, "end_line": 289
    },
    {
        "name": "submit_agent",
        "type": "function",
        "content": "@app.route('/submit', methods=['POST'])\n@login_required\ndef submit_agent():\n    custom_headers = {}\n    header_keys = request.form.getlist('header_keys[]')\n    header_values = request.form.getlist('header_values[]')\n    for k, v in zip(header_keys, header_values):\n        if k.strip(): custom_headers[k.strip()] = v.strip()\n    agent_data = {'name': ..., 'io_schema': json.loads(...), 'tags': ...split(',')}\n    agent = asyncio.run(agent_service.create_agent(agent_data, current_user.id))",
        "summary": "Agent submission handler (POST, login required). Processes multipart form data: name, description, category, base_url, run_path, custom headers as parallel key-value arrays, content_type, authentication method, io_schema/out_schema parsed as JSON, tags split by comma. Sets defaults: status=pending, success_rate/total_runs/avg_rating/avg_latency/upvotes=0. Creates agent via agent_service.create_agent() with current_user.id as creator.",
        "keywords": ["submit", "agent", "create", "form", "headers", "schema", "json", "tags", "login_required", "creator_studio", "multipart"],
        "start_line": 334, "end_line": 411
    },
    {
        "name": "dashboard",
        "type": "function",
        "content": "@app.route('/dashboard')\n@login_required\ndef dashboard():\n    profile_res = supabase.table('profiles').select('*').eq('id', current_user.id).execute()\n    if profile_res.data:\n        profile = profile_res.data[0]\n    else:\n        profile = {'email': current_user.email, 'username': current_user.email.split('@')[0]}",
        "summary": "Dashboard page (login required). Fetches user profile from Supabase profiles table matching current_user.id. Features fallback profile construction from email if database record not found (handles sync issues). Renders dashboard.html with profile data and is_own_profile=True flag for edit controls.",
        "keywords": ["dashboard", "profile", "supabase", "login_required", "user", "fallback", "profiles_table", "is_own_profile"],
        "start_line": 420, "end_line": 441
    },
    {
        "name": "login",
        "type": "function",
        "content": "@app.route('/login', methods=['POST'])\ndef login():\n    email = request.form.get('email', '').strip().lower()\n    password = request.form.get('password')\n    auth = supabase.auth.sign_in_with_password({'email': email, 'password': password})\n    session['sb_access_token'] = auth.session.access_token\n    session['sb_refresh_token'] = auth.session.refresh_token\n    login_user(User(user_id=auth.user.id, email=email))",
        "summary": "POST login handler. Cleans email (strip+lowercase), validates credentials via Supabase Auth sign_in_with_password. Stores sb_access_token and sb_refresh_token in Flask session for subsequent authenticated requests. Creates User object with id and email, calls Flask-Login's login_user(). Redirects to 'next' URL parameter or homepage on success. Flashes error message on AuthApiError.",
        "keywords": ["login", "authentication", "supabase", "password", "token", "session", "flask-login", "sign_in_with_password", "access_token", "refresh_token", "email"],
        "start_line": 447, "end_line": 479
    },
    {
        "name": "login_google",
        "type": "function",
        "content": "@app.route('/login/google')\ndef login_google():\n    redirect_url = url_for('auth_callback', _external=True)\n    oauth_url = f'{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={redirect_url}'\n    return redirect(oauth_url)",
        "summary": "Google OAuth login initiation. Constructs Supabase OAuth authorization URL with provider=google and redirect_to pointing to auth_callback endpoint. Redirects browser to Google consent screen where user grants access.",
        "keywords": ["google", "oauth", "login", "supabase", "authorization", "redirect", "consent", "provider", "social_login"],
        "start_line": 481, "end_line": 485
    },
    {
        "name": "auth_callback",
        "type": "function",
        "content": "@app.route('/auth/callback', methods=['GET', 'POST'])\ndef auth_callback():\n    if request.method == 'GET':\n        return render_template('auth_callback.html')\n    data = request.get_json()\n    access_token = data.get('access_token')\n    user_info = supabase.auth.get_user(access_token)\n    existing = supabase_backend.table('profiles').select('*').eq('email', email).execute()\n    if not existing.data:\n        supabase_backend.table('profiles').upsert({...}).execute()",
        "summary": "OAuth callback handler with dual-phase flow. GET phase: serves auth_callback.html containing JavaScript that extracts access_token and refresh_token from URL hash fragment and POSTs them back. POST phase: receives tokens as JSON, validates via supabase.auth.get_user(), checks if profile exists in profiles table, upserts new profile with email/username/role via service role client (bypasses RLS), stores tokens in Flask session, creates User object, calls login_user().",
        "keywords": ["oauth", "callback", "google", "token", "validation", "profile", "upsert", "supabase", "hash_fragment", "javascript", "dual_phase"],
        "start_line": 487, "end_line": 547
    },
    {
        "name": "github_verify",
        "type": "function",
        "content": "@app.route('/auth/github/verify', methods=['POST'])\ndef github_verify():\n    access_token = request.get_json().get('access_token')\n    user_info = supabase_backend.auth.get_user(access_token)\n    username = user_metadata.get('user_name') or user_metadata.get('preferred_username')\n    while supabase_backend.table('profiles').select('id').eq('username', candidate).execute().data:\n        counter += 1\n        candidate = f'{base_username}{counter}'",
        "summary": "GitHub OAuth verification endpoint. Receives access_token via POST JSON. Validates against Supabase Auth using service role client. Extracts email and username from user_metadata. Implements username deduplication: checks profiles table for existing username, appends incrementing counter until unique (e.g. john -> john1 -> john2). For new users: inserts profile with github_url from identity_data. For existing users: updates missing github_url field. Creates User object and calls login_user().",
        "keywords": ["github", "oauth", "verify", "profile", "username", "unique", "deduplication", "counter", "supabase", "authentication", "identity_data", "user_metadata"],
        "start_line": 560, "end_line": 649
    },
    {
        "name": "register",
        "type": "function",
        "content": "@app.route('/register', methods=['POST'])\ndef register():\n    email = request.form.get('email')\n    password = request.form.get('password')\n    confirm = request.form.get('confirm_password')\n    username = request.form.get('username')\n    role = request.form.get('role', 'user')\n    if password != confirm: flash('Passwords do not match')\n    if len(password) < 8: flash('Password must be at least 8 characters')\n    auth = supabase.auth.sign_up({'email': email, 'password': password})\n    supabase_backend.table('profiles').insert({...}).execute()",
        "summary": "User registration handler with multi-step validation. Validates: all required fields present (email, password, confirm_password, username, full_name), passwords match, password >= 8 characters, username uniqueness via profiles table query. Signs up via Supabase Auth sign_up() with email_redirect_to for confirmation link. Handles duplicate email detection via identities array check. Inserts profile with role-specific fields: creator gets portfolio_url/expertise, user gets primary_interest. Handles email confirmation flow (flash confirmation message) vs direct login when auto-confirmed.",
        "keywords": ["register", "signup", "supabase", "validation", "profile", "password", "email", "confirmation", "username_unique", "role", "creator", "identities"],
        "start_line": 658, "end_line": 755
    },
    {
        "name": "edit_profile",
        "type": "function",
        "content": "@app.route('/profile/edit', methods=['GET', 'POST'])\n@login_required\ndef edit_profile():\n    if request.method == 'POST':\n        full_name = request.form.get('full_name')\n        username = request.form.get('username')\n        # Check username uniqueness if changed\n        supabase.table('profiles').update(update_data).eq('id', current_user.id).execute()",
        "summary": "Profile editing handler (GET/POST, login required). GET: fetches current profile from profiles table and renders edit form. POST: validates full_name and username are provided, checks username uniqueness against profiles table if changed from current value, updates profile with bio, avatar_url, and role-specific fields (creator: portfolio_url/expertise, user: primary_interest). Redirects to profile page on success with success flash message.",
        "keywords": ["edit", "profile", "update", "username", "validation", "supabase", "form", "login_required", "avatar_url", "bio"],
        "start_line": 846, "end_line": 912
    },
    {
        "name": "run_agent",
        "type": "function",
        "content": "def run_agent(user_input, agent_data):\n    agent_data = parse_to_dict(html.unescape(agent_data))\n    url = agent_data.get('base_url')\n    run_path = agent_data.get('run_path', '')\n    if run_path: url = url.rstrip('/') + '/' + run_path.lstrip('/')\n    response = requests.post(url, json=user_input['body'])\n    return f'Agent processed: {response.text}'",
        "summary": "Executes an AI agent by sending HTTP POST request. Parses agent_data from HTML-escaped string using parse_to_dict() utility. Constructs full URL by joining base_url and run_path with proper slash handling. Sends user_input['body'] as JSON payload. Returns response text on success or detailed error message on requests.RequestException or general Exception. Not a route handler — called internally by other routes.",
        "keywords": ["run", "agent", "execute", "post", "request", "base_url", "run_path", "payload", "html_unescape", "parse_to_dict", "internal"],
        "start_line": 974, "end_line": 1010
    },
    {
        "name": "parse_to_dict",
        "type": "function",
        "content": "def parse_to_dict(raw: str):\n    pattern = re.compile(r'(\\w+)=((?:\"[^\"]*\"|'[^']*'|\\[[^\\]]*\\]|\\{[^}]*\\}|[^,\\s]+))')\n    for match in pattern.finditer(raw):\n        key, value = match.groups()\n        try: result[key] = ast.literal_eval(value)\n        except: result[key] = value\n    return result",
        "summary": "Utility function to parse raw HTML-escaped string into Python dictionary. Uses regex to capture key=value pairs supporting quoted strings, lists, dicts, and bare tokens. Applies ast.literal_eval() for safe Python literal evaluation of values. Falls back to raw string for complex unparseable types like datetime() or class references. Called by run_agent() to deserialize agent configuration.",
        "keywords": ["parse", "dict", "regex", "literal_eval", "utility", "string", "conversion", "deserialize", "html_escape"],
        "start_line": 955, "end_line": 970
    },
    {
        "name": "create_api_key",
        "type": "function",
        "content": "@app.route('/api/keys', methods=['POST'])\n@login_required\ndef create_api_key():\n    key_val = secrets.token_urlsafe(32)\n    hashed = hashlib.sha256(key_val.encode()).hexdigest()\n    expires_at = datetime.utcnow() + timedelta(days=int(expiration.split()[0]))\n    record = {'id': str(uuid.uuid4()), 'user_id': current_user.id, 'hashed_key': hashed, 'masked_key': key_val[:8]+'...'+key_val[-4:]}\n    supabase_backend.table('api_keys').insert(record).execute()",
        "summary": "API key creation endpoint (POST, login required). Generates cryptographically secure key via secrets.token_urlsafe(32). Computes SHA-256 hash — never stores plaintext key. Calculates expiration from user-provided string (e.g. '30 Days', '90 Days'). Creates masked version showing first 8 and last 4 characters. Inserts record with uuid, user_id, name, hashed_key, masked_key, permissions JSON, limits JSON, expiration timestamps into api_keys table. Falls back to legacy 'key' column if hashed_key column doesn't exist. Returns full plaintext key only once on creation response.",
        "keywords": ["api_key", "create", "hash", "security", "permissions", "limits", "expiration", "supabase", "sha256", "secrets", "token_urlsafe", "masked_key", "login_required"],
        "start_line": 1077, "end_line": 1168
    },
    {
        "name": "error_handlers",
        "type": "block",
        "content": "@app.errorhandler(404)\ndef not_found(error):\n    return render_template('404.html'), 404\n\n@app.errorhandler(500)\ndef internal_error(error):\n    return render_template('500.html'), 500",
        "summary": "Custom HTTP error handlers. 404 Not Found: renders 404.html template with 404 status code. 500 Internal Server Error: renders 500.html template with 500 status code. Both use Flask's @app.errorhandler decorator pattern for consistent error page rendering.",
        "keywords": ["error", "handler", "404", "500", "not_found", "internal_error", "template", "http_status", "errorhandler"],
        "start_line": 1170, "end_line": 1178
    },
    {
        "name": "join_waitlist",
        "type": "function",
        "content": "@app.route('/join-waitlist', methods=['POST'])\ndef join_waitlist():\n    email = request.form.get('email', '').strip().lower()\n    if not re.match(r'[^@]+@[^@]+\\.[^@]+', email): return jsonify({'error': 'Invalid email'})\n    existing = supabase.table('waitlist').select('id').eq('email', email).execute()\n    threading.Thread(target=send_welcome_email, args=(email,)).start()\n    insert_result = supabase.table('waitlist').insert({'email': email, 'user_id': user_id}).execute()\n    count = supabase.table('waitlist').select('id', count='exact').execute().count + 114",
        "summary": "Waitlist signup endpoint. Validates email format via regex pattern. Checks for existing entry in waitlist table to prevent duplicates. Sends HTML welcome email asynchronously via threading.Thread (includes logo image, styled template). For new entries: inserts record with email and optional user_id. Returns success response with total count (adds base offset of 114 to actual count for social proof). Handles both new and returning signups gracefully.",
        "keywords": ["waitlist", "signup", "email", "validation", "regex", "supabase", "async", "thread", "welcome_email", "count", "social_proof", "duplicate_check"],
        "start_line": 1208, "end_line": 1342
    },
    {
        "name": "join_gitmem_waitlist",
        "type": "function",
        "content": "@app.route('/join-gitmem-waitlist', methods=['POST'])\ndef join_gitmem_waitlist():\n    data = request.get_json()\n    email = data.get('email')\n    name = data.get('name')\n    tools = data.get('tools')  # e.g. 'Cursor, VSCode'\n    stack = data.get('stack')  # e.g. 'Python, React'\n    goals = data.get('goals')\n    supabase.table('gitmem_waitlist').insert(gitmem_data).execute()",
        "summary": "GitMem-specific waitlist signup with extended profile data. Accepts JSON body: email, name, tools (IDE/editor preferences), stack (programming languages/frameworks), goals (what user hopes to achieve), setup (current development setup), open_to_feedback (boolean). Validates email format, checks duplicates in gitmem_waitlist table. Sends styled HTML welcome email asynchronously. Inserts record with user_id if authenticated.",
        "keywords": ["gitmem", "waitlist", "signup", "tools", "stack", "goals", "feedback", "email", "json", "ide", "programming_languages"],
        "start_line": 1366, "end_line": 1505
    },
    {
        "name": "memory",
        "type": "function",
        "content": "@app.route('/memory', methods=['GET', 'POST'])\n@login_required\ndef memory():\n    if request.method == 'POST':\n        files = request.files.getlist('files')\n        ALLOWED = {'.pdf','.ppt','.pptx','.doc','.docx','.txt','.csv','.png','.jpg','.jpeg'}\n        controller = RAG_DB_Controller_FILE_DATA()\n        controller.update_file_data_to_db(temp_path, current_user.id)\n        os.remove(temp_path)",
        "summary": "Memory page handler (GET/POST, login required). POST processes multiple file uploads: validates extensions against whitelist (.pdf, .ppt, .doc, .txt, .csv, images), saves to temporary files, sends to RAG database via RAG_DB_Controller_FILE_DATA.update_file_data_to_db() with user_id. Also handles plain text input via send_data_to_rag_db(). Cleans up temporary files. Provides upload count feedback via flash messages. GET renders memory.html template.",
        "keywords": ["memory", "upload", "file", "rag", "database", "controller", "text", "multifile", "login_required", "temp_file", "whitelist", "extension_validation"],
        "start_line": 1507, "end_line": 1607
    },
    {
        "name": "homepage",
        "type": "function",
        "content": "@app.route('/')\ndef homepage():\n    return render_template('homepage.html', user=current_user)",
        "summary": "Root route handler serving the main landing page. Renders homepage.html template with current_user context variable for conditional display of login/dashboard buttons. The '/' path is the entry point URL for the AI Agent Marketplace application.",
        "keywords": ["homepage", "root", "route", "landing", "template", "marketplace", "entry_point", "index"],
        "start_line": 1653, "end_line": 1656
    },
]


def verify(query, results, expected_names):
    """Extract names from nested result format and score."""
    retrieved = results.get("results", [])
    names = []
    scores = []
    for r in retrieved:
        chunk = r.get("chunk", r)
        names.append(chunk.get("name", ""))
        scores.append(round(r.get("score", 0), 4))
    
    found = [n for n in expected_names if n in names]
    missed = [n for n in expected_names if n not in names]
    pct = len(found) / max(len(expected_names), 1) * 100
    
    # Position bonus: expected items should be near the top
    position_score = 0
    for exp in expected_names:
        if exp in names:
            idx = names.index(exp)
            position_score += max(0, (5 - idx)) / 5  # 1.0 for #1, 0.8 for #2, etc.
    position_pct = position_score / max(len(expected_names), 1) * 100
    
    return {
        "pass": pct >= 40,
        "score": round(pct, 1),
        "position_score": round(position_pct, 1),
        "found": found,
        "missed": missed,
        "retrieved": names,
        "scores": scores
    }


def run_all_tests(api):
    """Run all hard test categories."""
    log = []
    all_results = []
    
    # =========================================================================
    # CATEGORY 1: Synonym & Paraphrase Queries
    # (Queries that use different words than what's in the chunks)
    # =========================================================================
    cat1 = [
        ("credential verification and user sign-in", ["login", "register", "github_verify"], "Synonym for authentication"),
        ("database rows permission policies", ["Supabase_config"], "RLS synonym"),
        ("prevent server hibernation on cloud hosting", ["keep_alive_task"], "Hibernation = sleep synonym"),
        ("cryptographic secret generation for API access", ["create_api_key"], "Crypto synonym for hash/token"),
        ("browser redirect after third-party authorization", ["auth_callback", "login_google"], "OAuth synonym"),
    ]
    
    # =========================================================================
    # CATEGORY 2: Negation & Exclusion (retrieve what's NOT something)
    # =========================================================================
    cat2 = [
        ("routes that do NOT require authentication", ["explore", "agent_detail", "homepage", "login", "login_google", "join_waitlist"], "Public routes"),
        ("functions that are not Flask route handlers", ["run_agent", "parse_to_dict", "keep_alive_task"], "Non-route functions"),
        ("endpoints that accept GET but not POST", ["explore", "agent_detail", "homepage", "login_google", "dashboard"], "GET-only routes"),
    ]
    
    # =========================================================================
    # CATEGORY 3: Multi-hop Reasoning
    # (Requires understanding relationships between chunks)
    # =========================================================================
    cat3 = [
        ("which function is called by run_agent to parse its input?", ["parse_to_dict", "run_agent"], "Dependency chain"),
        ("what happens after a user clicks Google login?", ["login_google", "auth_callback"], "Sequential flow"),
        ("complete flow from signup to first dashboard view", ["register", "login", "dashboard"], "Multi-step flow"),
        ("how does the system ensure usernames are unique during social login?", ["github_verify", "register"], "Username dedup"),
        ("what token pair does the system store after successful login?", ["login", "auth_callback"], "Token storage"),
    ]
    
    # =========================================================================
    # CATEGORY 4: Vague / Abstract Queries
    # =========================================================================
    cat4 = [
        ("security measures in the application", ["create_api_key", "login", "register"], "Abstract security"),
        ("data persistence layer configuration", ["Supabase_config", "memory"], "Abstract DB"),
        ("user onboarding experience", ["register", "login", "dashboard"], "Abstract UX"),
        ("infrastructure and DevOps concerns", ["keep_alive_task", "FlaskApp_init", "SocketIO_MCP_setup"], "Abstract infra"),
        ("content validation and sanitization", ["register", "join_waitlist", "memory"], "Abstract validation"),
    ]
    
    # =========================================================================
    # CATEGORY 5: Very Specific / Technical Queries
    # =========================================================================
    cat5 = [
        ("sha256 hashing of secret token", ["create_api_key"], "Exact technical detail"),
        ("gevent monkey patching at module level", ["FlaskApp_init"], "Specfic gevent detail"),
        ("ast.literal_eval for safe string parsing", ["parse_to_dict"], "Specific stdlib usage"),
        ("threading.Thread with daemon=True", ["keep_alive_task"], "Specific threading pattern"),
        ("secrets.token_urlsafe(32)", ["create_api_key"], "Exact function call"),
        ("sign_in_with_password Supabase method", ["login"], "Exact Supabase API"),
        ("Blueprint registration at /mcp prefix", ["SocketIO_MCP_setup"], "Specific Blueprint detail"),
    ]
    
    # =========================================================================
    # CATEGORY 6: Return Value / Output Queries
    # =========================================================================
    cat6 = [
        ("which routes return JSON responses?", ["create_api_key", "join_waitlist", "join_gitmem_waitlist"], "JSON response routes"),
        ("which routes redirect to other pages?", ["login", "login_google", "agent_detail", "register"], "Redirect routes"),
        ("what does the explore page render?", ["explore"], "Template output"),
        ("which endpoint returns the full API key only once?", ["create_api_key"], "One-time response"),
    ]
    
    # =========================================================================
    # CATEGORY 7: Edge Cases
    # =========================================================================
    cat7 = [
        ("404", ["error_handlers"], "Single token query"),
        ("@login_required decorated functions", ["dashboard", "edit_profile", "submit_agent", "create_api_key", "memory"], "Decorator query"),
        ("the function at line 447", ["login"], "Line number query"),
        ("index.py main Flask application entry point", ["FlaskApp_init", "homepage"], "File-level query"),
    ]
    
    categories = [
        ("SYNONYM & PARAPHRASE", cat1),
        ("NEGATION & EXCLUSION", cat2),
        ("MULTI-HOP REASONING", cat3),
        ("VAGUE / ABSTRACT", cat4),
        ("VERY SPECIFIC / TECHNICAL", cat5),
        ("RETURN VALUE / OUTPUT", cat6),
        ("EDGE CASES", cat7),
    ]
    
    for cat_name, cases in categories:
        log.append(f"\n{'='*70}")
        log.append(f"  CATEGORY: {cat_name}")
        log.append(f"{'='*70}")
        
        cat_passed = 0
        cat_total = 0
        cat_score_sum = 0
        
        for query, expected, desc in cases:
            cat_total += 1
            log.append(f"\n  [{desc}]")
            log.append(f"  Q: \"{query}\"")
            log.append(f"  Expected: {expected}")
            
            t = time.time()
            result = api.get_mem(AGENT_ID, query)
            elapsed = time.time() - t
            
            v = verify(query, result, expected)
            
            tag = "PASS" if v["pass"] else "FAIL"
            log.append(f"  {tag} | Relevance: {v['score']}% | Position: {v['position_score']}% | Time: {elapsed:.2f}s")
            log.append(f"  Retrieved: {v['retrieved']}")
            log.append(f"  Scores: {v['scores']}")
            log.append(f"  Found: {v['found']}")
            if v['missed']:
                log.append(f"  MISSED: {v['missed']}")
            
            if v["pass"]:
                cat_passed += 1
            cat_score_sum += v["score"]
        
        avg = cat_score_sum / max(cat_total, 1)
        log.append(f"\n  >>> {cat_name}: {cat_passed}/{cat_total} passed | Avg: {avg:.1f}%")
        all_results.append({"category": cat_name, "passed": cat_passed, "total": cat_total, "avg": avg})
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_p = sum(r["passed"] for r in all_results)
    total_t = sum(r["total"] for r in all_results)
    overall_avg = sum(r["avg"] for r in all_results) / len(all_results)
    
    log.append(f"\n\n{'='*70}")
    log.append("  HARD TEST FINAL SUMMARY")
    log.append(f"{'='*70}")
    for r in all_results:
        log.append(f"  {r['category']:30s}: {r['passed']}/{r['total']} passed | Avg: {r['avg']:.1f}%")
    log.append(f"\n  TOTAL: {total_p}/{total_t} passed ({total_p/total_t*100:.1f}%)")
    log.append(f"  OVERALL AVG RELEVANCE: {overall_avg:.1f}%")
    
    if overall_avg >= 70:
        log.append("\n  VERDICT: EXCELLENT")
    elif overall_avg >= 50:
        log.append("\n  VERDICT: ACCEPTABLE — room for improvement")
    else:
        log.append("\n  VERDICT: NEEDS WORK — retrieval quality must improve")
    
    return "\n".join(log)


def main():
    print("=" * 70)
    print("  HARD RETRIEVAL TEST — create_mem & get_mem")
    print("=" * 70)
    
    # Clean
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    
    # Init
    api = CodingAPI(root_path=TEST_DATA_DIR)
    print(f"CodingAPI initialized at {TEST_DATA_DIR}")
    
    # Create flow with 24 rich chunks
    print(f"\nStoring {len(HARD_CHUNKS)} chunks...")
    t = time.time()
    result = api.create_mem(AGENT_ID, FILE_PATH, HARD_CHUNKS)
    print(f"create_mem: {json.dumps(result)} ({time.time()-t:.1f}s)")
    
    # Run hard tests
    print("\nRunning hard retrieval tests...\n")
    output = run_all_tests(api)
    
    # Save
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(output)
    
    print(output)
    print(f"\nFull results at: {LOG_FILE}")


if __name__ == "__main__":
    main()
