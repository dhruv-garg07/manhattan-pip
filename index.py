#!/usr/bin/env python3
"""Flask AI Agent Marketplace Application."""

# CRITICAL: Gevent monkey patching MUST happen before any other imports
# This ensures all standard library modules work properly with gevent workers
try:
    from gevent import monkey
    monkey.patch_all()
    print("[STARTUP] Gevent monkey patching applied successfully")
except ImportError:
    print("[STARTUP] Gevent not available - running without monkey patching")

import os
import sys
# import sys, os
sys.path.append(os.path.dirname(__file__))
# Get the current file's directory
current_dir = os.path.abspath(os.path.dirname(__file__))

# Get the parent directory (one level up)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Add parent directory to sys.path
sys.path.insert(0, parent_dir)
print(parent_dir)

# Get the current file's directory
current_dir = os.path.dirname(__file__)

# Go two levels up
grandparent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

# Add lib directory to sys.path
lib_dir = os.path.abspath(os.path.join(parent_dir, 'lib'))
sys.path.insert(0, lib_dir)

# Add to sys.path
sys.path.insert(0, grandparent_dir)

import uuid
import asyncio
import smtplib
import shutil
import tempfile
from datetime import datetime, timedelta
import secrets
import threading
import requests
import time

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from werkzeug.utils import secure_filename
# Fix import for Octave_mem when running from api/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Octave_mem.RAG_DB_CONTROLLER.write_data_RAG_file_uploads import RAG_DB_Controller_FILE_DATA

from supabase import create_client, Client
from postgrest.exceptions import APIError
import json

# Ensure backend_examples can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import services from the backend_examples
from backend_examples.python.services.agents import agent_service
from backend_examples.python.services.creators import creator_service

from backend_examples.python.models import SearchFilters
# API Key helpers
from key_utils import hash_key, mask_key, generate_secret_key
from dotenv import load_dotenv
# Email service
from utlis.email_service import get_email_service

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')
TEMPLATES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates'))
app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)

# --- Flask-SocketIO for Real-Time Updates ---
try:
    from flask_socketio import SocketIO
    # Use 'gevent' async_mode to match gunicorn worker class
    # This is critical for websocket/SSE support in production
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent', ping_timeout=60, ping_interval=25)
    
    # Initialize GitMem WebSocket handlers
    from gitmem.api.websocket_events import init_websocket
    init_websocket(socketio)
    print("[STARTUP] Flask-SocketIO initialized for real-time updates (gevent mode)")
    
    # Register MCP Blueprint synchronously to avoid race conditions
    try:
        from mcp_socketio_gateway import init_mcp_socketio, mcp_bp
        app.register_blueprint(mcp_bp)
        print("[STARTUP] MCP SSE Blueprint registered at /mcp")
    except Exception as e:
        print(f"[STARTUP] MCP Blueprint registration error: {e}")
        mcp_bp = None
    
    # Initialize MCP Socket.IO Gateway after app context is ready
    def init_mcp_gateway():
        try:
            if mcp_bp:
                from mcp_socketio_gateway import init_mcp_socketio
                init_mcp_socketio(socketio)
                print("[STARTUP] MCP Socket.IO Gateway initialized on /mcp namespace")
                print("[STARTUP] MCP SSE Transport initialized at /mcp/sse")
        except Exception as e:
            print(f"[STARTUP] MCP Gateway initialization error: {e}")
    
    # Run MCP initialization in background to not block main app
    import threading
    mcp_init_thread = threading.Thread(target=init_mcp_gateway, daemon=True)
    mcp_init_thread.start()
    print("[STARTUP] MCP Gateway initialization started in background thread")
    
except ImportError as e:
    print(f"[STARTUP] Flask-SocketIO not available: {e}")
    socketio = None

# --- GitMem Integration ---
from gitmem.api.routes import gitmem_bp
app.register_blueprint(gitmem_bp)

# --- MCP Client Compatibility ---

try:
     # Try importing from root (parent_dir)
     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    #  from mcp_compat_shim import mcp_compat_bp
    #  app.register_blueprint(mcp_compat_bp)
    #  print("[MOCK] MCP Shim registered (root)")
except ImportError as e:
    #  print(f"Shim import failed: {e}")
     # Fallback to local (if moved) or skip
     pass
# ------------------------------
# ------------------------------

app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
supabase_backend: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth'

# ==================== Keep-Alive Background Task ====================
# This function pings the website every 5 minutes to prevent Render from sleeping
def keep_alive_task():
    """Background task that pings the website every 5 minutes."""
    WEBSITE_URL = "https://themanhattanproject.ai"
    PING_INTERVAL = 300  # 5 minutes in seconds
    
    def ping_website():
        while True:
            try:
                time.sleep(PING_INTERVAL)
                response = requests.get(f"{WEBSITE_URL}/ping", timeout=10)
                if response.status_code == 200:
                    print(f"[KEEP-ALIVE] Successfully pinged {WEBSITE_URL}/ping at {datetime.now().isoformat()}")
                else:
                    print(f"[KEEP-ALIVE] Ping returned status {response.status_code} at {datetime.now().isoformat()}")
            except Exception as e:
                print(f"[KEEP-ALIVE] Error pinging website: {e}")
    
    # Start the keep-alive thread as a daemon so it doesn't block shutdown
    keep_alive_thread = threading.Thread(target=ping_website, daemon=True)
    keep_alive_thread.start()
    print("[KEEP-ALIVE] Background pinging task started. Will ping every 5 minutes.")

# =====================================================================

class User:
    def __init__(self, user_id=None, email=None):
        self.id = user_id
        self.email = email

    @property
    def is_authenticated(self):
        return self.id is not None

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return self.id is None

    def get_id(self):
        return str(self.id) if self.id else None


@login_manager.user_loader
def load_user(user_id):
    if not user_id:
        return None
    return User(user_id)


# ==================== Health Check Endpoints ====================
@app.route('/ping')
def ping():
    """Simple ping endpoint for keep-alive and health checks."""
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route('/health')
def health_check():
    """Detailed health check endpoint for monitoring."""
    from datetime import datetime
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "socketio_enabled": socketio is not None,
        "mcp_enabled": mcp_bp is not None if 'mcp_bp' in dir() else False,
    }
    return jsonify(status)

@app.route('/mcp-docs')
def mcp_docs():
    """MCP Server Documentation Page"""
    return render_template('mcp_docs.html')

@app.route('/explore')
def explore():
    """Explore agents with search and filters."""
    # Get filter parameters from URL
    search = request.args.get('search', '')
    category = request.args.get('category', '')
    model = request.args.get('model', '')
    status = request.args.get('status', '')
    sort_by = request.args.get('sort_by', 'created_at')
    modalities = request.args.getlist('modalities')
    capabilities = request.args.getlist('capabilities')
    
    # Create filters object
    filters = SearchFilters(
        search=search,
        category=category,
        model=model,
        status=status,
        sort_by=sort_by,
        modalities=modalities,
        capabilities=capabilities
    )
    print("Filters applied:", filters)
    try:
        agents = asyncio.run(agent_service.fetch_agents(filters))
        print("Fetched agents:", agents)
    except Exception as e:
        flash(f'Error fetching agents: {str(e)}', 'error')
        agents = []
    
    return render_template('explore.html', 
                        agents=agents, 
                        filters=filters,
                        search=search,
                        category=category,
                        #  model=model,
                        sort_by=sort_by)

@app.route('/agent/<agent_id>')
def agent_detail(agent_id):
    """Agent detail page."""
    try:
        # agent = agent_service.get_agent_by_id(agent_id)
        agent = asyncio.run(agent_service.get_agent_by_id(agent_id))
        print("Fetched agent:", agent)
        if not agent:
            flash('Agent not found', 'error')
            return redirect(url_for('explore'))
    except Exception as e:
        flash(f'Error fetching agent: {str(e)}', 'error')
        return redirect(url_for('explore'))
    
    return render_template('agent_detail.html', agent=agent)

@app.route('/creators')
def creators():
    search = request.args.get('search', '')
    sort_by = request.args.get('sort_by', 'reputation')
    try:
        # If fetch_creators is async, run it and get the list
        all_creators = asyncio.run(creator_service.fetch_creators())
        # If filter_and_sort_creators is async, run it too
        if asyncio.iscoroutinefunction(creator_service.filter_and_sort_creators):
            filtered_creators = asyncio.run(creator_service.filter_and_sort_creators(all_creators, search, sort_by))
        else:
            filtered_creators = creator_service.filter_and_sort_creators(all_creators, search, sort_by)
    except Exception as e:
        flash(f'Error fetching creators: {str(e)}', 'error')
        filtered_creators = []
    return render_template('creators.html', 
                        creators=filtered_creators,
                        search=search,
                        sort_by=sort_by)


@app.route('/harshit')
def harshit_page():
    """Founder page for Harshit."""
    try:
        return render_template('harshit.html')
    except Exception as e:
        # If template missing or render fails, return a simple fallback
        return f"<h1>Harshit</h1><p>Unable to render page: {e}</p>", 500

@app.route('/ranaji')
def ranaji_page():
    """Rana Ji Ka Rishta page."""
    try:
        return render_template('ranaji.html')
    except Exception as e:
        return f"<h1>Rana Ji</h1><p>Unable to render page: {e}</p>", 500

@app.route('/submit')
@login_required
def creator_studio():
    """Creator studio for submitting agents."""
    return render_template('creator_studio.html')

@app.route('/submit', methods=['POST'])
@login_required
def submit_agent():
    """Handle agent submission."""
    try:
        # agent_data = {
        #     'name': request.form.get('name'),
        #     'description': request.form.get('description'),
        #     'category': request.form.get('category'),
        #     'model': request.form.get('model'),
        #     'tags': request.form.get('tags', '').split(',') if request.form.get('tags') else [],
        #     'github_url': request.form.get('github_url'),
        #     'dockerfile_url': request.form.get('dockerfile_url'),
        #     'status': 'pending',
        #     'created_at': datetime.utcnow(),
        #     'updated_at': datetime.utcnow()
        # }
        header_keys = request.form.getlist('header_keys[]')
        header_values = request.form.getlist('header_values[]')
        headers = {}
        for key, value in zip(header_keys, header_values):
            if key and value:  # Only add non-empty headers
                headers[key] = value
        
        # Process authentication
        auth_keys = request.form.getlist('auth_keys[]')
        auth_values = request.form.getlist('auth_values[]')
        authentication = {}
        for key, value in zip(auth_keys, auth_values):
            if key and value:  # Only add non-empty auth fields
                authentication[key] = value
        
        # Not much 
        okay_ish = True
        
        agent_data = {
            'name': request.form.get('name'),
            'description': request.form.get('description'),
            'category': request.form.get('category'),
            'base_url': request.form.get('base_url'),
            'run_path': request.form.get('run_path'),
            'headers': headers,
            'content_type': request.form.get('content_type'),
            'authentication': authentication,
            # 'data_format': request.form.get('data_format'),
            'io_schema': json.loads(request.form.get('sample_input')) if request.form.get('sample_input') else {},
            'out_schema': json.loads(request.form.get('sample_output')) if request.form.get('sample_output') else {},
            'tags': [tag.strip() for tag in request.form.get('tags', '').split(',')] if request.form.get('tags') else [],
            'status': 'pending',
            
            # 'created_at': json.dumps(datetime.utcnow()),
            # 'updated_at': json.dumps(datetime.utcnow()),
            'creator_id': current_user.id,
            'success_rate': 0,
            'total_runs': 0,
            'avg_rating': 0,
            'avg_latency': 0,
            'upvotes': 0,
            'runtime_dependencies': ['python'],
            
        }
    
        print("Submitting agent data:", agent_data)

        print("Current user ID:", current_user)
        agent = asyncio.run(agent_service.create_agent(agent_data, current_user.id))
        
        print("Created agent:", agent)

        if agent:
            flash('Agent submitted successfully and is pending review!', 'success')
        else:
            flash('Failed to submit agent. Please try again.', 'error')
            
    except Exception as e:
        flash(f'Error submitting agent: {str(e)}', 'error')
    
    return redirect(url_for('creator_studio'))

@app.route('/auth')
def auth():
    """Authentication page."""
    if current_user.is_authenticated:
        return redirect(url_for('homepage'))
    return render_template('auth.html')

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        profile_res = supabase.table('profiles').select('*').eq('id', current_user.id).execute()
        if not profile_res.data:
            # Fallback for sync issues
            current_profile = {
                "full_name": current_user.email,
                "username": current_user.email.split('@')[0],
                "user_role": "user",
                "email": current_user.email,
                "created_at": datetime.utcnow().isoformat()
            }
        else:
            current_profile = profile_res.data[0]
            
        return render_template('dashboard.html', profile=current_profile, is_own_profile=True)
    except Exception as e:
        print(f"Error loading dashboard: {e}")
        flash(f"Error loading dashboard: {e}", "error")
        return redirect(url_for('homepage'))


def _clean_email(v: str) -> str:
    return (v or "").strip().lower()

@app.route('/login', methods=['POST'])
def login():
    print("Supabase URL:", SUPABASE_URL)
    print("Supabase client:", supabase)
    email = _clean_email(request.form.get('email'))
    password = request.form.get('password') or ""

    if not email or not password:
        flash('Please fill in both email and password.', 'error')
        return redirect(url_for('auth'))

    try:
        auth = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if not auth.user or not auth.session:
            # This usually means email not confirmed or bad credentials
            flash('Invalid credentials or email not confirmed.', 'error')
            return redirect(url_for('auth'))

        # Persist tokens if you need to call Supabase on behalf of the user later
        session['sb_access_token'] = auth.session.access_token
        session['sb_refresh_token'] = auth.session.refresh_token

        user = User(user_id=auth.user.id, email=email)
        login_user(user)

        flash('Logged in successfully!', 'success')
        next_url = request.args.get('next') or url_for('homepage')
        return redirect(next_url)

    except Exception as e:
        # Optional: log e
        flash('Login failed. Please check your credentials.', 'error')
        return redirect(url_for('auth'))
    
@app.route("/login/google")
def login_google():
    redirect_url = url_for("auth_callback", _external=True)
    oauth_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={redirect_url}"
    return redirect(oauth_url)

@app.route("/auth/callback", methods=["GET", "POST"])
def auth_callback():
    # --- GET request: serve HTML with JS to extract tokens ---
    if request.method == "GET":
        return render_template("auth_callback.html")  # your JS in this page will POST the tokens

    # --- POST request: handle token sent from frontend ---
    if request.method == "POST":
        data = request.get_json(silent=True)
        if not data:
            return {"error": "Expected JSON body"}, 400

        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")

        if not access_token:
            return {"error": "Missing access_token"}, 400

        try:
            # --- Validate token with Supabase ---
            user_resp = supabase.auth.get_user(access_token)
            if not user_resp or not user_resp.user:
                return {"error": "Invalid token"}, 401

            user = user_resp.user

            # --- Save tokens in server-side session ---
            session["sb_access_token"] = access_token
            session["sb_refresh_token"] = refresh_token
            session["user_email"] = user.email

            # --- Ensure profile exists in 'profiles' table ---
            try:
                existing_profile = supabase.table("profiles").select("id").eq("id", user.id).execute()
                if not existing_profile.data:
                    profile_data = {
                        "id": user.id,  # same UUID as auth.users
                        "email": user.email,
                        "username": user.email.split("@")[0],  # default username
                        "full_name": user.user_metadata.get("full_name") or user.email,
                        "user_role": "user",  # default role
                        "portfolio_url": None,
                        "primary_interest": None,
                        "portfolio_url": None,
                        "expertise": None,
                        "created_at": datetime.utcnow().isoformat()
                    }
                    # Use service role key to bypass RLS, and handle duplicates gracefully
                    supabase_backend.table("profiles").upsert(profile_data, on_conflict="id").execute()
            except Exception as e:
                print("Error syncing profile:", e)

            # --- Log in the user with Flask-Login ---
            user_obj = User(user_id=user.id, email=user.email)
            login_user(user_obj)

            return {"message": "Logged in successfully"}

        except Exception as e:
            print("Error during Google login:", e)
            return {"error": "Login failed"}, 500

@app.route("/login/github")
def login_github():
    redirect_url = url_for("github_callback", _external=True)
    oauth_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=github&redirect_to={redirect_url}"
    return redirect(oauth_url)

@app.route("/auth/github/callback")
def github_callback():
    # Supabase will redirect with #access_token in URL fragment
    return render_template("oauth_redirect.html")

@app.route("/auth/github/verify", methods=["POST"])
def github_verify():
    print("Verifying GitHub login...")
    data = request.get_json()
    access_token = data.get("access_token")
    print("Received GitHub access token:", access_token[:15] + "...")

    if not access_token:
        return jsonify({"error": "Missing access token"}), 400

    try:
        # Fetch user from Supabase Auth
        user_info = supabase_backend.auth.get_user(access_token)

        if not user_info or not user_info.user:
            return jsonify({"error": "Invalid GitHub user response"}), 400

        github_user = user_info.user
        github_email = github_user.email
        print("GitHub user info:", github_user)
        github_username = github_user.user_metadata.get("preferred_username") or github_user.user_metadata.get("user_name")
        github_profile_url = f"https://github.com/{github_username}" if github_username else None

        if not github_email:
            return jsonify({"error": "GitHub account has no email"}), 400

        # -----------------------------
        # Check if user already exists
        # -----------------------------
        existing_profile = (
            supabase_backend.table("profiles").select("*").eq("email", github_email).execute()
        )

        if existing_profile.data and len(existing_profile.data) > 0:
            # Existing profile → login
            profile_id = existing_profile.data[0]["id"]
            print(f"Profile exists: {github_email}")

            if not existing_profile.data[0].get("github_url"):
                print("GitHub URL missing, updating...")
                update_res = supabase_backend.table("profiles").update({
                    "github_url": github_profile_url
                }).eq("id", profile_id).execute()

                if update_res.data:
                    print(f"Updated GitHub URL for {github_email}")
                else:
                    print(f"Failed to update GitHub URL for {github_email}, response: {update_res}")


        else:
            # -----------------------------
            # New user → create profile
            # -----------------------------
            print("Creating new profile for:", github_email)
            base_username = github_email.split("@")[0]
            username = base_username
            counter = 1
            while True:
                username_check = supabase_backend.table("profiles").select("id").eq("username", username).execute()
                if username_check.data and len(username_check.data) > 0:
                    username = f"{base_username}{counter}"
                    counter += 1
                else:
                    break

            profile_id = github_user.id or str(uuid.uuid4())
            supabase_backend.table("profiles").insert({
                "id": profile_id,
                "email": github_email,
                "username": username,
                "full_name": github_user.user_metadata.get("full_name", ""),
                "user_role": "user",
                "portfolio_url": None,
                "expertise": None,
                "primary_interest": None,
                "github_url": github_profile_url,
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            print(f"Created new profile: {github_email}")

        # -----------------------------
        # Log in with Flask-Login
        # -----------------------------
        login_user(User(profile_id))
        return jsonify({"success": True, "redirect": url_for("homepage")})

    except Exception as e:
        print("Error during GitHub login:", str(e))
        return jsonify({"error": str(e)}), 500




# You might have a helper for this already, if not, it's good practice
def _clean_email(email):
    return (email or "").lower().strip()

@app.route('/register', methods=['POST'])
def register():
    # --- 1. Get all form data ---
    email = _clean_email(request.form.get('email'))
    password = request.form.get('password') or ""
    confirm_password = request.form.get('confirm_password') or ""
    full_name = request.form.get('full_name') or ""
    username = request.form.get('username') or ""
    user_role = request.form.get('user_role') or ""
    
    # Role-specific fields
    portfolio_url = request.form.get('portfolio_url')
    expertise = request.form.get('expertise')
    primary_interest = request.form.get('primary_interest')

    # --- 2. Perform validation ---
    if not all([email, password, full_name, username, user_role]):
        flash('Please fill in all required fields.', 'error')
        return redirect(url_for('auth'))
    
    if password != confirm_password:
        flash('Passwords do not match.', 'error')
        return redirect(url_for('auth'))
    
    if len(password) < 8:
        flash('Password must be at least 8 characters long.', 'error')
        return redirect(url_for('auth'))

    # --- 3. Check for unique username before trying to create the user ---
    try:
        # Query your 'profiles' table to see if the username exists
        existing_user = supabase.table('profiles').select('id').eq('username', username).execute()
        if existing_user.data:
            flash('That username is already taken. Please choose another.', 'error')
            return redirect(url_for('auth'))
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('auth'))

    # --- 4. Attempt to sign up the user with Supabase Auth ---
    try:
        auth = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"email_redirect_to": url_for('auth', _external=True)}
        })

        if auth.user and not auth.user.identities:
            flash("That email is already registered. Try logging in.", "error")
            return redirect(url_for("auth"))

        # --- 5. If user auth is created, insert data into the profiles table ---
        if auth.user:
            profile_data = {
                'id': auth.user.id,  # Link to the auth.users table
                'username': username,
                'full_name': full_name,
                'user_role': user_role,
                'portfolio_url': portfolio_url if user_role == 'creator' else None,
                'expertise': expertise if user_role == 'creator' else None,
                'primary_interest': primary_interest if user_role == 'user' else None,
            }
            # Insert the new profile. Use a try-except block for safety.
            try:
                supabase.table('profiles').insert(profile_data).execute()
            except Exception as e:
                # This is a critical error. The auth user was created, but the profile failed.
                # You should log this error for manual review.
                # For the user, a generic error is okay for now.
                print(f"CRITICAL: Failed to create profile for user {auth.user.id}. Error: {e}")
                flash('Registration failed at the final step. Please contact support.', 'error')
                return redirect(url_for('auth'))


        # --- Handle session based on email confirmation settings ---
        if not auth.session: # Email confirmation required
            flash('Account created! Please check your email to confirm your address.', 'success')
            return redirect(url_for('auth'))
        
        if auth.session: # Email confirmation is disabled, user logged in directly
            session['sb_access_token'] = auth.session.access_token
            session['sb_refresh_token'] = auth.session.refresh_token
            # Your User model might need to be updated to load profile data
            user = User(user_id=auth.user.id, email=email) 
            login_user(user)
            flash('Account created successfully!', 'success')
            return redirect(url_for('homepage'))

        flash('An unknown error occurred during registration.', 'error')
        return redirect(url_for('auth'))

    except Exception as e:
        msg = str(e)
        if 'User already registered' in msg:
            flash('That email is already registered. Try logging in.', 'error')
        else:
            flash(f'Registration failed: {msg}', 'error')
        return redirect(url_for('auth'))

@app.route('/logout', methods=['GET','POST'])
@login_required
def logout():
    try:
        # Optional: sign out from Supabase (mainly relevant if you’re using refresh token rotation)
        if session.get('sb_access_token'):
            supabase.auth.sign_out()
    except Exception:
        pass
    session.pop('sb_access_token', None)
    session.pop('sb_refresh_token', None)
    logout_user()
    flash('Logged out.', 'success')
    session.clear()
    return redirect(url_for('auth'))



from uuid import UUID

@app.route('/profile')
@login_required
def my_profile():
    """
    Displays the profile page for the currently logged-in user.
    Redirects to their public username-based URL.
    """
    try:
        # Ensure the ID is a proper UUID string
        user_id = str(UUID(str(current_user.id)))
        print("Current user ID:", user_id)

        # Supabase query
        user_profile_res = (
            supabase.table('profiles')
            .select('username')
            .eq('id', user_id)
            .execute()
        )


        print("User profile response:", user_profile_res.data)

        all_profiles_res = supabase.table('profiles').select('*').execute()
        print("All profiles:", all_profiles_res.data)

        if user_profile_res.data and len(user_profile_res.data) > 0:
            username = user_profile_res.data[0]['username']
            return redirect(url_for('view_profile', username=username))
        else:
            flash("Your profile has not been set up yet. Please contact support or re-register.", "error")
            return redirect(url_for('homepage'))

    except Exception as e:
        flash(f"An error occurred while fetching your profile: {e}", "error")
        return redirect(url_for('homepage'))





@app.route('/profile/<username>')
def view_profile(username):
    """
    Displays a user's public profile page, identified by their username.
    """
    try:
        # Fetch the profile data from Supabase using the username
        # REMOVED .single() to prevent a similar potential error
        profile_res = supabase.table('profiles').select('*').eq('username', username).execute()

        # If the data list is empty, the user does not exist
        if not profile_res.data:
            abort(404) # Renders a "Not Found" page

        # Since we are no longer using .single(), the result is a list. Get the first item.
        profile_data = profile_res.data[0]
        
        # Determine if the person viewing the page is the owner of the profile
        is_own_profile = False
        if current_user.is_authenticated and current_user.id == profile_data['id']:
            is_own_profile = True

        return render_template('profile.html', profile=profile_data, is_own_profile=is_own_profile)

    except Exception as e:
        flash(f"An error occurred while fetching the profile: {e}", "error")
        return redirect(url_for('homepage'))

@app.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """
    Allow the current user to edit their profile information.
    """
    # First, get the user's current profile data to populate the form
    try:
        profile_res = supabase.table('profiles').select('*').eq('id', current_user.id).execute()
        if not profile_res.data:
            flash("Your profile could not be found. Cannot edit.", "error")
            return redirect(url_for('homepage'))
        
        current_profile = profile_res.data[0]
    except Exception as e:
        flash(f"An error occurred while fetching your profile: {e}", "error")
        return redirect(url_for('homepage'))

    if request.method == 'POST':
        # Handle the form submission
        full_name = request.form.get('full_name') or ""
        username = request.form.get('username') or ""
        portfolio_url = request.form.get('portfolio_url')
        expertise = request.form.get('expertise')
        primary_interest = request.form.get('primary_interest')
        
        # --- Validation ---
        if not full_name or not username:
            flash("Full Name and Username are required.", "error")
            return render_template('edit_profile.html', profile=current_profile)

        # --- Unique Username Check (if it was changed) ---
        if username != current_profile['username']:
            try:
                existing_user = supabase.table('profiles').select('id').eq('username', username).execute()
                if existing_user.data:
                    flash('That username is already taken. Please choose another.', 'error')
                    submitted_data = current_profile.copy()
                    submitted_data.update({
                        'full_name': full_name, 'username': username,
                        'portfolio_url': portfolio_url, 'expertise': expertise,
                        'primary_interest': primary_interest
                    })
                    return render_template('edit_profile.html', profile=submitted_data)
            except Exception as e:
                flash(f'An error occurred while checking the username: {e}', 'error')
                return render_template('edit_profile.html', profile=current_profile)
        
        # --- Prepare data for update ---
        update_data = {
            'full_name': full_name, 'username': username,
            'portfolio_url': portfolio_url if current_profile['user_role'] == 'creator' else None,
            'expertise': expertise if current_profile['user_role'] == 'creator' else None,
            'primary_interest': primary_interest if current_profile['user_role'] == 'user' else None,
        }

        # --- Execute Update ---
        try:
            supabase.table('profiles').update(update_data).eq('id', current_user.id).execute()
            flash('Your profile has been updated successfully!', 'success')
            return redirect(url_for('view_profile', username=username))
        except Exception as e:
            flash(f'An error occurred while updating your profile: {e}', 'error')
            return render_template('edit_profile.html', profile=current_profile)

    # --- For GET request, just show the form ---
    return render_template('edit_profile.html', profile=current_profile)

@app.route('/trending')
def trending():
    """Trending agents - redirect to explore with trending sort."""
    return redirect(url_for('explore', sort_by='popular'))

@app.route('/categories')
def categories():
    """Categories - redirect to explore."""
    return redirect(url_for('explore'))

# API endpoints for AJAX requests
@app.route('/api/agents')
def api_agents():
    """API endpoint for fetching agents."""
    # Get filter parameters
    search = request.args.get('search', '')
    category = request.args.get('category', '')
    model = request.args.get('model', '')
    status = request.args.get('status', '')
    sort_by = request.args.get('sort_by', 'created_at')
    modalities = request.args.getlist('modalities')
    capabilities = request.args.getlist('capabilities')
    
    filters = SearchFilters(
        search=search,
        category=category,
        model=model,
        status=status,
        sort_by=sort_by,
        modalities=modalities,
        capabilities=capabilities
    )
    
    try:
        agents = agent_service.fetch_agents(filters)
        return jsonify([agent.dict() for agent in agents])
    except Exception as e:
        return jsonify({'error': str(e)}), 500
import re
import ast

def parse_to_dict(raw: str):
    # Regex to capture key=value pairs (value can be quoted or unquoted)
    pattern = re.compile(r"(\w+)=((?:'[^']*')|(?:\[[^\]]*\])|(?:\{[^}]*\})|(?:\S+))")
    result = {}

    for match in pattern.finditer(raw):
        key, value = match.groups()

        # Try to safely evaluate Python literals (lists, dicts, numbers, booleans, None, strings)
        try:
            result[key] = ast.literal_eval(value)
        except Exception:
            # If not a pure literal (like datetime(...), Creator(...)), keep as string
            result[key] = value

    return result
import requests
import html
from pydantic.json import pydantic_encoder
def run_agent(user_input, agent_data):
    """
    Runs an agent by sending a POST request to the agent's base_url with the input payload.
    agent_data is expected to be a dict with at least 'base_url'.
    """
    
    print("==== USER INPUT ==== for now:",user_input)
    # agent_data = agent_data.dict()
    print("==== AGENT DATA ==== for now:",agent_data)
    agent_data = parse_to_dict(html.unescape(agent_data))
    print("==== AGENT DATA ==== for now:",agent_data)
    if not agent_data or not isinstance(agent_data, dict):
        return "Invalid agent data. Must be a dict."

    url = agent_data.get("base_url")
    run_path = agent_data.get("run_path", "")
    if run_path:
        url = url.rstrip("/") + "/" + run_path.lstrip("/")  # Ensure single slash between base_url and run_path     
        
    if not url:
        return "Agent base_url not found."

    # Build the command/payload
    command = {}
    if isinstance(user_input, dict):
        command.update(user_input)
    else:
        command["user_input"] = user_input

    try:
        response = requests.post(url, json=user_input['body'])
        response.raise_for_status()  # raise if not 2xx
        return f"Agent processed: {response.text}"
    except requests.RequestException as e:
        return f"Agent request failed: {str(e)}"
    except Exception as e:
        return f"Agent processing failed: {str(e)}"


@app.route("/run-agent", methods=["POST"])
def run_agent_route():
    data = request.get_json(force=True) or {}
    user_input = data.get("input")
    agent_data = data.get("agent")
    result = run_agent(user_input, agent_data)
    agent_data = parse_to_dict(html.unescape(agent_data))
    agent_id = agent_data.get("id")
    # Fetch the latest agent from DB
    agent = asyncio.run(agent_service.get_agent_by_id(agent_id))
    if agent:
        new_total_runs = (agent.total_runs or 0) + 1
        asyncio.run(agent_service.update_agent_field(agent_id, "total_runs", new_total_runs))
    else:
        new_total_runs = None
    return jsonify({"response": result, "total_runs": new_total_runs})

@app.route('/api/creators')
def api_creators():
    """API endpoint for fetching creators."""
    search = request.args.get('search', '')
    sort_by = request.args.get('sort_by', 'reputation')
    
    try:
        all_creators = creator_service.fetch_creators()
        filtered_creators = creator_service.filter_and_sort_creators(all_creators, search, sort_by)
        return jsonify([creator.dict() for creator in filtered_creators])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== API Key Management Endpoints =====
import secrets
@app.route('/api/keys', methods=['GET'])
@login_required
def list_api_keys():
    """
    List API keys for the logged-in user. Returns masked keys only.
    """
    try:
        resp = supabase.table('api_keys').select('id, name, masked_key, expiration, expires_at, created_at').eq('user_id', current_user.id).order('created_at', desc=True).execute()
        # Supabase client returns a response with .data property
        data = getattr(resp, 'data', None) or (resp.data if hasattr(resp, 'data') else None) or resp
        # Ensure we return an array
        keys = data if isinstance(data, list) else []
        return jsonify(keys)
    except Exception as e:
        print('Error listing API keys:', e)
        return jsonify({'error': 'Failed to fetch API keys', 'details': str(e)}), 500

@app.route('/api/keys/<key_id>', methods=['DELETE'])
@login_required
def delete_api_key(key_id):
    """
    Delete an API key by id for the logged-in user.
    """
    try:
        # Revoke the API key instead of deleting so auditing is preserved
        resp = supabase_backend.table('api_keys').update({'status': 'revoked', 'updated_at': datetime.utcnow().isoformat()}).eq('id', key_id).eq('user_id', current_user.id).execute()
        return jsonify({'ok': True})
    except Exception as e:
        print('Error revoking API key:', e)
        return jsonify({'error': 'Failed to revoke API key', 'details': str(e)}), 500

@app.route('/api/keys', methods=['POST'])
@login_required
def create_api_key():
    """
    Create and persist an API key for the logged-in user into Supabase table 'api_keys'.
    Accepts optional JSON body: { name, expiration, key }.
    If key is not provided, server generates a secure key and returns it in the response once.
    """
    print("[create_api_key] called. SUPABASE_URL set:", bool(SUPABASE_URL), "SERVICE_ROLE_KEY set:", bool(SUPABASE_SERVICE_ROLE_KEY))

    data = request.get_json(silent=True) or {}
    # Avoid logging plaintext API keys; if provided, redact before printing
    redacted = dict(data) if isinstance(data, dict) else {}
    if 'key' in redacted:
        redacted['key'] = '[REDACTED]'
    print("[create_api_key] incoming data:", redacted)
    print("[create_api_key] current_user id:", getattr(current_user, 'id', None))

    name = data.get('name', 'Untitled Key')
    expiration = data.get('expiration', 'Never')
    key_val = data.get('key')

    # If no key provided, generate a secure server-side key
    generated = False
    if not key_val:
        generated = True
        key_val = generate_secret_key()

    # Compute expires_at if expiration is specified as e.g. '30 Days'
    expires_at = None
    if expiration and expiration != 'Never':
        try:
            days = int(str(expiration).split()[0])
            expires_at = (datetime.utcnow() + timedelta(days=days)).isoformat()
        except Exception:
            expires_at = None

    # Permissions and limits can be supplied by client; fall back to sensible defaults
    permissions = data.get('permissions') or {'chat': True, 'embeddings': True, 'tools': False}
    limits = data.get('limits') or {'rpm': 60, 'tpm': 100000, 'concurrency': 5}

    # Hash the API key before storing; do NOT store plaintext key
    hashed = hash_key(key_val)

    record = {
        'id': str(uuid.uuid4()),
        'user_id': current_user.id,
        'name': name,
        'hashed_key': hashed,
        'masked_key': mask_key(key_val),
        'status': 'active',
        'permissions': permissions,
        'limits': limits,
        'expiration': expiration,
        'expires_at': expires_at,
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat(),
    }

    try:
        resp = supabase_backend.table('api_keys').insert(record).execute()
        print('[create_api_key] supabase insert response:', getattr(resp, '__dict__', resp))

        # Return the full key only once (on creation) so client can show and copy it.
        return jsonify({'ok': True, 'id': record['id'], 'key': key_val, 'masked_key': record['masked_key']}), 201
    except APIError as e:
        # Handle missing column gracefully: older schemas may not have 'hashed_key'
        msg = getattr(e, 'args', [str(e)])[0]
        print('[create_api_key] APIError inserting key:', msg)
        if "Could not find the 'hashed_key'" in msg or 'PGRST204' in msg:
            try:
                # Fallback: store hashed value in legacy 'key' column (do NOT store plaintext)
                legacy_record = record.copy()
                legacy_record.pop('hashed_key', None)
                legacy_record['key'] = hashed
                resp2 = supabase_backend.table('api_keys').insert(legacy_record).execute()
                print('[create_api_key] fallback insert response (stored hash in key column):', getattr(resp2, '__dict__', resp2))
                return jsonify({'ok': True, 'id': legacy_record['id'], 'key': key_val, 'masked_key': legacy_record['masked_key'], 'note': 'stored-hash-in-legacy-key-column'}), 201
            except Exception as e2:
                import traceback
                print('[create_api_key] fallback insert failed:', e2)
                traceback.print_exc()
                return jsonify({'error': 'Failed to save API key (fallback)', 'details': str(e2)}), 500
        else:
            import traceback
            traceback.print_exc()
            return jsonify({'error': 'Failed to save API key', 'details': str(e)}), 500
    except Exception as e:
        import traceback
        print('Error saving API key:', e)
        traceback.print_exc()
        return jsonify({'error': 'Failed to save API key', 'details': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """404 error handler."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler."""
    return render_template('500.html'), 500

@app.route('/agent/<agent_id>/upvote', methods=['POST'])
def agent_upvote(agent_id):
    agent = asyncio.run(agent_service.get_agent_by_id(agent_id))
    if not agent:
        return jsonify({'error': 'Agent not found'}), 404
    new_upvotes = (agent.upvotes or 0) + 1
    asyncio.run(agent_service.update_agent_field(agent_id, 'upvotes', new_upvotes))
    return jsonify({'upvotes': new_upvotes})

@app.route('/agent/<agent_id>/rate', methods=['POST'])
def agent_rate(agent_id):
    agent = asyncio.run(agent_service.get_agent_by_id(agent_id))
    if not agent:
        return jsonify({'error': 'Agent not found'}), 404
    rating = request.json.get('rating', 0)
    # For demo: just set avg_rating to new rating (implement real average logic as needed)
    asyncio.run(agent_service.update_agent_field(agent_id, 'avg_rating', rating))
    return jsonify({'avg_rating': rating})

@app.route('/agent/<agent_id>/version', methods=['POST'])
def agent_version(agent_id):
    agent = asyncio.run(agent_service.get_agent_by_id(agent_id))
    if not agent:
        return jsonify({'error': 'Agent not found'}), 404
    version = request.json.get('version', '1.0.0')
    asyncio.run(agent_service.update_agent_field(agent_id, 'version', version))
    return jsonify({'version': version})

@app.route('/join-waitlist', methods=['POST'])
def join_waitlist():
    """Handle waitlist signups"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        # Validate email
        if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return jsonify({
                'success': False, 
                'message': 'Please enter a valid email address.'
            }), 400
        
        # Get user ID if authenticated
        user_id = None
        if current_user.is_authenticated:
            user_id = current_user.id
        
        # Prepare data for insertion
        waitlist_data = {'email': email}
        if user_id:
            waitlist_data['user_id'] = user_id
        
        # Check if email already exists in waitlist
        existing_entry = supabase.table('waitlist').select('email').eq('email', email).execute()
        
        if existing_entry.data:
            # Send welcome email asynchronously even if already registered
            email_service = get_email_service()
            receiver_email = email
            subject = "Welcome to the Agent Architects Waitlist!"
            body = "Thanks for signing up we will keep you posted :)"
            
            # Prepare content
            name = "there" # Main waitlist doesn't capture name in this route
            plain_body = f"Hello {name},\n\nWe are absolutely thrilled that you took the time to sign up for our waitlist! 🚀\n\nWe are currently working hard behind the scenes to build something special. We will notify you the moment we are ready.\n\nIn the meantime, feel free to explore our website.\n\nWarm regards,\nThe Manhattan Project Team"
            
            html_body = f"""
            <div style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
              <p>Hello {name},</p>
              <p>We are absolutely thrilled that you took the time to sign up for our waitlist! 🚀</p>
              <p>We are currently working hard behind the scenes to build something special, and we can't wait to share it with you. We will notify you the moment we are ready to onboard you.</p>
              <p>In the meantime, please feel free to explore our website and get a feel for what we are building.</p>
              <p>Warm regards,</p>
              <p>The Manhattan Project Team</p>
              
              <table role="presentation" border="0" cellpadding="0" cellspacing="0" style="margin-top: 30px; border-top: 1px solid #eee; padding-top: 15px;">
                <tr>
                   <td style="vertical-align: middle; padding-right: 12px;">
                      <img src="cid:logo" width="30" height="30" style="display: block;" alt="Logo">
                   </td>
                   <td style="vertical-align: middle;">
                      <span style="font-family: 'Mr Dafoe', cursive, serif; font-size: 26px; color: #EC4899; line-height: 1;">The Manhattan Project</span>
                   </td>
                </tr>
              </table>
            </div>
            """

            # Path to logo
            logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'favicon.svg')

            # Send asynchronously to avoid blocking
            email_service.send_email_async(receiver_email, subject, plain_body, html_body=html_body, image_attachment_path=logo_path)

            # Get current count
            count_result = supabase.table('waitlist').select('id', count='exact').execute()
            return jsonify({
                'success': True,
                'message': 'Email already registered',
                'new_count': count_result.count + 114,
                'already_registered': True
            })
        
        print("Waitlist data to insert:", waitlist_data)
        # Insert new email (with user_id if available)
        insert_result = supabase.table('waitlist').insert(waitlist_data).execute()
        
        if insert_result.data:
            # Send welcome email asynchronously
            email_service = get_email_service()
            receiver_email = email
            subject = "Welcome to the Agent Architects Waitlist!"
            # Prepare content
            name = "there"
            plain_body = f"Hello {name},\n\nWe are absolutely thrilled that you took the time to sign up for our waitlist! 🚀\n\nWe are currently working hard behind the scenes to build something special. We will notify you the moment we are ready.\n\nIn the meantime, feel free to explore our website.\n\nWarm regards,\nThe Manhattan Project Team"
            
            html_body = f"""
            <div style="font-family: 'Inter', Arial, sans-serif; color: #333; line-height: 1.6;">
              <p>Hello {name},</p>
              <p>We are absolutely thrilled that you took the time to sign up for our waitlist! 🚀</p>
              <p>We are currently working hard behind the scenes to build something special, and we can't wait to share it with you. We will notify you the moment we are ready to onboard you.</p>
              <p>In the meantime, please feel free to explore our website and get a feel for what we are building.</p>
              <p>Warm regards,</p>
              <p>The Manhattan Project Team</p>
              
              <table role="presentation" border="0" cellpadding="0" cellspacing="0" style="margin-top: 30px; border-top: 1px solid #eee; padding-top: 15px;">
                <tr>
                   <td style="vertical-align: middle; padding-right: 12px;">
                      <img src="cid:logo" width="30" height="30" style="display: block;" alt="Logo">
                   </td>
                   <td style="vertical-align: middle;">
                      <span style="font-family: 'Mr Dafoe', cursive, serif; font-size: 26px; color: #EC4899; line-height: 1;">The Manhattan Project</span>
                   </td>
                </tr>
              </table>
            </div>
            """
            
            # Path to logo
            logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'favicon.svg')

            # Send asynchronously to avoid blocking
            email_service.send_email_async(receiver_email, subject, plain_body, html_body=html_body, image_attachment_path=logo_path)
            
            # Get updated count
            count_result = supabase.table('waitlist').select('id', count='exact').execute()
            return jsonify({
                'success': True,
                'message': 'Successfully joined waitlist',
                'new_count': count_result.count + 114
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to add to waitlist'
            }), 500
            
    except Exception as e:
        print(f"Error in join_waitlist: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500

@app.route('/api/agent-count')
def get_agent_count():
    """Get the current number of agents from Supabase"""
    try:
        # Query the agents table to get the count
        count = supabase.table('agents').select('id', count='exact').execute()
        return jsonify({'count': count.count})
    except Exception as e:
        print(f"Error fetching agent count: {e}")
        # Return a default value if there's an error
        return jsonify({'count': 1000})

def update_waitlist_count():
    """Helper function to get current waitlist count"""
    try:
        count_result = supabase.table('waitlist').select('id', count='exact').execute()
        return (count_result.count + 114)  # Starting offset
    except Exception as e:
        print(f"Error getting waitlist count: {e}")
        return 114  # Default fallback


@app.route('/join-gitmem-waitlist', methods=['POST'])
def join_gitmem_waitlist():
    """Handle GitMem waitlist signups with full details"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        # Validate email
        if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return jsonify({
                'success': False, 
                'message': 'Please enter a valid email address.'
            }), 400
        
        # Check if email already exists in gitmem_waitlist
        try:
            existing_entry = supabase.table('gitmem_waitlist').select('email').eq('email', email).execute()
            if existing_entry.data:
                # Send welcome email asynchronously even if already registered
                email_service = get_email_service()
                receiver_email = email
                subject = "Welcome to GitMem Waitlist"
                # Prepare content
                # Prepare content
                name = gitmem_data.get('name') if 'gitmem_data' in locals() else "there"
                input_name = data.get('name', '').strip() or "there"
                
                plain_body = f"Hello {input_name},\n\nWe are absolutely thrilled that you took the time to sign up for our waitlist! 🚀\n\nWe are currently working hard behind the scenes to build something special. We will notify you the moment we are ready.\n\nIn the meantime, feel free to explore our website.\n\nWarm regards,\nThe Manhattan Project Team"
                
                html_body = f"""
                <div style="font-family: 'Inter', Arial, sans-serif; color: #333; line-height: 1.6;">
                  <p>Hello {input_name},</p>
                  <p>We are absolutely thrilled that you took the time to sign up for our waitlist! 🚀</p>
                  <p>We are currently working hard behind the scenes to build something special, and we can't wait to share it with you. We will notify you the moment we are ready to onboard you.</p>
                  <p>In the meantime, please feel free to explore our website and get a feel for what we are building.</p>
                  <p>Warm regards,</p>
                  <p>The Manhattan Project Team</p>
                  
                  <table role="presentation" border="0" cellpadding="0" cellspacing="0" style="margin-top: 30px; border-top: 1px solid #eee; padding-top: 15px;">
                    <tr>
                       <td style="vertical-align: middle; padding-right: 12px;">
                          <img src="cid:logo" width="30" height="30" style="display: block;" alt="Logo">
                       </td>
                       <td style="vertical-align: middle;">
                          <span style="font-family: 'Mr Dafoe', cursive, serif; font-size: 26px; color: #EC4899; line-height: 1;">The Manhattan Project</span>
                       </td>
                    </tr>
                  </table>
                </div>
                """
                
                # Path to logo
                logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'logo.png')

                # Send asynchronously to avoid blocking
                email_service.send_email_async(receiver_email, subject, plain_body, html_body=html_body, image_attachment_path=logo_path)
                
                return jsonify({
                    'success': True,
                    'message': 'Email already registered for GitMem',
                    'already_registered': True
                })
        except Exception as e:
            print(f"Error checking existing GitMem entry: {e}")
        
        # Prepare data for insertion
        gitmem_data = {
            'email': email,
            'name': data.get('name', '').strip() or None,
            'tools': data.get('tools', ''),
            'stack': data.get('stack', ''),
            'goals': data.get('goals', ''),
            'setup': data.get('setup', ''),
            'open_to_feedback': data.get('open_to_feedback', False),
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Get user ID if authenticated
        if current_user.is_authenticated:
            gitmem_data['user_id'] = current_user.id
        
        print("GitMem waitlist data to insert:", gitmem_data)
        
        # Insert new entry
        insert_result = supabase.table('gitmem_waitlist').insert(gitmem_data).execute()
        
        if insert_result.data:
            # Send welcome email asynchronously
            email_service = get_email_service()
            receiver_email = email
            subject = "Welcome to GitMem Waitlist"
            # Prepare content
            # Prepare content
            name = gitmem_data.get('name') or "there"
            plain_body = f"Hello {name},\n\nWe are absolutely thrilled that you took the time to sign up for our waitlist! 🚀\n\nWe are currently working hard behind the scenes to build something special. We will notify you the moment we are ready.\n\nIn the meantime, feel free to explore our website.\n\nWarm regards,\nThe Manhattan Project Team"
            
            html_body = f"""
            <div style="font-family: 'Inter', Arial, sans-serif; color: #333; line-height: 1.6;">
              <p>Hello {name},</p>
              <p>We are absolutely thrilled that you took the time to sign up for our waitlist! 🚀</p>
              <p>We are currently working hard behind the scenes to build something special, and we can't wait to share it with you. We will notify you the moment we are ready to onboard you.</p>
              <p>In the meantime, please feel free to explore our website and get a feel for what we are building.</p>
              <p>Warm regards,</p>
              <p>The Manhattan Project Team</p>
              
              <table role="presentation" border="0" cellpadding="0" cellspacing="0" style="margin-top: 30px; border-top: 1px solid #eee; padding-top: 15px;">
                <tr>
                   <td style="vertical-align: middle; padding-right: 12px;">
                      <img src="cid:logo" width="30" height="30" style="display: block;" alt="Logo">
                   </td>
                   <td style="vertical-align: middle;">
                      <span style="font-family: 'Mr Dafoe', cursive, serif; font-size: 26px; color: #EC4899; line-height: 1;">The Manhattan Project</span>
                   </td>
                </tr>
              </table>
            </div>
            """
            
            # Path to logo
            logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'logo.png')

            # Send asynchronously to avoid blocking
            email_service.send_email_async(receiver_email, subject, plain_body, html_body=html_body, image_attachment_path=logo_path)
            
            return jsonify({
                'success': True,
                'message': 'Successfully joined GitMem waitlist'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to add to waitlist'
            }), 500
            
    except Exception as e:
        print(f"Error in join_gitmem_waitlist: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500

@app.route('/memory', methods=['GET', 'POST'])
@login_required
def memory():
    if request.method == 'POST':
        try:
            text = request.form.get('memory_text')
            files = request.files.getlist('memory_file')  # Accept multiple files

            allowed_extensions = [
                '.pdf', '.ppt', '.pptx', '.doc', '.docx', '.txt',
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg',
                '.webp', '.heic', '.tiff', '.xls', '.xlsx', '.csv'
            ]

            controller = RAG_DB_Controller_FILE_DATA()
            file_count = 0
            error_count = 0
            
            # Handle multiple files
            for file in files:
                if file and file.filename:
                    filename = file.filename
                    ext = os.path.splitext(filename)[1].lower()
                    
                    # Validate file extension
                    if ext not in allowed_extensions:
                        print(f"[FILE_UPLOAD] Unsupported file type: {filename}")
                        flash(f'Unsupported file type: {filename}', 'error')
                        error_count += 1
                        continue
                    
                    try:
                        # Create temp file and save
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            file.save(tmp.name)
                            file_path = tmp.name
                        
                        # Check if file was actually created
                        if not os.path.exists(file_path):
                            print(f"[FILE_UPLOAD] Failed to create temp file for: {filename}")
                            flash(f'Failed to save file: {filename}', 'error')
                            error_count += 1
                            continue
                        
                        print(f"[FILE_UPLOAD] Processing file: {filename}, path: {file_path}")
                        
                        # Send to RAG database
                        controller.update_file_data_to_db(
                            user_ID=str(current_user.id),
                            file_path=file_path,
                            message_type="user",
                            file_name=filename
                        )
                        
                        print(f"[FILE_UPLOAD] Successfully processed: {filename}")
                        file_count += 1
                        
                        # Clean up temp file
                        try:
                            os.remove(file_path)
                            print(f"[FILE_UPLOAD] Cleaned up temp file: {file_path}")
                        except Exception as e:
                            print(f"[FILE_UPLOAD] Error deleting temp file {file_path}: {e}")
                    
                    except Exception as e:
                        print(f"[FILE_UPLOAD] Error processing file {filename}: {str(e)}")
                        flash(f'Error processing file: {filename}', 'error')
                        error_count += 1
                        continue

            # Handle plain text (no file case) → send directly to DB
            if text and text.strip():
                try:
                    print(f"[TEXT_UPLOAD] Saving text memory from user: {current_user.id}")
                    controller.send_data_to_rag_db(
                        user_ID=str(current_user.id),
                        chunks=[text],
                        message_type="user"
                    )
                    print(f"[TEXT_UPLOAD] Successfully saved text memory")
                except Exception as e:
                    print(f"[TEXT_UPLOAD] Error saving text: {str(e)}")
                    flash(f'Error saving text: {str(e)}', 'error')

            # Provide feedback to user
            if file_count > 0:
                flash(f'Successfully uploaded {file_count} file(s)', 'success')
            if error_count > 0:
                flash(f'{error_count} file(s) failed to upload', 'error')
            if not text and not files:
                flash('Please provide text or upload at least one file', 'warning')

            print(f"[MEMORY_UPLOAD] Complete - Files: {file_count}, Errors: {error_count}")
            return redirect(url_for('memory'))

        except Exception as e:
            print(f"[MEMORY_UPLOAD] Unexpected error: {str(e)}")
            flash(f'An unexpected error occurred: {str(e)}', 'error')
            return redirect(url_for('memory'))

    return render_template('memory.html', user=current_user)

# @app.post("/api/chat")
# @login_required
# def chat():
#     data = request.get_json(force=True)
#     user_msg = data.get("message", "")
#     session_id = data.get("session_id")
#     history = data.get("context", [])

#     # Call your Python AI function here
#     reply_text = run_ai(user_msg, history=history, session_id=session_id)

#     # Optionally return RAG results for the right panel
#     rag_results = [
#         {"id": 1, "score": 0.92, "text": "...", "source": "...", "timestamp": "...", "matches": ["..."]}
#     ]

#     return jsonify({"reply": reply_text, "rag_results": rag_results})

# def run_ai(message, history, session_id):
#     # your model / tool-calling / RAG pipeline
#     # This function should call LLM responses from the Response controller.
#     return f"Echo This is the AI response: {message}"  # replace with real response

from api_chats import api
app.register_blueprint(api)

# Initialize keep-alive background task to prevent Render from sleeping
keep_alive_task()

# Register Manhattan API blueprint (simple ping/health endpoints)
try:
    from api_manhattan import manhattan_api
    app.register_blueprint(manhattan_api)
except Exception as e:
    print('[STARTUP] Could not register manhattan_api blueprint:', e)

try:
    from my_agents import apis_my_agents
    app.register_blueprint(apis_my_agents)

except Exception as e:
    print('[STARTUP] Could not register apis_my_agents blueprint:', e)

# Routes
@app.route('/')
def homepage():
    """Redirect root to memory page."""
    return render_template('homepage.html', user=current_user)


@app.route('/api/docs')
def api_docs():
    """Render the API documentation placeholder page."""
    # Attempt to load the static docs JSON and inject into the template to avoid client-side fetch issues
    docs_path = os.path.join(STATIC_DIR, 'index.json')
    docs_json = None
    try:
        with open(docs_path, 'r', encoding='utf-8') as f:
            docs_json = json.load(f)
    except Exception as e:
        print('[STARTUP] Could not load static/index.json:', e)

    # Pass serialized JSON (or null) to the template. The template will use this as INITIAL_DOCS.
    return render_template('api_docs.html', docs_json=json.dumps(docs_json) if docs_json is not None else None)


# MCP SSE endpoint is now handled by the mcp_bp blueprint registered above
# See mcp_socketio_gateway.py for implementation

if __name__ == '__main__':
    # MCP SSE is now served via the mcp_bp blueprint (no separate thread needed)
    if socketio:
        # Run with SocketIO for WebSocket support
        print("[STARTUP] Running with Flask-SocketIO (WebSocket enabled)")
        socketio.run(app, debug=True, host='0.0.0.0', port=1078, allow_unsafe_werkzeug=True)
    else:
        # Fallback to standard Flask
        print("[STARTUP] Running with standard Flask (no WebSocket)")
        app.run(debug=True, host='0.0.0.0', port=1078)
