"""
Comprehensive Retrieval Quality Test

Tests the create_flow and get_flow pipeline with complex, realistic queries.
Validates that the hybrid retriever returns relevant, well-scored results.
"""
import sys
import os
import shutil
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_retrieval_quality")
AGENT_ID = "qa_test_agent"


def setup():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    return CodingAPI(root_path=TEST_DIR)


def print_result(query, results, expected_top_name=None, expected_min_count=1):
    """Print and validate a query result."""
    count = len(results)
    status = "✅" if count >= expected_min_count else "❌"
    
    print(f"\n{'─'*60}")
    print(f"  Query: \"{query}\"")
    print(f"  Results: {count} (expected >= {expected_min_count}) {status}")
    
    for i, r in enumerate(results[:5]):
        name = r["chunk"]["name"]
        score = r["score"]
        vec = r["vector_score"]
        kw = r["keyword_score"]
        ctype = r["chunk"].get("type", "?")
        print(f"    [{i+1}] {name} ({ctype}) — score={score:.4f} (vec={vec:.4f}, kw={kw:.4f})")
    
    if expected_top_name and results:
        top_name = results[0]["chunk"]["name"]
        match = "✅" if expected_top_name.lower() in top_name.lower() else "❌"
        print(f"  Top match check: expected '{expected_top_name}' in '{top_name}' → {match}")
        return expected_top_name.lower() in top_name.lower()
    return count >= expected_min_count


def main():
    api = setup()
    
    # ── Ingest rich chunks ──────────────────────────────────────────────
    print("="*60)
    print("STEP 1: Ingesting code chunks")
    print("="*60)
    
    chunks = [
        {
            "content": "class AuthManager:\n    def __init__(self, db_client, jwt_secret):\n        self.db = db_client\n        self.jwt_secret = jwt_secret\n        self.token_expiry = 3600",
            "end_line": 60, "start_line": 1,
            "keywords": ["AuthManager", "authentication", "jwt", "session", "token", "security", "db_client"],
            "name": "AuthManager",
            "summary": "Central authentication manager. Initializes with a database client and JWT secret. Manages user sessions with configurable token expiry (default 3600s). Handles login, logout, token refresh, and password reset flows.",
            "type": "class"
        },
        {
            "content": "def login(self, email, password):\n    user = self.db.query('users').where('email', email).first()\n    if not user:\n        raise AuthError('User not found')\n    if not bcrypt.verify(password, user.hashed_password):\n        raise AuthError('Invalid password')\n    token = jwt.encode({'user_id': user.id, 'exp': time.time() + self.token_expiry}, self.jwt_secret)\n    self.db.insert('sessions', {'user_id': user.id, 'token': token})\n    return {'token': token, 'user': user.to_dict()}",
            "end_line": 25, "start_line": 10,
            "keywords": ["login", "email", "password", "bcrypt", "jwt", "token", "session", "authenticate", "AuthError", "hashed_password"],
            "name": "AuthManager.login",
            "summary": "Authenticates user by email and password. Queries database for user, verifies password with bcrypt, generates JWT token with user_id and expiry, stores session in DB, returns token and user dict. Raises AuthError on failure.",
            "type": "function"
        },
        {
            "content": "def refresh_token(self, old_token):\n    payload = jwt.decode(old_token, self.jwt_secret, algorithms=['HS256'])\n    new_token = jwt.encode({'user_id': payload['user_id'], 'exp': time.time() + self.token_expiry}, self.jwt_secret)\n    self.db.update('sessions', {'token': new_token}, where={'user_id': payload['user_id']})\n    return new_token",
            "end_line": 35, "start_line": 27,
            "keywords": ["refresh", "token", "jwt", "decode", "encode", "session", "expiry", "HS256"],
            "name": "AuthManager.refresh_token",
            "summary": "Refreshes an expired JWT token. Decodes old token to extract user_id, generates new JWT with fresh expiry, updates session record in database. Returns new token string.",
            "type": "function"
        },
        {
            "content": "def reset_password(self, email):\n    user = self.db.query('users').where('email', email).first()\n    if not user:\n        raise AuthError('User not found')\n    reset_token = secrets.token_urlsafe(32)\n    self.db.insert('password_resets', {'user_id': user.id, 'token': reset_token, 'expires_at': time.time() + 1800})\n    send_email(email, 'Password Reset', f'Reset link: /reset?token={reset_token}')\n    return {'status': 'reset_email_sent'}",
            "end_line": 48, "start_line": 37,
            "keywords": ["reset", "password", "email", "token", "secrets", "password_resets", "send_email", "expiry"],
            "name": "AuthManager.reset_password",
            "summary": "Initiates password reset flow. Looks up user by email, generates secure URL-safe reset token (32 bytes), stores in password_resets table with 30-minute expiry, sends reset email with link. Returns status dict.",
            "type": "function"
        },
        {
            "content": "class RateLimiter:\n    def __init__(self, redis_client, max_requests=100, window_seconds=60):\n        self.redis = redis_client\n        self.max_requests = max_requests\n        self.window = window_seconds\n    \n    def check(self, client_ip):\n        key = f'rate:{client_ip}'\n        count = self.redis.incr(key)\n        if count == 1:\n            self.redis.expire(key, self.window)\n        return count <= self.max_requests",
            "end_line": 80, "start_line": 62,
            "keywords": ["RateLimiter", "redis", "rate_limit", "throttle", "middleware", "API", "client_ip", "sliding_window", "max_requests"],
            "name": "RateLimiter",
            "summary": "Rate limiter using Redis sliding window. Configurable max requests per time window (default 100/60s). check() method increments counter for client IP in Redis, sets TTL on first request, returns True if under limit. Used as middleware for API endpoint protection.",
            "type": "class"
        },
        {
            "content": "def validate_email_format(email: str) -> bool:\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))",
            "end_line": 85, "start_line": 82,
            "keywords": ["validate", "email", "regex", "format", "pattern", "validation"],
            "name": "validate_email_format",
            "summary": "Validates email format using regex pattern. Checks for standard email structure. Returns boolean.",
            "type": "function"
        },
        {
            "content": "class DatabaseMigration:\n    def __init__(self, db_url):\n        self.engine = create_engine(db_url)\n        self.migrations = []\n    \n    def add_migration(self, name, up_sql, down_sql):\n        self.migrations.append({'name': name, 'up': up_sql, 'down': down_sql})\n    \n    def run_up(self):\n        for m in self.migrations:\n            self.engine.execute(m['up'])\n            self.engine.execute(\"INSERT INTO migration_log VALUES (%s)\", m['name'])\n    \n    def rollback(self):\n        if self.migrations:\n            last = self.migrations[-1]\n            self.engine.execute(last['down'])\n            self.engine.execute(\"DELETE FROM migration_log WHERE name = %s\", last['name'])",
            "end_line": 120, "start_line": 90,
            "keywords": ["DatabaseMigration", "migration", "schema", "SQLAlchemy", "rollback", "up_sql", "down_sql", "migration_log", "database", "version_control"],
            "name": "DatabaseMigration",
            "summary": "Database migration manager using SQLAlchemy engine. Supports adding named migrations with up/down SQL, running migrations forward, and rolling back. Tracks history in migration_log table. Used for schema version control.",
            "type": "class"
        },
        {
            "content": "def cache_result(ttl_seconds=300):\n    def decorator(func):\n        _cache = {}\n        def wrapper(*args, **kwargs):\n            key = str(args) + str(sorted(kwargs.items()))\n            if key in _cache and time.time() - _cache[key]['ts'] < ttl_seconds:\n                return _cache[key]['value']\n            result = func(*args, **kwargs)\n            _cache[key] = {'value': result, 'ts': time.time()}\n            return result\n        return wrapper\n    return decorator",
            "end_line": 140, "start_line": 125,
            "keywords": ["cache", "decorator", "TTL", "memoize", "caching", "performance", "in_memory", "cache_result"],
            "name": "cache_result",
            "summary": "Decorator factory for caching function results with configurable TTL (default 300s). Creates in-memory dict cache. Useful for expensive computations or API calls.",
            "type": "function"
        },
    ]

    result = api.create_flow(AGENT_ID, "/app/src/auth_manager.py", chunks)
    print(f"  Ingested: {result}")
    
    # Verify vectors
    vecs = api.vector_store._load_vectors(AGENT_ID)
    print(f"  Vectors stored: {len(vecs)}")
    assert len(vecs) == 8, f"Expected 8 vectors, got {len(vecs)}"
    print("  ✅ All chunks embedded successfully\n")

    # ── Run Retrieval Tests ─────────────────────────────────────────────
    print("="*60)
    print("STEP 2: Retrieval Quality Tests")
    print("="*60)

    passed = 0
    total = 0

    # Test 1: Direct concept query
    total += 1
    res = api.get_flow(AGENT_ID, "How does the system handle password reset when a user forgets their password?")
    if print_result(
        "How does the system handle password reset when a user forgets their password?",
        res["results"], expected_top_name="reset_password"
    ):
        passed += 1

    # Test 2: Indirect concept query (no keyword "rate" in query)
    total += 1
    res = api.get_flow(AGENT_ID, "What protects API endpoints from being overwhelmed by too many requests?")
    if print_result(
        "What protects API endpoints from being overwhelmed by too many requests?",
        res["results"], expected_top_name="RateLimiter"
    ):
        passed += 1

    # Test 3: Schema management
    total += 1
    res = api.get_flow(AGENT_ID, "How are database schema changes managed and versioned?")
    if print_result(
        "How are database schema changes managed and versioned?",
        res["results"], expected_top_name="DatabaseMigration"
    ):
        passed += 1

    # Test 4: Token expiry
    total += 1
    res = api.get_flow(AGENT_ID, "What happens when a JWT token expires and the user needs a new one?")
    if print_result(
        "What happens when a JWT token expires and the user needs a new one?",
        res["results"], expected_top_name="refresh_token"
    ):
        passed += 1

    # Test 5: Broad security query (should return multiple results)
    total += 1
    res = api.get_flow(AGENT_ID, "Show me all the security related code", top_k=8)
    ok = print_result(
        "Show me all the security related code",
        res["results"], expected_min_count=3
    )
    if ok and len(res["results"]) >= 3:
        passed += 1

    # Test 6: Memoization (synonym matching)
    total += 1
    res = api.get_flow(AGENT_ID, "How do I add memoization to improve performance of expensive function calls?")
    if print_result(
        "How do I add memoization to improve performance of expensive function calls?",
        res["results"], expected_top_name="cache_result"
    ):
        passed += 1

    # Test 7: DB table interaction
    total += 1
    res = api.get_flow(AGENT_ID, "Which functions interact with the users table in the database?")
    ok = print_result(
        "Which functions interact with the users table in the database?",
        res["results"], expected_min_count=2
    )
    if ok and len(res["results"]) >= 2:
        names = [r["chunk"]["name"] for r in res["results"][:3]]
        print(f"    Names in top 3: {names}")
        # Both login and reset_password query the users table
        if any("login" in n.lower() for n in names) and any("reset" in n.lower() for n in names):
            passed += 1
            print(f"    ✅ Both login and reset_password found!")
        else:
            print(f"    ⚠️ Missing one of login/reset_password in top results")

    # Test 8: Specific symbol query
    total += 1
    res = api.get_flow(AGENT_ID, "How is bcrypt used in the codebase?")
    if print_result(
        "How is bcrypt used in the codebase?",
        res["results"], expected_top_name="login"
    ):
        passed += 1

    # Test 9: Line number query (should prefer tight ranges)
    total += 1
    res = api.get_flow(AGENT_ID, "What code is around line 30?")
    results = res["results"]
    if results:
        top_name = results[0]["chunk"]["name"]
        # refresh_token (27-35) is tighter than AuthManager (1-60)
        is_tight = "refresh_token" in top_name.lower()
        status = "✅" if is_tight else "⚠️"
        print_result("What code is around line 30?", results)
        print(f"    Tight match check: {status} (got '{top_name}')")
        if is_tight:
            passed += 1
    total_adj = total - 0  # no adjustment

    # Test 10: Parent-child relationship query
    total += 1
    res = api.get_flow(AGENT_ID, "What methods does AuthManager have?")
    ok = print_result(
        "What methods does AuthManager have?",
        res["results"], expected_min_count=3
    )
    if ok and len(res["results"]) >= 3:
        passed += 1

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RETRIEVAL QUALITY RESULTS: {passed}/{total} tests passed")
    print(f"{'='*60}")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Retrieval quality is excellent.")
    elif passed >= total * 0.7:
        print("✅ Most tests passed. Retrieval quality is good.")
    else:
        print("⚠️ Several tests failed. Further improvements needed.")

    # Cleanup
    shutil.rmtree(TEST_DIR)
    return passed, total


if __name__ == "__main__":
    try:
        passed, total = main()
        sys.exit(0 if passed >= total * 0.7 else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
