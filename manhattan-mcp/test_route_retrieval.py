"""
Test: Route-style query retrieval

Verifies that:
1. Queries with route-style tokens (e.g., /api/keys) are NOT treated as file filters
2. Route tokens are decomposed into individual keywords for matching
3. Content-based matching boosts chunks that contain the route pattern
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import shutil
import json

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

# Setup
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_route_retrieval_data")
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)

print("=" * 60)
print("TEST: Route-Style Query Retrieval")
print("=" * 60)

api = CodingAPI(root_path=TEST_DIR)
AGENT_ID = "test_agent"
FILE_PATH = os.path.normpath("/src/api/index.py")

# Simulate chunks similar to what's stored for the real index.py
SEMANTIC_CHUNKS = [
    {
        "content": "@app.route('/api/keys', methods=['GET', 'DELETE'])\n@login_required\ndef api_keys():\n    if request.method == 'GET':\n        keys = supabase.table('api_keys').select('*').eq('user_id', current_user.id).execute()\n        return jsonify(keys.data)\n    elif request.method == 'DELETE':\n        key_id = request.json.get('key_id')\n        supabase.table('api_keys').update({'status': 'revoked'}).eq('id', key_id).execute()",
        "type": "block",
        "name": "API Key Management Routes",
        "start_line": 872,
        "end_line": 892,
        "keywords": ["api", "keys", "management", "security", "masking", "revocation", "route", "endpoint"],
        "summary": "Provides GET endpoint to list user's masked API keys and DELETE endpoint to revoke API keys."
    },
    {
        "content": "@app.route('/api/keys/create', methods=['POST'])\n@login_required\ndef create_api_key():\n    key = generate_secret_key()\n    hashed = hash_key(key)\n    supabase.table('api_keys').insert({'key_hash': hashed, 'user_id': current_user.id}).execute()\n    return jsonify({'key': key})",
        "type": "function",
        "name": "create_api_key Endpoint",
        "start_line": 894,
        "end_line": 957,
        "keywords": ["api", "key-creation", "hashing", "security", "endpoint", "permissions", "limits"],
        "summary": "POST endpoint to create new API key for user. Generates secure key, hashes before storing."
    },
    {
        "content": "def authenticate_user(username, password):\n    user = db.find_user(username)\n    if user and user.check_password(password):\n        return create_session(user)\n    raise AuthenticationError('Invalid credentials')",
        "type": "function",
        "name": "authenticate_user",
        "start_line": 426,
        "end_line": 450,
        "keywords": ["auth", "login", "password", "session", "authentication"],
        "summary": "Authenticates a user by checking username/password against the database."
    },
    {
        "content": "@app.route('/health')\ndef health_check():\n    return jsonify({'status': 'healthy'})",
        "type": "function",
        "name": "health_check",
        "start_line": 219,
        "end_line": 229,
        "keywords": ["health", "monitoring", "status", "endpoint"],
        "summary": "Health check endpoint returning system status."
    }
]

# 1. Create flow with chunks
print("\n[1] Creating flow with API key chunks...")
result = api.create_flow(AGENT_ID, FILE_PATH, chunks=SEMANTIC_CHUNKS)
print(f"    Created: {result.get('status')}")

# 2. Test: Query with route-style token (the bug case)
print("\n[2] Testing query: '/api/keys endpoint from index.py'...")
search_result = api.get_flow(AGENT_ID, "/api/keys endpoint from index.py")
print(f"    Status: {search_result.get('status')}")
print(f"    Filter: {search_result.get('filter')}")
print(f"    Query: {search_result.get('query')}")
results = search_result.get("results", [])
print(f"    Results count: {len(results)}")

assert len(results) > 0, "FAIL: No results for '/api/keys endpoint from index.py'!"

# Verify the top result is an API key management chunk
top_chunk = results[0]["chunk"]
print(f"    Top result: {top_chunk.get('name')} (score={results[0]['score']:.4f})")
assert "api" in top_chunk.get("name", "").lower() or "key" in top_chunk.get("name", "").lower(), \
    f"FAIL: Top result should be API key related, got: {top_chunk.get('name')}"
print("    ✅ Route-style query returns correct results!")

# 3. Test: File filter is correctly extracted for actual file paths  
print("\n[3] Testing that index.py is correctly used as file filter...")
search_result2 = api.get_flow(AGENT_ID, "authentication from index.py")
print(f"    Filter: {search_result2.get('filter')}")
assert search_result2.get("filter") is not None, "FAIL: index.py should be used as file filter!"
assert "index.py" in search_result2.get("filter", ""), "FAIL: filter should contain 'index.py'"
print("    ✅ File path correctly extracted as filter!")

# 4. Test: Route-only query (no file path in query)
print("\n[4] Testing query: 'explain /api/keys'...")
search_result3 = api.get_flow(AGENT_ID, "explain /api/keys")
print(f"    Filter: {search_result3.get('filter')}")
print(f"    Results count: {search_result3.get('count')}")
assert search_result3.get("filter") is None, "FAIL: /api/keys should NOT be treated as file filter!"
assert search_result3.get("count", 0) > 0, "FAIL: Should return results for 'explain /api/keys'!"
print("    ✅ Route-only query works correctly!")

# 5. Test: Query with just keywords (baseline)
print("\n[5] Testing keyword query: 'api key management'...")
search_result4 = api.get_flow(AGENT_ID, "api key management")
print(f"    Results count: {search_result4.get('count')}")
assert search_result4.get("count", 0) > 0, "FAIL: keyword query should return results!"
top4 = search_result4["results"][0]["chunk"]
print(f"    Top result: {top4.get('name')}")
print("    ✅ Keyword query works correctly!")

# 6. Test: Query should NOT match unrelated chunks
print("\n[6] Verifying relevance: health_check should not be top result for /api/keys...")
for r in results[:2]:
    assert r["chunk"].get("name") != "health_check", \
        "FAIL: health_check should not be a top result for /api/keys query!"
print("    ✅ Irrelevant chunks correctly ranked lower!")

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("  - Route-style tokens NOT treated as file filters ✅")
print("  - Route tokens decomposed into keywords ✅")
print("  - File extensions correctly identify file paths ✅")
print("  - Content-based route matching works ✅")
print("=" * 60)

# Cleanup
shutil.rmtree(TEST_DIR)
