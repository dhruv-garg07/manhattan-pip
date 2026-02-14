"""
Test: Semantic Chunking with Dedicated vectors.json

Verifies that:
1. Vectors are stored in a dedicated agents/{agent_id}/vectors.json
2. Chunks in file_contexts.json do NOT have inline vector fields
3. Global chunks.json does NOT have inline vectors
4. Hybrid search retrieves using vectors from vectors.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import shutil
import json

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

# Setup
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_semantic_vectors")
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)

print("=" * 60)
print("TEST: Dedicated vectors.json for Code Chunks")
print("=" * 60)

api = CodingAPI(root_path=TEST_DIR)
AGENT_ID = "test_agent"
FILE_PATH = os.path.normpath("/src/example.py")

# Simulate what Claude would provide as semantic chunks
SEMANTIC_CHUNKS = [
    {
        "content": "def authenticate_user(username, password):\n    \"\"\"Authenticate a user against the database.\"\"\"\n    user = db.find_user(username)\n    if user and user.check_password(password):\n        return create_session(user)\n    raise AuthenticationError('Invalid credentials')",
        "type": "function",
        "name": "authenticate_user",
        "start_line": 1,
        "end_line": 6,
        "keywords": ["auth", "login", "password", "session", "database"],
        "summary": "Authenticates a user by checking username/password against the database and creating a session."
    },
    {
        "content": "class UserSession:\n    def __init__(self, user, token):\n        self.user = user\n        self.token = token\n        self.created_at = datetime.now()\n\n    def is_valid(self):\n        return (datetime.now() - self.created_at).seconds < 3600",
        "type": "class",
        "name": "UserSession",
        "start_line": 8,
        "end_line": 15,
        "keywords": ["session", "user", "token", "expiry", "validation"],
        "summary": "Represents a user session with a token and 1-hour expiry check."
    },
    {
        "content": "def logout_user(session_token):\n    \"\"\"Invalidate a user session.\"\"\"\n    sessions.pop(session_token, None)",
        "type": "function",
        "name": "logout_user",
        "start_line": 17,
        "end_line": 19,
        "keywords": ["logout", "session", "invalidate"],
        "summary": "Logs out a user by removing their session token."
    }
]

# 1. Create flow with pre-chunked semantic inputs
print("\n[1] Creating flow with semantic chunks...")
result = api.create_flow(AGENT_ID, FILE_PATH, chunks=SEMANTIC_CHUNKS)
print(f"    Result keys: {list(result.keys())}")

# 2. Verify vectors.json exists with entries
print("\n[2] Checking vectors.json...")
agent_dir = os.path.join(TEST_DIR, "agents", AGENT_ID)
vectors_path = os.path.join(agent_dir, "vectors.json")

assert os.path.exists(vectors_path), f"vectors.json NOT found at {vectors_path}"
with open(vectors_path, 'r', encoding='utf-8') as f:
    vectors_data = json.load(f)

print(f"    vectors.json has {len(vectors_data)} entries")
assert len(vectors_data) == 3, f"Expected 3 vectors, got {len(vectors_data)}"

for hash_id, vec in vectors_data.items():
    print(f"    - hash={hash_id[:12]}... dim={len(vec)}")
    assert len(vec) > 0, f"Vector for {hash_id} is empty!"
print("    ✅ vectors.json has 3 entries with vectors!")

# 3. Verify chunks in file_contexts.json do NOT have inline vectors
print("\n[3] Checking file_contexts.json (no inline vectors)...")
fc_path = os.path.join(agent_dir, "file_contexts.json")
with open(fc_path, 'r', encoding='utf-8') as f:
    contexts = json.load(f)

for ctx in contexts:
    for chunk in ctx.get("chunks", []):
        assert "vector" not in chunk, (
            f"Chunk '{chunk.get('name')}' still has inline 'vector' field!"
        )
        assert chunk.get("embedding_id"), (
            f"Chunk '{chunk.get('name')}' is missing 'embedding_id' reference!"
        )
        print(f"    - {chunk.get('name')}: embedding_id={chunk.get('embedding_id', '')[:12]}..., "
              f"has_vector_field=False ✅")

print("    ✅ No inline vectors in file_contexts.json!")

# 4. Verify global chunks.json has no inline vectors
print("\n[4] Checking global chunks.json (no inline vectors)...")
global_chunks_path = os.path.join(TEST_DIR, "chunks.json")
if os.path.exists(global_chunks_path):
    with open(global_chunks_path, 'r', encoding='utf-8') as f:
        global_chunks = json.load(f)
    for hash_id, chunk_data in global_chunks.items():
        assert "vector" not in chunk_data, (
            f"Global chunk {hash_id} still has inline 'vector' field!"
        )
        print(f"    - {chunk_data.get('name', '?')}: no inline vector ✅")
    print("    ✅ Global chunks.json has no inline vectors!")

# 5. Test hybrid search retrieval (uses vectors from vectors.json)
print("\n[5] Testing hybrid search (loads vectors from vectors.json)...")
search_result = api.get_flow(AGENT_ID, "authentication login session")
print(f"    Search status: {search_result.get('status')}")
results = search_result.get("results", [])
print(f"    Results count: {len(results)}")
for r in results:
    chunk = r.get("chunk", {})
    print(f"    - {chunk.get('name')}: score={r.get('score', 0):.4f} "
          f"(vec={r.get('vector_score', 0):.4f}, kw={r.get('keyword_score', 0):.4f})")

assert len(results) > 0, "Hybrid search returned no results!"
# Verify vector scores are nonzero (proves vectors.json was read)
any_vec_score = any(r.get("vector_score", 0) > 0 for r in results)
assert any_vec_score, "No results have vector scores — vectors.json may not be loaded!"
print("    ✅ Hybrid search works with vectors from vectors.json!")

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("  - Vectors stored in dedicated vectors.json (3 entries)")
print("  - Chunks in file_contexts.json have NO inline vectors")
print("  - Global chunks.json has NO inline vectors")
print("  - Hybrid search retrieves using vectors from vectors.json")
print("=" * 60)

# Cleanup
shutil.rmtree(TEST_DIR)
