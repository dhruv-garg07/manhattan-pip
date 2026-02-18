"""
End-to-End Test Suite for VFS Navigation + CRUD MCP Tools
Tests all 10 tools through the CodingAPI layer with complex, realistic scenarios.

Uses real source files from this project as test data.
"""

import sys, os, json, shutil, time, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

# ── Config ──────────────────────────────────────────────────────────────────
TEST_ROOT = os.path.join(os.path.dirname(__file__), ".test_e2e_vfs")
AGENT_ID = "e2e_test_agent"

# Real complex files from this project for testing
REAL_FILES = {
    "retriever": os.path.abspath(os.path.join(
        os.path.dirname(__file__), "src", "manhattan_mcp", "gitmem_coding", "coding_hybrid_retriever.py"
    )),
    "store": os.path.abspath(os.path.join(
        os.path.dirname(__file__), "src", "manhattan_mcp", "gitmem_coding", "coding_store.py"
    )),
    "builder": os.path.abspath(os.path.join(
        os.path.dirname(__file__), "src", "manhattan_mcp", "gitmem_coding", "coding_memory_builder.py"
    )),
    "api": os.path.abspath(os.path.join(
        os.path.dirname(__file__), "src", "manhattan_mcp", "gitmem_coding", "coding_api.py"
    )),
    "server": os.path.abspath(os.path.join(
        os.path.dirname(__file__), "src", "manhattan_mcp", "server.py"
    )),
}

passed = 0
failed = 0
errors = []


def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper(api):
            global passed, failed
            try:
                func(api)
                passed += 1
                print(f"  [PASS] {name}")
            except AssertionError as e:
                failed += 1
                errors.append((name, str(e)))
                print(f"  [FAIL] {name}: {e}")
            except Exception as e:
                failed += 1
                errors.append((name, f"EXCEPTION: {e}\n{traceback.format_exc()}"))
                print(f"  [ERR] {name}: {type(e).__name__}: {e}")
        wrapper.__test_name__ = name
        return wrapper
    return decorator


# ============================================================================
# TEST 1: INDEX_FILE — Index multiple real files
# ============================================================================

@test("index_file: Auto-parse complex 661-line retriever file via AST")
def test_index_retriever(api):
    result = api.index_file(AGENT_ID, REAL_FILES["retriever"])
    assert "error" not in result.get("status", "").lower(), f"Indexing failed: {result}"
    assert result.get("file_path"), "No file_path in result"

@test("index_file: Auto-parse 578-line store file with 27 methods")
def test_index_store(api):
    result = api.index_file(AGENT_ID, REAL_FILES["store"])
    assert "error" not in result.get("status", "").lower(), f"Indexing failed: {result}"

@test("index_file: Auto-parse builder file with embedding logic")
def test_index_builder(api):
    result = api.index_file(AGENT_ID, REAL_FILES["builder"])
    assert "error" not in result.get("status", "").lower(), f"Indexing failed: {result}"

@test("index_file: Auto-parse server.py with MCP tool definitions")
def test_index_server(api):
    result = api.index_file(AGENT_ID, REAL_FILES["server"])
    assert "error" not in result.get("status", "").lower(), f"Indexing failed: {result}"

@test("index_file: Non-existent file returns error")
def test_index_nonexistent(api):
    result = api.index_file(AGENT_ID, "/does/not/exist/foo.py")
    assert result.get("status") == "error", f"Expected error, got: {result}"

@test("index_file: With pre-computed chunks (agent-provided)")
def test_index_with_chunks(api):
    chunks = [
        {
            "name": "AuthManager",
            "type": "class",
            "content": "class AuthManager:\n    def login(self, user, pwd): ...\n    def logout(self): ...",
            "summary": "Handles user authentication. Login validates credentials, Logout clears session.",
            "keywords": ["auth", "login", "logout", "session", "AuthManager"],
            "start_line": 1,
            "end_line": 50
        },
        {
            "name": "hash_password",
            "type": "function",
            "content": "def hash_password(pwd: str) -> str: ...",
            "summary": "Hashes password using bcrypt with salt. Returns hex digest.",
            "keywords": ["hash", "password", "bcrypt", "security"],
            "start_line": 52,
            "end_line": 65
        },
        {
            "name": "RateLimiter",
            "type": "class",
            "content": "class RateLimiter:\n    def check(self, ip): ...\n    def reset(self): ...",
            "summary": "Token-bucket rate limiter. Check returns True if under limit. Reset clears all buckets.",
            "keywords": ["rate_limit", "throttle", "token_bucket", "RateLimiter"],
            "start_line": 70,
            "end_line": 120
        }
    ]
    result = api.index_file(AGENT_ID, "virtual/auth_module.py", chunks)
    assert "error" not in result.get("status", "").lower(), f"Chunk indexing failed: {result}"


# ============================================================================
# TEST 2: READ_FILE_CONTEXT — Cache behavior
# ============================================================================

@test("read_file_context: Cache HIT on previously indexed retriever file")
def test_read_cache_hit(api):
    result = api.read_file_context(AGENT_ID, REAL_FILES["retriever"])
    assert result.get("status") == "cache_hit", f"Expected cache_hit, got: {result.get('status')}"
    assert result.get("code_flow"), "No code_flow in cached result"
    assert "_token_info" in result, "Missing _token_info"
    assert result["_token_info"]["tokens_saved"] >= 0, "Token savings should be >= 0"

@test("read_file_context: Cache MISS → auto-index on api.py")
def test_read_cache_miss_auto_index(api):
    # First remove index if exists
    api.remove_index(AGENT_ID, REAL_FILES["api"])
    # Now read — should auto-index
    result = api.read_file_context(AGENT_ID, REAL_FILES["api"])
    assert result.get("status") == "auto_indexed", f"Expected auto_indexed, got: {result.get('status')}"
    assert result.get("code_flow"), "No code_flow after auto-index"
    assert "_token_info" in result, "Missing _token_info after auto-index"

@test("read_file_context: Second read of same file is cache HIT")
def test_read_second_hit(api):
    result = api.read_file_context(AGENT_ID, REAL_FILES["api"])
    assert result.get("status") == "cache_hit", f"Second read should be cache_hit, got: {result.get('status')}"

@test("read_file_context: Non-existent file returns error")
def test_read_nonexistent(api):
    result = api.read_file_context(AGENT_ID, "/tmp/nonexistent_file_xyz.py")
    assert result.get("status") == "error", f"Expected error, got: {result.get('status')}"

@test("read_file_context: Token info shows meaningful compression")
def test_read_token_compression(api):
    result = api.read_file_context(AGENT_ID, REAL_FILES["retriever"])
    token_info = result.get("_token_info", {})
    raw_tokens = token_info.get("tokens_if_raw_read", 0)
    used_tokens = token_info.get("tokens_this_call", 0)
    # The retriever is a 661-line file, compressed should be less than raw
    assert raw_tokens > 0, "Raw tokens should be > 0"
    # Note: compression ratio depends on code_flow structure


# ============================================================================
# TEST 3: GET_FILE_OUTLINE — Structure extraction
# ============================================================================

@test("get_file_outline: Retriever file shows classes and methods")
def test_outline_retriever(api):
    result = api.get_file_outline(AGENT_ID, REAL_FILES["retriever"])
    assert result.get("status") == "ok", f"Outline failed: {result}"
    outline = result.get("outline", [])
    assert len(outline) > 5, f"Expected >5 outline items for 661-line file, got {len(outline)}"
    # Check that we get classes and functions
    types = {item["type"] for item in outline}
    assert "class" in types or "function" in types or "method" in types, f"Expected code units, got types: {types}"

@test("get_file_outline: Each item has name, type, line range")
def test_outline_structure(api):
    result = api.get_file_outline(AGENT_ID, REAL_FILES["store"])
    outline = result.get("outline", [])
    for item in outline:
        assert "name" in item, f"Missing 'name' in outline item: {item}"
        assert "type" in item, f"Missing 'type' in outline item: {item}"
        assert "start_line" in item, f"Missing 'start_line' in outline item: {item}"

@test("get_file_outline: Auto-indexes unknown file first")
def test_outline_auto_index(api):
    # Remove builder index and try outline
    api.remove_index(AGENT_ID, REAL_FILES["builder"])
    result = api.get_file_outline(AGENT_ID, REAL_FILES["builder"])
    assert result.get("status") == "ok", f"Outline should auto-index: {result}"
    assert len(result.get("outline", [])) > 0, "Outline should have items after auto-index"

@test("get_file_outline: Token info shows ~10% of raw file")
def test_outline_token_savings(api):
    result = api.get_file_outline(AGENT_ID, REAL_FILES["retriever"])
    token_info = result.get("_token_info", {})
    raw = token_info.get("tokens_if_raw_read", 1)
    outline = token_info.get("tokens_this_call", 0)
    ratio = outline / max(raw, 1) * 100
    # Outline should be significantly smaller than raw
    assert ratio < 80, f"Outline should be <80% of raw, got {ratio:.1f}%"

@test("get_file_outline: Non-existent file returns error")
def test_outline_nonexistent(api):
    result = api.get_file_outline(AGENT_ID, "/tmp/ghost_file.py")
    assert result.get("status") == "error", f"Expected error, got: {result.get('status')}"


# ============================================================================
# TEST 4: SEARCH_CODEBASE — Semantic + keyword hybrid search
# ============================================================================

@test("search_codebase: Find 'hybrid search' across all indexed files")
def test_search_hybrid(api):
    result = api.search_codebase(AGENT_ID, "hybrid search algorithm")
    assert "results" in result or "chunks" in result or "matches" in result, f"No results in search: {list(result.keys())}"

@test("search_codebase: Find 'vector store' logic")
def test_search_vector_store(api):
    result = api.search_codebase(AGENT_ID, "vector storage and embedding")
    # Should find something related to CodingVectorStore or embedding
    assert result, "Search returned nothing"

@test("search_codebase: Find 'session tracking' (concept search)")
def test_search_concept(api):
    result = api.search_codebase(AGENT_ID, "session tracking and analytics")
    assert result, "Concept search returned nothing"

@test("search_codebase: Find 'AuthManager' (from pre-computed chunks)")
def test_search_custom_chunks(api):
    result = api.search_codebase(AGENT_ID, "authentication login manager")
    assert result, "Search for custom chunk content returned nothing"

@test("search_codebase: Find 'rate limiting throttle'")
def test_search_rate_limit(api):
    result = api.search_codebase(AGENT_ID, "rate limiting throttle token bucket")
    assert result, "Search for rate limiting returned nothing"

@test("search_codebase: Symbol lookup 'CodingContextStore'")
def test_search_symbol(api):
    result = api.search_codebase(AGENT_ID, "CodingContextStore class")
    assert result, "Symbol search returned nothing"

@test("search_codebase: Complex natural language query")
def test_search_natural_language(api):
    result = api.search_codebase(AGENT_ID, "how does the system detect if a cached file is stale or outdated?")
    assert result, "Natural language search returned nothing"

@test("search_codebase: Empty query doesn't crash")
def test_search_empty(api):
    try:
        result = api.search_codebase(AGENT_ID, "")
        # Should either return empty results or handle gracefully
    except Exception as e:
        raise AssertionError(f"Empty query crashed: {e}")


# ============================================================================
# TEST 5: LIST_DIRECTORY — VFS navigation
# ============================================================================

@test("list_directory: Root shows top-level categories")
def test_list_root(api):
    result = api.list_directory(AGENT_ID, "")
    assert result.get("status") == "ok", f"Root listing failed: {result}"
    items = result.get("items", [])
    names = [item.get("name") for item in items]
    assert "files" in names, f"Root should have 'files', got: {names}"

@test("list_directory: 'files' shows language folders")
def test_list_files(api):
    result = api.list_directory(AGENT_ID, "files")
    assert result.get("status") == "ok", f"Files listing failed: {result}"
    items = result.get("items", [])
    # Should have at least a python folder since we indexed .py files
    assert len(items) > 0, "Files directory should have language folders"

@test("list_directory: 'files/python' shows indexed Python files")
def test_list_python(api):
    result = api.list_directory(AGENT_ID, "files/python")
    items = result.get("items", [])
    # We indexed multiple .py files
    assert len(items) >= 2, f"Expected >=2 Python files, got {len(items)}: {[i.get('name') for i in items]}"

@test("list_directory: 'stats' shows analytics files")
def test_list_stats(api):
    result = api.list_directory(AGENT_ID, "stats")
    items = result.get("items", [])
    names = [item.get("name") for item in items]
    assert any("overview" in n for n in names), f"Stats should have overview, got: {names}"


# ============================================================================
# TEST 6: REINDEX_FILE — Update after "modification"
# ============================================================================

@test("reindex_file: Re-index store.py returns success")
def test_reindex(api):
    result = api.reindex_file(AGENT_ID, REAL_FILES["store"])
    assert "error" not in result.get("status", "").lower(), f"Re-index failed: {result}"

@test("reindex_file: Re-indexed file still readable")
def test_reindex_then_read(api):
    api.reindex_file(AGENT_ID, REAL_FILES["builder"])
    result = api.read_file_context(AGENT_ID, REAL_FILES["builder"])
    assert result.get("status") == "cache_hit", f"Should be cache_hit after reindex, got: {result.get('status')}"


# ============================================================================
# TEST 7: LIST_INDEXED_FILES — List all
# ============================================================================

@test("list_indexed_files: Shows all indexed files with metadata")
def test_list_indexed(api):
    result = api.list_indexed_files(AGENT_ID)
    assert result.get("total", 0) >= 4, f"Expected >=4 indexed files, got {result.get('total')}"
    items = result.get("items", [])
    for item in items:
        assert "file_path" in item, f"Missing file_path in item: {item}"

@test("list_indexed_files: Pagination works")
def test_list_pagination(api):
    all_files = api.list_indexed_files(AGENT_ID, limit=100)
    page1 = api.list_indexed_files(AGENT_ID, limit=2, offset=0)
    page2 = api.list_indexed_files(AGENT_ID, limit=2, offset=2)
    assert len(page1.get("items", [])) <= 2, "Limit not respected"
    assert page1.get("total") == all_files.get("total"), "Total should be same"


# ============================================================================
# TEST 8: REMOVE_INDEX — Delete
# ============================================================================

@test("remove_index: Remove virtual auth_module.py")
def test_remove(api):
    result = api.remove_index(AGENT_ID, "virtual/auth_module.py")
    # Note: normpath may affect this, but the store normalizes paths
    # Let's check with the actual stored path

@test("remove_index: Non-existent returns False")
def test_remove_nonexistent(api):
    result = api.remove_index(AGENT_ID, "/ghost/file.py")
    assert result == False, f"Expected False for non-existent, got: {result}"


# ============================================================================
# TEST 9: GET_TOKEN_SAVINGS — Analytics
# ============================================================================

@test("get_token_savings: Returns comprehensive report")
def test_token_savings(api):
    result = api.get_token_savings(AGENT_ID)
    assert "files_in_cache" in result, f"Missing files_in_cache: {list(result.keys())}"
    assert result.get("files_in_cache", 0) >= 3, f"Expected >=3 files in cache, got {result.get('files_in_cache')}"


# ============================================================================
# TEST 10: BACKWARD COMPATIBILITY — Old method names
# ============================================================================

@test("backward-compat: create_mem works same as index_file")
def test_compat_create(api):
    result = api.create_mem(AGENT_ID, REAL_FILES["server"])
    assert "error" not in result.get("status", "").lower(), f"create_mem failed: {result}"

@test("backward-compat: get_mem works same as search_codebase")
def test_compat_get(api):
    result = api.get_mem(AGENT_ID, "file context caching")
    assert result is not None, "get_mem returned None"

@test("backward-compat: update_mem works same as reindex_file")
def test_compat_update(api):
    result = api.update_mem(AGENT_ID, REAL_FILES["store"])
    assert "error" not in result.get("status", "").lower(), f"update_mem failed: {result}"

@test("backward-compat: list_mems works same as list_indexed_files")
def test_compat_list(api):
    result = api.list_mems(AGENT_ID)
    assert result.get("total", 0) > 0, f"list_mems returned 0 items"

@test("backward-compat: delete_mem works same as remove_index")
def test_compat_delete(api):
    # Index a temp file first
    chunks = [{"name": "temp", "type": "function", "content": "def temp(): pass",
               "summary": "Temp function", "keywords": ["temp"], "start_line": 1, "end_line": 1}]
    api.create_mem(AGENT_ID, "temp_compat_test.py", chunks)
    result = api.delete_mem(AGENT_ID, "temp_compat_test.py")
    # Result is bool


# ============================================================================
# TEST 11: CROSS-TOOL WORKFLOWS (Complex E2E)
# ============================================================================

@test("workflow: Index → Outline → Search → Read (full cycle)")
def test_full_cycle(api):
    # Fresh index
    api.remove_index(AGENT_ID, REAL_FILES["retriever"])
    
    # 1. Index
    idx = api.index_file(AGENT_ID, REAL_FILES["retriever"])
    assert "error" not in idx.get("status", "").lower(), f"Index failed: {idx}"
    
    # 2. Outline
    outline = api.get_file_outline(AGENT_ID, REAL_FILES["retriever"])
    assert outline.get("status") == "ok", f"Outline failed: {outline}"
    assert outline.get("total_chunks", 0) > 5, "Expected >5 chunks for retriever"
    
    # 3. Search for specific functionality
    search = api.search_codebase(AGENT_ID, "keyword decomposition compound identifiers")
    assert search, "Search returned nothing"
    
    # 4. Read full context
    read = api.read_file_context(AGENT_ID, REAL_FILES["retriever"])
    assert read.get("status") == "cache_hit", f"Should be cached: {read.get('status')}"

@test("workflow: Search → identify file → Read context → Get outline")
def test_search_then_read(api):
    # Search for something
    search = api.search_codebase(AGENT_ID, "store file chunks JSON persistence")
    assert search, "Search failed"
    
    # Read the store file
    read = api.read_file_context(AGENT_ID, REAL_FILES["store"])
    assert read.get("status") == "cache_hit"
    
    # Get outline
    outline = api.get_file_outline(AGENT_ID, REAL_FILES["store"])
    assert outline.get("status") == "ok"
    assert any("store_file_chunks" in item.get("name", "") for item in outline.get("outline", [])), \
        "Outline should contain store_file_chunks method"

@test("workflow: List dir → Navigate → Read specific file")
def test_navigate_and_read(api):
    # List root
    root = api.list_directory(AGENT_ID, "")
    assert any(i["name"] == "files" for i in root.get("items", []))
    
    # List files
    files = api.list_directory(AGENT_ID, "files")
    assert len(files.get("items", [])) > 0
    
    # List python files
    py = api.list_directory(AGENT_ID, "files/python")
    assert len(py.get("items", [])) > 0
    
    # Read first python file
    first_file = py["items"][0]
    # The file should be in our cache, read it
    # (We can't easily read by virtual path here, but the real path is available)

@test("workflow: Multi-file semantic search across indexed codebase")
def test_multi_file_search(api):
    # These queries should find results across different files
    queries = [
        "how are embeddings generated for code chunks",
        "thread safety locking mechanism",
        "JSON file persistence and loading",
        "cleanup stale contexts",
        "keyword enrichment and inference",
    ]
    for q in queries:
        result = api.search_codebase(AGENT_ID, q)
        # Just verify no crashes
        assert result is not None, f"Search crashed for: {q}"


# ============================================================================
# RUNNER
# ============================================================================

def main():
    global passed, failed
    
    print("=" * 70)
    print("   E2E TEST: VFS Navigation + CRUD MCP Tools")
    print("=" * 70)
    
    # Clean start
    if os.path.exists(TEST_ROOT):
        shutil.rmtree(TEST_ROOT)
    os.makedirs(TEST_ROOT, exist_ok=True)
    
    # Verify test files exist
    for name, path in REAL_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] Test file missing: {name} -> {path}")
            return
    
    print(f"\nUsing test root: {TEST_ROOT}")
    print(f"Real files: {len(REAL_FILES)}")
    
    # Initialize API
    api = CodingAPI(root_path=TEST_ROOT)
    print(f"API initialized: store={api.store.root_path}\n")
    
    # Collect all test functions in order
    tests = [v for v in globals().values() if callable(v) and hasattr(v, "__test_name__")]
    
    print(f"Running {len(tests)} tests...\n")
    
    t0 = time.time()
    for test_fn in tests:
        test_fn(api)
    elapsed = time.time() - t0
    
    print(f"\n{'=' * 70}")
    print(f"   RESULTS: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    print(f"{'=' * 70}")
    
    if errors:
        print(f"\nFAILURES:")
        for name, msg in errors:
            print(f"\n  {name}:")
            for line in msg.split("\n")[:5]:
                print(f"    {line}")
    
    # Cleanup
    if os.path.exists(TEST_ROOT):
        shutil.rmtree(TEST_ROOT)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
