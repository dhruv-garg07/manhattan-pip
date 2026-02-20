"""
Test script for Tier 1 Features: cross_reference, dependency_graph, delta_update, cache_stats
Tests directly via CodingAPI (not MCP transport).
"""
import sys
import os
import json
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

# Setup
test_dir = tempfile.mkdtemp(prefix="tier1_test_")
api = CodingAPI(root_path=os.path.join(test_dir, ".gitmem_coding"))
AGENT = "test_tier1"

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name} - {detail}")
        failed += 1

# ==============================================================================
# Setup: Index real project files
# ==============================================================================
print("\n=== SETUP: Indexing project files ===")

real_files = [
    os.path.join(os.path.dirname(__file__), "src", "manhattan_mcp", "gitmem_coding", "coding_api.py"),
    os.path.join(os.path.dirname(__file__), "src", "manhattan_mcp", "gitmem_coding", "coding_store.py"),
    os.path.join(os.path.dirname(__file__), "src", "manhattan_mcp", "server.py"),
    os.path.join(os.path.dirname(__file__), "src", "manhattan_mcp", "gitmem_coding", "coding_memory_builder.py"),
    os.path.join(os.path.dirname(__file__), "src", "manhattan_mcp", "gitmem_coding", "coding_hybrid_retriever.py"),
]

for f in real_files:
    result = api.index_file(AGENT, f)
    chunks = result.get("message", "")
    print(f"  Indexed: {os.path.basename(f)} - {chunks}")

# ==============================================================================
# TEST 1: cross_reference
# ==============================================================================
print("\n=== TEST 1: cross_reference ===")

# 1a: Search for a class that exists in multiple files
result = api.cross_reference(AGENT, "CodingContextStore")
test("cross_reference finds CodingContextStore",
     result["total_references"] > 0,
     f"Got {result['total_references']} refs")
test("cross_reference returns file paths",
     result.get("files_matched", 0) >= 1,
     f"Matched {result.get('files_matched', 0)} files")
test("cross_reference has _token_info",
     "_token_info" in result)

# 1b: Search for a function name
result2 = api.cross_reference(AGENT, "retrieve_file_context")
test("cross_reference finds retrieve_file_context",
     result2["total_references"] > 0,
     f"Got {result2['total_references']} refs")

# 1c: Check reference details
if result2["total_references"] > 0:
    ref = result2["references"][0]
    test("reference has chunk_name", "chunk_name" in ref)
    test("reference has chunk_type", "chunk_type" in ref)
    test("reference has start_line", "start_line" in ref)
    test("reference has match_reason", "match_reason" in ref)
else:
    for _ in range(4):
        test("reference detail", False, "No references returned")

# 1d: Search for non-existent symbol
result3 = api.cross_reference(AGENT, "ThisDoesNotExistAnywhere12345")
test("cross_reference returns 0 for non-existent symbol",
     result3["total_references"] == 0)

# ==============================================================================
# TEST 2: dependency_graph
# ==============================================================================
print("\n=== TEST 2: dependency_graph ===")

# 2a: Get dependency graph for coding_api.py
api_path = real_files[0]  # coding_api.py
result = api.dependency_graph(AGENT, api_path)
test("dependency_graph returns ok status",
     result.get("status") == "ok",
     f"Got status: {result.get('status')}")
test("dependency_graph has imports list",
     isinstance(result.get("imports"), list))
test("dependency_graph has imported_by list",
     isinstance(result.get("imported_by"), list))
test("dependency_graph has calls_to list",
     isinstance(result.get("calls_to"), list))
test("dependency_graph has graph_summary",
     "graph_summary" in result)
test("dependency_graph has _token_info",
     "_token_info" in result)

# 2b: coding_api.py should import from coding_store etc.
test("coding_api imports modules",
     len(result.get("imports", [])) > 0,
     f"Imports: {result.get('imports', [])}")

# 2c: coding_api.py should detect self.store.xxx calls
test("coding_api has cross-file calls",
     len(result.get("calls_to", [])) > 0,
     f"Calls: {json.dumps(result.get('calls_to', [])[:3])}")

# 2d: Test depth=2 transitive imports
result_deep = api.dependency_graph(AGENT, api_path, depth=2)
test("depth=2 returns ok",
     result_deep.get("status") == "ok")

# ==============================================================================
# TEST 3: delta_update
# ==============================================================================
print("\n=== TEST 3: delta_update ===")

# 3a: Create a temp file, index it, then delta_update without changes (all unchanged)
temp_file = os.path.join(test_dir, "sample.py")
with open(temp_file, "w") as f:
    f.write("""
def greet(name):
    \"\"\"Say hello.\"\"\"
    return f"Hello, {name}!"

def farewell(name):
    \"\"\"Say goodbye.\"\"\"
    return f"Goodbye, {name}!"

class Calculator:
    def add(self, a, b):
        return a + b
    def subtract(self, a, b):
        return a - b
""")

# Index first
api.index_file(AGENT, temp_file)

# Delta update without changes
result = api.delta_update(AGENT, temp_file)
test("delta_update returns delta_applied",
     result.get("status") == "delta_applied",
     f"Got status: {result.get('status')}")
test("delta_update shows chunks_unchanged > 0",
     result.get("chunks_unchanged", 0) > 0,
     f"Unchanged: {result.get('chunks_unchanged', 0)}")
test("delta_update shows chunks_added == 0 (no changes)",
     result.get("chunks_added", 0) == 0,
     f"Added: {result.get('chunks_added', 0)}")
test("delta_update has _token_info",
     "_token_info" in result)

# 3b: Modify the file and delta_update (should detect changes)
with open(temp_file, "w") as f:
    f.write("""
def greet(name):
    \"\"\"Say hello.\"\"\"
    return f"Hello, {name}!"

def farewell(name):
    \"\"\"Say goodbye.\"\"\"
    return f"Goodbye, {name}!"

def new_function():
    \"\"\"Brand new function.\"\"\"
    return 42

class Calculator:
    def add(self, a, b):
        return a + b
    def subtract(self, a, b):
        return a - b
    def multiply(self, a, b):
        return a * b
""")

result2 = api.delta_update(AGENT, temp_file)
test("delta_update after modification returns delta_applied",
     result2.get("status") == "delta_applied")
test("delta_update detects added chunks",
     result2.get("chunks_added", 0) > 0,
     f"Added: {result2.get('chunks_added', 0)}")
test("delta_update shows total_chunks",
     result2.get("total_chunks", 0) > 0,
     f"Total: {result2.get('total_chunks', 0)}")

# 3c: Delta update on non-existent file
result3 = api.delta_update(AGENT, "/nonexistent/file.py")
test("delta_update returns error for missing file",
     result3.get("status") == "error")

# ==============================================================================
# TEST 4: cache_stats
# ==============================================================================
print("\n=== TEST 4: cache_stats ===")

result = api.cache_stats(AGENT)

test("cache_stats has overview",
     "overview" in result)
test("cache_stats overview has total_files",
     result.get("overview", {}).get("total_files", 0) > 0,
     f"Files: {result.get('overview', {}).get('total_files', 0)}")
test("cache_stats overview has total_chunks",
     result.get("overview", {}).get("total_chunks", 0) > 0,
     f"Chunks: {result.get('overview', {}).get('total_chunks', 0)}")
test("cache_stats overview has total_tokens_cached",
     result.get("overview", {}).get("total_tokens_cached", 0) >= 0)
test("cache_stats has freshness breakdown",
     "freshness" in result)
test("cache_stats freshness has fresh/stale/missing keys",
     all(k in result.get("freshness", {}) for k in ["fresh", "stale", "missing"]))
test("cache_stats has per_file list",
     isinstance(result.get("per_file"), list))
test("cache_stats per_file has entries",
     len(result.get("per_file", [])) > 0)

# Check per-file entry structure
if result.get("per_file"):
    entry = result["per_file"][0]
    test("per_file entry has file name", "file" in entry)
    test("per_file entry has chunks count", "chunks" in entry)
    test("per_file entry has language", "language" in entry)
    test("per_file entry has freshness", "freshness" in entry)
    test("per_file entry has access_count", "access_count" in entry)

test("cache_stats has recommendations",
     isinstance(result.get("recommendations"), list))

# ==============================================================================
# Cleanup & Results
# ==============================================================================
print("\n" + "=" * 60)
print(f"  RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

# Cleanup
shutil.rmtree(test_dir, ignore_errors=True)

sys.exit(0 if failed == 0 else 1)
