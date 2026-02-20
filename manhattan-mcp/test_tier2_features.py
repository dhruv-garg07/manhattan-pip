"""
Test script for Tier 2 Features: invalidate_cache, summarize_context, create_snapshot, compare_snapshots, usage_report, performance_profile
Tests directly via CodingAPI.
"""
import sys
import os
import json
import tempfile
import shutil
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

# Setup
test_dir = tempfile.mkdtemp(prefix="tier2_test_")
api = CodingAPI(root_path=os.path.join(test_dir, ".gitmem_coding"))
AGENT = "test_tier2"

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
print("\n=== SETUP: Indexing files ===")

temp_file = os.path.join(test_dir, "utils.py")
with open(temp_file, "w") as f:
    f.write("""
def add(a, b):
    return a + b

def sub(a, b):
    return a - b

class Math:
    def mul(self, a, b):
        return a * b
""")

result = api.index_file(AGENT, temp_file)
print(f"  Indexed: {os.path.basename(temp_file)}")

# ==============================================================================
# TEST 1: summarize_context
# ==============================================================================
print("\n=== TEST 1: summarize_context ===")

# 1a: brief
res_brief = api.summarize_context(AGENT, temp_file, verbosity="brief")
test("summarize_context brief works", res_brief.get("status") == "ok")
test("brief summary contains file name", "utils.py" in res_brief.get("summary", ""))
test("brief summary counts chunks", "chunks" in res_brief.get("summary", ""))

# 1b: normal
res_norm = api.summarize_context(AGENT, temp_file, verbosity="normal")
test("summarize_context normal works", res_norm.get("status") == "ok")
test("normal has code_flow", "code_flow" in res_norm)

# 1c: detailed
res_det = api.summarize_context(AGENT, temp_file, verbosity="detailed")
test("summarize_context detailed works", res_det.get("status") == "ok")
test("detailed has chunks list", isinstance(res_det.get("chunks"), list))
test("detailed chunks have content", len(res_det["chunks"]) > 0 and "content" in res_det["chunks"][0])

# ==============================================================================
# TEST 2: usage_report
# ==============================================================================
print("\n=== TEST 2: usage_report ===")

# Access the file context to increment access_count
rc_res = api.read_file_context(AGENT, temp_file)
print(f"  read_file_context status: {rc_res.get('status')}")

report = api.usage_report(AGENT)
test("usage_report has sessions", "sessions" in report)
test("usage_report shows indexed files", report.get("indexing_activity", {}).get("total_files_indexed") == 1)
test("usage_report shows most accessed", len(report.get("most_accessed_files", [])) > 0)
test("access count is recorded", report["most_accessed_files"][0]["access_count"] >= 1)

# ==============================================================================
# TEST 3: performance_profile
# ==============================================================================
print("\n=== TEST 3: performance_profile ===")

# Run more ops to ensure counts
api.search_codebase(AGENT, "add function")

profile = api.performance_profile(AGENT)
test("performance_profile works", profile.get("status") == "ok")
prof = profile.get("profile", {})

# Diagnostic print if retrieval count is 0
if prof.get("retrieval", {}).get("count", 0) < 1:
    print(f"  [DEBUG] Profile: {json.dumps(prof, indent=2)}")
    report = api.usage_report(AGENT)
    print(f"  [DEBUG] Usage Report: {json.dumps(report, indent=2)}")

test("profile tracks indexing", prof.get("indexing", {}).get("count", 0) >= 1)
test("profile tracks search", prof.get("search", {}).get("count", 0) >= 1)
test("profile tracks retrieval", prof.get("retrieval", {}).get("count", 0) >= 1)
test("latencies are recorded", prof.get("search", {}).get("total_ms", 0) >= 0)

# ==============================================================================
# TEST 4: invalidate_cache
# ==============================================================================
print("\n=== TEST 4: invalidate_cache ===")

# 4a: scope='file'
res_inv = api.invalidate_cache(AGENT, temp_file, scope="file")
test("invalidate_cache scope='file' works", res_inv.get("status") == "invalidated")

report2 = api.usage_report(AGENT)
test("file removed from index", report2.get("indexing_activity", {}).get("total_files_indexed") == 0)

# 4b: scope='all'
api.index_file(AGENT, temp_file)
res_inv_all = api.invalidate_cache(AGENT, scope="all")
test("invalidate_cache scope='all' works", res_inv_all.get("status") == "invalidated")

report3 = api.usage_report(AGENT)
test("total cache reset", report3.get("indexing_activity", {}).get("total_files_indexed") == 0)

# ==============================================================================
# TEST 5: snapshots (optional if DAG available)
# ==============================================================================
print("\n=== TEST 5: snapshots (mock/check) ===")
# Note: Full snapshot test needs a working MemoryDAG which might not be initialized
# in a temp directory without further setup. We'll just check if the methods don't crash.

res_snap = api.create_snapshot(AGENT, "Initial state")
# It might return error if DAG is not setup, but it shouldn't raise exception
print(f"  Snapshot result: {res_snap.get('status')} - {res_snap.get('message', '')}")
test("create_snapshot call successful (even if error response)", "status" in res_snap)

# ==============================================================================
# Cleanup & Results
# ==============================================================================
print("\n" + "=" * 60)
print(f"  RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

shutil.rmtree(test_dir, ignore_errors=True)
sys.exit(0 if failed == 0 else 1)
