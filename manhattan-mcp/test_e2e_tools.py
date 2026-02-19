"""
Rigorous End-to-End Test Suite for Manhattan MCP tools.
Tests all tools with complex, multi-file indexing and difficult conceptual/technical queries.
"""

import sys
import os
import json
import shutil
import asyncio
import traceback

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

# Test configuration
TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".test_e2e_data"))
AGENT_ID = "rigorous_e2e_agent"

# Complex files to index
FILES_TO_INDEX = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "manhattan_mcp", "server.py")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "manhattan_mcp", "gitmem_coding", "coding_hybrid_retriever.py")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "manhattan_mcp", "gitmem_coding", "ast_skeleton.py")),
]

# Difficult queries
DIFFICULT_QUERIES = [
    {
        "name": "Conceptual: Token vs Accuracy",
        "query": "How does the system balance token savings with search accuracy?"
    },
    {
        "name": "Technical: Hybrid Scoring",
        "query": "Explain the scoring mechanism used in hybrid search including weights between vector and keyword scores."
    },
    {
        "name": "Implementation: Path Resolution",
        "query": "Where is the platform-specific logic for data path resolution for Windows, macOS, and Linux?"
    },
    {
        "name": "Constraint: Agent Instructions",
        "query": "What are the specific instructions or mandatory tool usage rules given to AI agents?"
    }
]

stats = {"passed": 0, "failed": 0}

def print_step(msg):
    print(f"\n>>> [PROCESS] {msg}")

def print_result(name, success, error=None):
    if success:
        stats["passed"] += 1
        print(f"  [PASS] {name}")
    else:
        stats["failed"] += 1
        print(f"  [FAIL] {name}")
        if error:
            print(f"         {error}")

async def run_rigorous_tests():
    print("=" * 70)
    print("   RIGOROUS E2E TEST: MULTI-FILE INDEXING & DIFFICULT QUERIES")
    print("=" * 70)

    # 1. SETUP
    print_step("Initializing API and cleaning test directory")
    if os.path.exists(TEST_ROOT):
        shutil.rmtree(TEST_ROOT)
    os.makedirs(TEST_ROOT, exist_ok=True)
    
    api = CodingAPI(root_path=TEST_ROOT)
    print(f"API initialized at: {TEST_ROOT}")

    # 2. SEED DATA (Multiple Files)
    print_step("Indexing multiple complex source files for realistic search")
    for fpath in FILES_TO_INDEX:
        try:
            print(f"  Indexing {os.path.basename(fpath)}...")
            result = api.index_file(AGENT_ID, fpath)
            success = "error" not in result.get("status", "").lower()
            if not success:
                print(f"  [WARN] Failed to index {fpath}: {result}")
        except Exception as e:
            print(f"  [ERR] Exception indexing {fpath}: {e}")

    # 3. TEST SEARCH WITH DIFFICULT QUERIES
    print_step("Testing SEARCH with difficult queries")
    for q_data in DIFFICULT_QUERIES:
        name = q_data["name"]
        query = q_data["query"]
        print(f"\n  Query: '{query}'")
        try:
            result = api.search_codebase(AGENT_ID, query, top_k=3)
            # Detailed reporting of results
            results = result.get("results", []) or result.get("chunks", []) or []
            print(f"  Found {len(results)} matches.")
            for i, res in enumerate(results[:2]): # Show top 2
                file_name = os.path.basename(res.get("file_path", "unknown"))
                score = res.get("score", 0)
                match_type = res.get("match_type", "unknown")
                print(f"    Match {i+1}: {file_name} (Score: {score:.3f}, Type: {match_type})")
            
            success = len(results) > 0
            print_result(f"Search: {name}", success)
        except Exception as e:
            print_result(f"Search: {name}", False, traceback.format_exc())

    # 4. TEST OTHER TOOLS
    print_step("Verifying other tools lifecycle")
    
    # get_file_outline
    try:
        fpath = FILES_TO_INDEX[0] # server.py
        result = api.get_file_outline(AGENT_ID, fpath)
        success = result.get("status") == "ok"
        print_result("get_file_outline", success)
    except Exception as e:
        print_result("get_file_outline", False, traceback.format_exc())

    # read_file_context
    try:
        fpath = FILES_TO_INDEX[1] # retriever
        result = api.read_file_context(AGENT_ID, fpath)
        success = result.get("status") == "cache_hit"
        print_result("read_file_context (cache hit)", success)
    except Exception as e:
        print_result("read_file_context", False, traceback.format_exc())

    # list_directory
    try:
        result = api.list_directory(AGENT_ID, "files/python")
        success = len(result.get("items", [])) >= 3 # Should have all 3 files
        print_result("list_directory (VFS check)", success)
    except Exception as e:
        print_result("list_directory", False, traceback.format_exc())

    # get_token_savings
    try:
        result = api.get_token_savings(AGENT_ID)
        saved = result.get("total_tokens_saved", 0)
        print(f"  Cumulative tokens saved: {saved}")
        print_result("get_token_savings", True)
    except Exception as e:
        print_result("get_token_savings", False, traceback.format_exc())

    # remove_index
    try:
        fpath = FILES_TO_INDEX[2] # ast_skeleton
        api.remove_index(AGENT_ID, fpath)
        # Check if gone from list_indexed_files
        list_res = api.list_indexed_files(AGENT_ID)
        found = any(fpath in item["file_path"] for item in list_res.get("items", []))
        print_result("remove_index", not found)
    except Exception as e:
        print_result("remove_index", False, traceback.format_exc())

    print(f"\n{'=' * 70}")
    print(f"   FINAL RESULTS: {stats['passed']} passed, {stats['failed']} failed")
    print(f"{'=' * 70}\n")

    # Cleanup
    if os.path.exists(TEST_ROOT):
        shutil.rmtree(TEST_ROOT)

if __name__ == "__main__":
    asyncio.run(run_rigorous_tests())
