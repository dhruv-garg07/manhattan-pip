"""
Complex Memory Test for Coding Agents

Simulates a real coding agent interacting with the memory system.
Tests multi-file scenarios, updates, deduplication, and persistence.
"""
import sys
import os
import shutil
import json
import time

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

# Setup
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_complex_memory")
AGENT_ID = "coding_agent_xv1"

def setup():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    print(f"Initialized test directory: {TEST_DIR}")
    return CodingAPI(root_path=TEST_DIR)

def print_step(title):
    print(f"\n{'='*60}")
    print(f"TEST STEP: {title}")
    print(f"{'='*60}")

def verify_vectors_count(api, agent_id, expected_count, msg=""):
    vectors = api.vector_store._load_vectors(agent_id)
    actual = len(vectors)
    print(f"    [Check] Vector count: {actual} (Expected: {expected_count}) {msg}")
    assert actual == expected_count, f"Vector count mismatch! Got {actual}, want {expected_count}"

# =============================================================================
# SCENARIO 1: Multi-File Ingestion & Shared Logic
# =============================================================================
def test_scenario_1_ingestion(api):
    print_step("Scenario 1: Multi-File Ingestion & Deduplication")
    
    # File A: Utility library
    file_a = "/src/utils.py"
    chunks_a = [
        {
            "content": "def shared_logging(msg):\n    print(f'[LOG] {msg}')",
            "type": "function", 
            "name": "shared_logging",
            "start_line": 1, "end_line": 2,
            "keywords": ["log", "print", "debug"],
            "summary": "Standard logging function used across modules."
        },
        {
            "content": "class ConfigLoader:\n    def load(self):\n        return {}",
            "type": "class",
            "name": "ConfigLoader",
            "start_line": 4, "end_line": 6
        }
    ]
    
    # File B: Service using utils
    file_b = "/src/service.py"
    chunks_b = [
        {
            "content": "def process_data(data):\n    shared_logging('Processing...')\n    return data * 2",
            "type": "function",
            "name": "process_data",
            "start_line": 1, "end_line": 3,
            "keywords": ["process", "data", "transform"]
        },
        # Intentionally identical chunk to File A (simulating copy-paste or shared code ref)
        # Note: In reality, shared code is imported, but let's test content deduplication
        {
            "content": "def shared_logging(msg):\n    print(f'[LOG] {msg}')",
            "type": "function",
            "name": "shared_logging_copy", # Different name, same content
            "start_line": 10, "end_line": 11,
            "keywords": ["log", "print", "debug"],
             "summary": "Standard logging function used across modules."
        }
    ]

    print("  1. Ingesting File A...")
    api.create_flow(AGENT_ID, file_a, chunks_a)
    
    # Vector count should be 2
    verify_vectors_count(api, AGENT_ID, 2, "After File A")

    print("  2. Ingesting File B (with 1 duplicate content chunk)...")
    api.create_flow(AGENT_ID, file_b, chunks_b)
    
    # Vector count should be 3 (2 from A + 1 unique from B). The duplicate shared_logging should reuse vector.
    # Wait, hash is based on content. If content is identical, hash is identical.
    # chunk[1] of B has same content as chunk[0] of A.
    verify_vectors_count(api, AGENT_ID, 3, "After File B (Deduplication Check)")
    print("    ✅ Deduplication successful! Reused existing vector for identical content.")


# =============================================================================
# SCENARIO 2: Update & Versioning
# =============================================================================
def test_scenario_2_updates(api):
    print_step("Scenario 2: Content Updates & Vector Refresh")
    
    file_a = "/src/utils.py"
    
    # Update ConfigLoader in File A
    new_chunks_a = [
        {
            "content": "def shared_logging(msg):\n    print(f'[LOG] {msg}')", # Same as before
            "type": "function", 
            "name": "shared_logging",
            "start_line": 1, "end_line": 2,
            "keywords": ["log", "print", "debug"],
            "summary": "Standard logging function used across modules."
        },
        {
            "content": "class ConfigLoader:\n    def load(self):\n        # Updated implementation\n        return {'env': 'prod'}", # CHANGED CONTENT
            "type": "class",
            "name": "ConfigLoader",
            "start_line": 4, "end_line": 7
        }
    ]
    
    print("  1. Updating File A with changed ConfigLoader...")
    api.update_flow(AGENT_ID, file_a, new_chunks_a)
    
    # Vectors:
    # - shared_logging (hash X) - Reused (exists)
    # - ConfigLoader old (hash Y) - Still in vector store (we don't GC yet)
    # - ConfigLoader new (hash Z) - New added
    # Total expected: 3 (initial) + 1 (new) = 4
    verify_vectors_count(api, AGENT_ID, 4, "After Update (Old vector remains, New added)")
    print("    ✅ Vector store grew by 1 (new version added, old preserved).")


# =============================================================================
# SCENARIO 3: Hybrid Search
# =============================================================================
def test_scenario_3_search(api):
    print_step("Scenario 3: Hybrid Search Retrieval")
    
    # Query: "logging"
    # Should match 'shared_logging' strongly via keyword and vector
    print("  1. Searching for 'logging'...")
    res = api.get_flow(AGENT_ID, "logging mechanism")
    results = res.get("results", [])
    
    print(f"    Found {len(results)} matches.")
    assert len(results) >= 2, "Should find at least duplicate logging functions"
    
    top = results[0]
    print(f"    Top result: {top['chunk']['name']} (Score: {top['score']:.4f})")
    assert "logging" in top['chunk']['name'], "Top result should be logging related"
    assert top['vector_score'] > 0, "Vector score shoud be present"
    assert "content" not in top['chunk'], "Full content should NOT be returned, only summary!"
    print("    ✅ Hybrid search returned relevant results with vector scores.")


# =============================================================================
# SCENARIO 4: Persistence
# =============================================================================
def test_scenario_4_persistence():
    print_step("Scenario 4: Persistence & Reload")
    
    print("  1. Re-instantiating API (Simulating restart)...")
    new_api = CodingAPI(root_path=TEST_DIR)
    
    # Check vector count - should still be 4
    verify_vectors_count(new_api, AGENT_ID, 4, "After Reload")
    
    # Check if we can still search
    res = new_api.get_flow(AGENT_ID, "process data")
    results = res.get("results", [])
    assert len(results) > 0, "Search failed after reload"
    assert results[0]['chunk']['name'] == "process_data", "Incorrect top result after reload"
    print("    ✅ Persistence verified! Data intact after restart.")

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    try:
        api = setup()
        
        test_scenario_1_ingestion(api)
        test_scenario_2_updates(api)
        test_scenario_3_search(api)
        
        # Test 4 requires fresh API instance
        test_scenario_4_persistence()
        
        print("\n" + "="*60)
        print("ALL COMPLEX SCENARIOS PASSED!")
        print("="*60)
        
        # Cleanup
        shutil.rmtree(TEST_DIR)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
