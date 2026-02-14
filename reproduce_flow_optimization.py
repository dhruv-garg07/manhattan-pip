
import os
import shutil
import json
from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

TEST_DIR = "./.gitmem_coding_test"
AGENT_ID = "test_agent"
FILE_A = "test_file_a.py"
FILE_B = "test_file_b.py"

def setup():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Create dummy files
    with open(FILE_A, "w") as f:
        f.write("def func_a_unique():\n    pass\n\ndef common_func():\n    pass")
    
    with open(FILE_B, "w") as f:
        f.write("def func_b_unique():\n    pass\n\ndef common_func():\n    pass")

def cleanup():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    if os.path.exists(FILE_A):
        os.remove(FILE_A)
    if os.path.exists(FILE_B):
        os.remove(FILE_B)

def test_optimization():
    print("Initializing API...")
    api = CodingAPI(root_path=TEST_DIR)
    
    print(f"Creating flow for {FILE_A}...")
    api.create_flow(AGENT_ID, os.path.abspath(FILE_A))
    
    print(f"Creating flow for {FILE_B}...")
    api.create_flow(AGENT_ID, os.path.abspath(FILE_B))
    
    # Verify Global Index exists
    index_path = os.path.join(TEST_DIR, "index", "global_index.json")
    if not os.path.exists(index_path):
        print("FAIL: Global index file not created.")
        return
    
    with open(index_path, "r") as f:
        index = json.load(f)
        print("Global Index Content:", json.dumps(index, indent=2))
        
        if "func_a_unique" not in index:
            print("FAIL: func_a_unique not in index")
        if "func_b_unique" not in index:
            print("FAIL: func_b_unique not in index")
        if "common_func" not in index or len(index["common_func"]) != 2:
            print("FAIL: common_func not correctly indexed")

    # Verify Search / Get Flow
    print("\nTesting get_flow with symbol 'func_a_unique'...")
    res_a = api.get_flow(AGENT_ID, "func_a_unique")
    print("Result A:", res_a.get("count"), "matches")
    if res_a.get("count") != 1:
        print("FAIL: Expected 1 match for func_a_unique")
        
    print("\nTesting get_flow with symbol 'common_func'...")
    res_common = api.get_flow(AGENT_ID, "common_func")
    print("Result Common:", res_common.get("count"), "matches")
    if res_common.get("count") != 2:
        print("FAIL: Expected 2 matches for common_func")

    # Verify Cleanup on Delete
    print(f"\nDeleting flow for {FILE_A}...")
    api.delete_flow(AGENT_ID, os.path.abspath(FILE_A))
    
    with open(index_path, "r") as f:
        index = json.load(f)
        if "func_a_unique" in index:
            print("FAIL: func_a_unique should have been removed from index")
        if len(index["common_func"]) != 1:
            print("FAIL: common_func should have 1 entry left")
            
    print("\nSUCCESS: All tests passed!")

if __name__ == "__main__":
    setup()
    try:
        test_optimization()
    finally:
        cleanup()
