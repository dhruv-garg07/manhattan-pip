
import os
import sys
import json
from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

def test_index():
    print("Testing indexing...")
    api = CodingAPI(root_path="./.gitmem_test")
    
    file_path = r"c:\Desktop\python_workspace_311\PdM-main\PdM-main\manhattan-pip\manhattan-mcp\src\manhattan_mcp\config.py"
    agent_id = "test_agent"
    
    print(f"Indexing {file_path}...")
    try:
        result = api.index_file(agent_id, file_path)
        print("Success!")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Caught Exception: {e}")

if __name__ == "__main__":
    # Add src to path
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    test_index()
