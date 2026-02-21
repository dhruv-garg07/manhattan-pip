
import os
import sys
import json
import traceback
from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

def simulate_all_indexing():
    api = CodingAPI(root_path="./.gitmem_test_all")
    agent_id = "test_agent"
    
    # Get all python files
    root_dir = r"c:\Desktop\python_workspace_311\PdM-main\PdM-main\manhattan-pip"
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        if ".gitmem" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    
    print(f"Found {len(py_files)} Python files to test.")
    
    for i, file_path in enumerate(py_files):
        print(f"[{i+1}/{len(py_files)}] Indexing {file_path}...", end=" ", flush=True)
        try:
            # Capture stdout if possible? No, we just want to see if it prints.
            # But we are ALREADY in a script that prints to stdout.
            # So any 'illegal' prints from the API will show up here.
            
            result = api.index_file(agent_id, file_path)
            print("Done.")
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    # Add src to path
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    simulate_all_indexing()
