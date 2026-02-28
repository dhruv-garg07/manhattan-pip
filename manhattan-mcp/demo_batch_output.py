import sys
import os
import json
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

# Setup
test_dir = tempfile.mkdtemp(prefix="demo_batch_dep_")
api = CodingAPI(root_path=os.path.join(test_dir, ".gitmem_coding"))
AGENT = "demo_agent"

# Files specified by user
user_files = [
    "/Users/gargdhruv/Desktop/manhattan-pip/manhattan-mcp/src/manhattan_mcp/gitmem_coding/coding_vector_store.py",
    "/Users/gargdhruv/Desktop/manhattan-pip/manhattan-mcp/src/manhattan_mcp/config.py",
    "/Users/gargdhruv/Desktop/manhattan-pip/manhattan-mcp/src/manhattan_mcp/server_test.py",
    "/Users/gargdhruv/Desktop/manhattan-pip/manhattan-mcp/test_coding_env/test_file.py"
]

print("=== Indexing Files ===")
for f in user_files:
    if os.path.exists(f):
        api.index_file(AGENT, f)
        print(f"  Indexed: {os.path.basename(f)}")
    else:
        print(f"  Skipping (not found): {f}")

print("\n=== Calling dependency_graph(file_paths=user_files) ===")
result = api.dependency_graph(AGENT, user_files)

# Output the result in a pretty JSON format
print(json.dumps(result, indent=2))

# Cleanup
shutil.rmtree(test_dir, ignore_errors=True)
