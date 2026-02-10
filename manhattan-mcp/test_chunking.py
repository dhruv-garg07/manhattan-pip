
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import shutil
import json
from manhattan_mcp.gitmem_coding.coding_store import CodingContextStore

# Setup test environment
TEST_DIR = "test_gitmem_coding_chunking"
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)

print(f"Initializing store in {TEST_DIR}...")
store = CodingContextStore(root_path=TEST_DIR)

# Define some dummy python code
file1_content = """
def shared_function():
    print("This is a shared function")
    return True

def unique_function_1():
    x = 10
    y = 20
    return x + y
"""

file2_content = """
import os

def shared_function():
    print("This is a shared function")
    return True

def unique_function_2():
    print("I am unique to file 2")
"""

agent_id = "test_agent"

# Store File 1
print("\nStoring File 1...")
res1 = store.store_file_context(
    agent_id=agent_id,
    file_path="/src/file1.py",
    content=file1_content,
    language="python"
)
print("File 1 Result:", json.dumps(res1, indent=2))

# Store File 2 (Contains shared_function)
print("\nStoring File 2...")
res2 = store.store_file_context(
    agent_id=agent_id,
    file_path="/src/file2.py",
    content=file2_content,
    language="python"
)
print("File 2 Result:", json.dumps(res2, indent=2))

# Check Stats
print("\nChecking Stats...")
stats = store.get_stats(agent_id)
print("Files Cached:", stats["total_files_cached"])
print("Global Unique Chunks:", stats["global_unique_chunks"])

# Analysis:
# File 1 has: shared_function, unique_function_1 -> 2 chunks
# File 2 has: shared_function, unique_function_2 -> 2 chunks (shared_function is dup)
# Expected Unique Chunks: 3 (shared, unique1, unique2)

assert stats["global_unique_chunks"] == 3, f"Expected 3 unique chunks, found {stats['global_unique_chunks']}"

# Verify Chunk Registry Content
chunks_path = os.path.join(TEST_DIR, "chunks.json")
with open(chunks_path, 'r') as f:
    chunks = json.load(f)

print("\nChunk Registry Keys:", list(chunks.keys()))
chunk_names = [c["name"] for c in chunks.values()]
print("Chunk Names:", chunk_names)

assert "shared_function" in chunk_names
assert "unique_function_1" in chunk_names
assert "unique_function_2" in chunk_names

print("\nSUCCESS: Chunking and Deduplication verified!")

# Cleanup
# shutil.rmtree(TEST_DIR)
