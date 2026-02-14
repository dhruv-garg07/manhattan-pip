
import os
import shutil
import json
import sys

# Ensure src is in path
sys.path.insert(0, os.path.abspath("manhattan-mcp/src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI
from manhattan_mcp.gitmem_coding.coding_store import CodingContextStore
from manhattan_mcp.gitmem_coding.coding_memory_builder import CodingMemoryBuilder

TEST_DIR = "test_gitmem_dedup"
AGENT_ID = "test_agent"
FILE_PATH = os.path.abspath("test_dedup.py")

# Mock embedding client with counter
class MockEmbeddingClient:
    def __init__(self):
        self.call_count = 0
        
    def embed(self, text):
        self.call_count += 1
        return [0.1, 0.2, 0.3]

def setup():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    with open(FILE_PATH, "w") as f:
        # A file with 2 chunks: one function, one class
        f.write("def stable_func():\n    return 42\n\nclass StableClass:\n    pass\n")

def teardown():
    if os.path.exists(TEST_DIR):
        # shutil.rmtree(TEST_DIR) # Keep for inspection if needed
        pass
    if os.path.exists(FILE_PATH):
        os.remove(FILE_PATH)

def verify():
    setup()
    try:
        print(f"Initializing API at {TEST_DIR}...")
        api = CodingAPI(root_path=TEST_DIR)
        
        # Patch embedding client
        mock_client = MockEmbeddingClient()
        api.builder.embedding_client = mock_client
        
        print("\n--- Run 1: Initial Ingestion ---")
        api.create_flow(AGENT_ID, FILE_PATH, chunks=None)
        
        count_run1 = mock_client.call_count
        print(f"Embedding Calls Run 1: {count_run1}")
        
        if count_run1 == 0:
            print("FAILURE: Expected embedding calls in Run 1.")
            return

        print("\n--- Run 2: Re-Ingestion (Same Content) ---")
        # Reset counter? No, let's keep cumulative or reset.
        mock_client.call_count = 0
        
        # We need to simulate a fresh builder/api session? 
        # Or just call again. Deduplication is via Store (disk based).
        # But to be sure, let's reload the API to simulate new session context
        api2 = CodingAPI(root_path=TEST_DIR)
        api2.builder.embedding_client = mock_client # Same mock instance to track calls?
        # No, mock_client.call_count is local to the instance.
        
        api2.create_flow(AGENT_ID, FILE_PATH, chunks=None)
        
        count_run2 = mock_client.call_count
        print(f"Embedding Calls Run 2: {count_run2}")
        
        if count_run2 == 0:
            print("SUCCESS: 0 embedding calls in Run 2. Deduplication working!")
        else:
            print(f"FAILURE: Expected 0 calls in Run 2, got {count_run2}.")

        # Verify chunks.json exists
        chunks_path = os.path.join(TEST_DIR, "chunks.json")
        if os.path.exists(chunks_path):
             with open(chunks_path) as f:
                 data = json.load(f)
                 print(f"\nGlobal Chunk Registry Items: {len(data)}")
                 if len(data) > 0:
                     print("SUCCESS: Chunks persisted to global registry.")
        else:
             print("FAILURE: chunks.json not found.")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        teardown()

if __name__ == "__main__":
    verify()
