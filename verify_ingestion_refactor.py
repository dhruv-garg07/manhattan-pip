
import os
import shutil
import json
import sys

# Ensure src is in path
sys.path.insert(0, os.path.abspath("manhattan-mcp/src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI
from manhattan_mcp.gitmem_coding.coding_store import CodingContextStore
from manhattan_mcp.gitmem_coding.coding_memory_builder import CodingMemoryBuilder

TEST_DIR = "test_gitmem_refactor"
AGENT_ID = "test_agent"
FILE_PATH = os.path.abspath("test_file.py")

def setup():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    # Ensure test dir exists before cleanup (coding store created it)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    with open(FILE_PATH, "w") as f:
        f.write("def hello():\n    print('world')\n")

def teardown():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    if os.path.exists(FILE_PATH):
        os.remove(FILE_PATH)

def verify():
    # Setup
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    
    with open(FILE_PATH, "w") as f:
        f.write("def hello():\n    print('world')\n")
        
    try:
        print(f"Initializing API with root_path={TEST_DIR}...")
        api = CodingAPI(root_path=TEST_DIR)
        store = api.store # Access the store created by API
        
        # Mock embedding client
        class MockEmbeddingClient:
            def embed(self, text):
                return [0.1, 0.2, 0.3]
        
        # Patch the embedding client on the builder instance
        api.builder.embedding_client = MockEmbeddingClient()
        
        print("Calling create_mem with chunks=None...")
        # This triggers the new _read_and_chunk_file path
        result = api.create_mem(AGENT_ID, FILE_PATH, chunks=None)
        
        print("Result:", json.dumps(result, indent=2))
        
        # Verify storage
        contexts = store._load_agent_data(AGENT_ID, "file_contexts")
        if not contexts:
            print("FAILURE: No contexts stored.")
            return

        ctx = contexts[0]
        print("\nVerifying Stored Context:")
        print(f"  File: {ctx.get('file_path')}")
        print(f"  Chunks Count: {len(ctx.get('chunks', []))}")
        
        chunks = ctx.get('chunks', [])
        if len(chunks) > 0:
            print("SUCCESS: Chunks were generated and stored.")
            # Verify vector field presence
            chunk = chunks[0]
            if "vector" in chunk:
                 print(f"SUCCESS: Vector field exists in chunk. Len: {len(chunk['vector'])}")
                 if chunk['vector'] == [0.1, 0.2, 0.3]:
                     print("SUCCESS: Mock embedding was used.")
                 else:
                     print("WARNING: Vector found but does not match mock.")
            else:
                 print("FAILURE: Vector field missing in chunk.")
        else:
            print("FAILURE: No chunks generated.")
            
        if ctx.get('compact_skeleton'):
            print("SUCCESS: compact_skeleton is present.")
        else:
            print("FAILURE: compact_skeleton is missing.")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        teardown()

if __name__ == "__main__":
    verify()
