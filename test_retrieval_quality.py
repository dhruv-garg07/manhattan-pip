
import os
import sys
import json
import shutil

sys.path.insert(0, os.path.abspath("manhattan-mcp/src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

TEST_DIR = "test_retrieval_quality_db"
AGENT_ID = "test_agent"
FILE_PATH = os.path.abspath("manhattan-mcp/src/manhattan_mcp/gitmem_coding/coding_store.py")

def test_retrieval():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    
    api = CodingAPI(root_path=TEST_DIR)
    
    # Ingest the file
    print(f"Ingesting {FILE_PATH}...")
    api.create_mem(AGENT_ID, FILE_PATH)
    
    print("\n--- Strategy 1: Path Retrieval (returns skeleton) ---")
    result_path = api.get_mem(AGENT_ID, FILE_PATH)
    # The return format of get_mem for path is store.retrieve_file_context
    print(f"Status: {result_path.get('status')}")
    # print(json.dumps(result_path.get('code_flow'), indent=2)[:500] + "...")
    
    print("\n--- Strategy 2: Search Inquiry (returns chunks) ---")
    # Query for something specific in the file, e.g., "store_file_chunks"
    query = "How is store_file_chunks implemented?"
    result_search = api.get_mem(AGENT_ID, query)
    
    # CodingHybridRetriever returns a dict with 'results' list
    matches = result_search.get('results', [])
    print(f"Matches found: {len(matches)}")
    for i, match in enumerate(matches[:2]):
        print(f"\nMatch {i+1}:")
        print(f"  File: {match.get('file_path')}")
        chunk = match.get('chunk', {})
        print(f"  Content Type: {chunk.get('type')}")
        print(f"  Content Snippet: {chunk.get('content', '')[:200]}...")
        print(f"  Summary: {chunk.get('summary', 'N/A')}")

if __name__ == "__main__":
    test_retrieval()
