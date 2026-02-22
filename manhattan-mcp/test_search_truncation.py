
import os
import sys
import json
import asyncio
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent / "src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

async def test_truncation():
    # Setup - use a temp directory for gitmem coding data
    test_root = Path("./test_gitmem_data").absolute()
    test_root.mkdir(exist_ok=True)
    
    api = CodingAPI(root_path=str(test_root))
    agent_id = "test_agent"
    
    # Create a large file
    test_file = test_root / "large_file.py"
    large_content = 'def large_function():\n    """\n    This function has a lot of content.\n    ' + "x = 1\n    " * 500 + '"""\n    pass\n'
    test_file.write_text(large_content)
    
    print(f"File size: {len(large_content)} characters")
    
    # Index the file
    print("Indexing file...")
    api.index_file(agent_id, str(test_file))
    
    # Search for the function
    print("Searching codebase...")
    results = api.search_codebase(agent_id, "large_function")
    
    # Verify results
    if results["status"] == "search_results" and results["results"]:
        top_chunk = results["results"][0]["chunk"]
        content = top_chunk.get("content", "")
        truncated = top_chunk.get("content_truncated", False)
        
        print(f"Matched chunk: {top_chunk.get('name')}")
        print(f"Content length: {len(content)}")
        print(f"Content truncated flag: {truncated}")
        
        if len(content) <= 1050 and truncated:
            print("SUCCESS: Content was truncated correctly.")
        else:
            print("FAILURE: Content was not truncated as expected.")
            if len(content) > 1050:
                print(f"Actual length: {len(content)}")
    else:
        print("FAILURE: No search results found.")

    # Cleanup
    # test_file.unlink()
    # (Optional: remove the test_root entirely)

if __name__ == "__main__":
    asyncio.run(test_truncation())
