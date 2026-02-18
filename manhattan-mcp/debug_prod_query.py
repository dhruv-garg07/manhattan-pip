#!/usr/bin/env python3
"""
Diagnostic: trace what happens with the exact user query against REAL production data.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

# Use the REAL production storage path
PROD_ROOT = os.path.expanduser("~/Library/Application Support/manhattan-mcp/.gitmem_coding")

api = CodingAPI(root_path=PROD_ROOT)

query = "Can you explain '/api/keys' endpoint from index.py file? Pull context only from manhattan_mcp"
agent_id = "default"

print("=" * 70)
print(f"QUERY: {query}")
print(f"AGENT: {agent_id}")
print("=" * 70)

# Run the search via the full API (get_mem calls retriever under the hood)
print("\n--- Running api.get_mem() ---")
result = api.get_mem(agent_id, query)
print(f"Type: {type(result)}")
if isinstance(result, str):
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2)[:3000])
    except:
        print(result[:3000])
elif isinstance(result, dict):
    print(json.dumps(result, indent=2)[:3000])
elif isinstance(result, list):
    print(f"List with {len(result)} items")
    for i, item in enumerate(result[:5]):
        if isinstance(item, dict):
            print(f"  [{i+1}] {json.dumps(item, indent=2)[:300]}")
        else:
            print(f"  [{i+1}] {str(item)[:300]}")

# Also try direct retriever
print("\n--- Running api.retriever.search() directly ---")
results = api.retriever.search(agent_id, query, top_k=5)
print(f"Type: {type(results)}, Count: {len(results)}")
for i, r in enumerate(results):
    if isinstance(r, dict):
        name = r.get("name", "?")
        score = r.get("score", 0)
        print(f"  [{i+1}] {name} (score={score:.4f})")
    else:
        print(f"  [{i+1}] type={type(r)} value={str(r)[:200]}")

print("\n" + "=" * 70)
print("DONE")
