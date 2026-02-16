#!/usr/bin/env python3
"""
Diagnostic: trace what happens when query is JUST "/api/keys"
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

# Use the REAL production storage path
PROD_ROOT = os.path.expanduser("~/Library/Application Support/manhattan-mcp/.gitmem_coding")

api = CodingAPI(root_path=PROD_ROOT)

query = "/api/keys"
agent_id = "default"

print("=" * 70)
print(f"QUERY: {query}")
print(f"AGENT: {agent_id}")
print("=" * 70)

# Run the search via the full API
print("\n--- Running api.get_flow() ---")
result = api.get_flow(agent_id, query)

if isinstance(result, str):
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2)[:3000])
    except:
        print(result[:3000])
elif isinstance(result, dict):
    print(json.dumps(result, indent=2)[:3000])

print("\n" + "=" * 70)
print("DONE")
