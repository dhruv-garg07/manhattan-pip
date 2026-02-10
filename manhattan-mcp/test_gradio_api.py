"""Debug test for Gradio API parsing."""
import urllib.request
import json
import time
import sys
sys.path.insert(0, '.')

url = 'https://iotacluster-embedding-model.hf.space/gradio_api/call/embed_dense'

print("=== GRADIO API DEBUG TEST ===")

# Step 1: POST
data = json.dumps({'data': ['test']}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
resp = urllib.request.urlopen(req, timeout=30)
result = json.loads(resp.read().decode('utf-8'))
event_id = result['event_id']
print(f"1. Event ID: {event_id}")

# Step 2: GET
time.sleep(2)
result_url = f'{url}/{event_id}'
req = urllib.request.Request(result_url)
resp = urllib.request.urlopen(req, timeout=60)
text = resp.read().decode('utf-8')

# Parse the event stream
for line in text.strip().split('\n'):
    line = line.strip()
    if line.startswith('data:'):
        json_str = line[5:].strip()
        if json_str:
            try:
                parsed = json.loads(json_str)
                print(f"2. Parsed type: {type(parsed).__name__}")
                
                if isinstance(parsed, list) and len(parsed) > 0:
                    first = parsed[0]
                    print(f"3. First element type: {type(first).__name__}")
                    
                    if isinstance(first, dict):
                        print(f"4. Dict keys: {list(first.keys())}")
                        for key, value in first.items():
                            if isinstance(value, list):
                                print(f"5. Key '{key}' has list of {len(value)} items")
                                if len(value) > 0:
                                    print(f"   First item type: {type(value[0]).__name__}")
                                    print(f"   First 3 values: {value[:3]}")
                                    print(f"\n=== EMBEDDING DIMENSION: {len(value)} ===")
            except json.JSONDecodeError as e:
                print(f"JSON error: {e}")

# Now test with our embedding client
print("\n=== TESTING EMBEDDING CLIENT ===")
import sys
sys.path.insert(0, 'src')
from manhattan_mcp.gitmem.embedding import RemoteEmbeddingClient

client = RemoteEmbeddingClient()
print(f"API URL: {client.api_url}")
print(f"Is Gradio: {client._is_gradio}")

embedding = client.embed("Hello world, test embedding")
print(f"\nEmbedding result:")
print(f"  Type: {type(embedding)}")
print(f"  Length: {len(embedding)}")
print(f"  First 5: {list(embedding[:5]) if hasattr(embedding, '__getitem__') else 'N/A'}")
