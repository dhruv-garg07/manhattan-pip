"""Quick verification of embedding."""
import sys
sys.path.insert(0, 'src')
from manhattan_mcp.gitmem.embedding import RemoteEmbeddingClient

c = RemoteEmbeddingClient()
e = c.embed('test sentence')
print(f"Dimension: {len(e)}")
print(f"Type: {type(e).__name__}")
if hasattr(e, 'tolist'):
    vals = e.tolist()[:3]
elif hasattr(e, 'data'):
    vals = e.data[:3]
else:
    vals = list(e)[:3]
print(f"First 3 values: {vals}")
print("SUCCESS!")
