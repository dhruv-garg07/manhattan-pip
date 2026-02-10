"""Simplified test for gitmem vector storage - outputs to console only."""
import sys
sys.path.insert(0, '.')
import shutil
from pathlib import Path

print("=" * 60)
print("  GitMem Local - Vector Storage Test")
print("=" * 60)

# Clean previous test data
TEST_PATH = "./.gitmem_quick_test"
if Path(TEST_PATH).exists():
    shutil.rmtree(TEST_PATH)

# Test 1: Import modules
print("\n[1] Testing Module Imports...")
try:
    import sys
    sys.path.insert(0, 'src')
    from manhattan_mcp.gitmem.embedding import RemoteEmbeddingClient, HAS_NUMPY
    print(f"    RemoteEmbeddingClient: OK")
    print(f"    HAS_NUMPY: {HAS_NUMPY}")
except ImportError as e:
    print(f"    RemoteEmbeddingClient: FAILED - {e}")

try:
    from manhattan_mcp.gitmem.vector_store import LocalVectorStore
    print(f"    LocalVectorStore: OK")
except ImportError as e:
    print(f"    LocalVectorStore: FAILED - {e}")

try:
    from manhattan_mcp.gitmem.hybrid_retriever import HybridRetriever, RetrievalConfig
    print(f"    HybridRetriever: OK")
except ImportError as e:
    print(f"    HybridRetriever: FAILED - {e}")

# Test 2: Memory Store
print("\n[2] Testing Memory Store...")
from manhattan_mcp.gitmem import LocalMemoryStore

store = LocalMemoryStore(root_path=TEST_PATH, enable_vectors=True)
print(f"    Created store at: {store.root_path}")
print(f"    Vectors enabled: {store.enable_vectors}")

# Test 3: Add memories
print("\n[3] Adding Test Memories...")
memories = [
    {"content": "User is Alice", "lossless_restatement": "The user's name is Alice", "keywords": ["name", "Alice"], "topic": "identity"},
    {"content": "User loves Python", "lossless_restatement": "Alice loves programming in Python", "keywords": ["Python", "programming"], "topic": "preferences"},
    {"content": "User's favorite is blue", "lossless_restatement": "Alice's favorite color is blue", "keywords": ["color", "blue"], "topic": "preferences"},
]

for mem in memories:
    entry_id = store.add_memory(agent_id="test-agent", **mem)
    print(f"    Added: {entry_id[:8]}... ({mem['topic']})")

# Test 4: Keyword search
print("\n[4] Keyword Search Test...")
results = store.search_memory("test-agent", "Python programming", top_k=3)
print(f"    Query: 'Python programming'")
print(f"    Results: {len(results)}")
for r in results:
    print(f"      - [{r.get('score', 0):.3f}] {r.get('lossless_restatement', '')[:40]}...")

# Test 5: Hybrid search
print("\n[5] Hybrid Search Test...")
try:
    results = store.hybrid_search_memory("test-agent", "user preferences", top_k=3)
    print(f"    Query: 'user preferences'")
    print(f"    Results: {len(results)}")
    for r in results:
        hybrid = r.get('hybrid_score')
        if hybrid is not None:
            print(f"      - [H:{hybrid:.2f}] {r.get('lossless_restatement', '')[:35]}...")
        else:
            print(f"      - [{r.get('score', 0):.2f}] {r.get('lossless_restatement', '')[:35]}... (fallback)")
except Exception as e:
    print(f"    Hybrid search error: {e}")

# Test 6: Vector store stats
print("\n[6] Vector Store Stats...")
try:
    if store.vector_store:
        stats = store.get_vector_stats("test-agent")
        print(f"    Stats: {stats}")
    else:
        print("    Vector store not initialized")
except Exception as e:
    print(f"    Error: {e}")

# Cleanup
print("\n" + "=" * 60)
shutil.rmtree(TEST_PATH)
print("  All tests completed! Test data cleaned up.")
print("=" * 60)
