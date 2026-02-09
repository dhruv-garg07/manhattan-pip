"""
Test script for GitMem Local Vector Storage Engine

Tests:
1. Remote embedding client
2. Local vector store
3. Hybrid search
4. Integration with memory store
"""

import sys
import os
from pathlib import Path

# Add gitmem to path
gitmem_path = Path(__file__).parent.absolute()
if str(gitmem_path) not in sys.path:
    sys.path.insert(0, str(gitmem_path.parent))

from gitmem import LocalMemoryStore

# Test constants
TEST_AGENT_ID = "test-vector-agent"


def test_memory_store_with_vectors():
    """Test the memory store with vector search capabilities."""
    print("\n" + "="*60)
    print("Test 1: Memory Store with Vector Search")
    print("="*60)
    
    # Create memory store with vectors enabled
    store = LocalMemoryStore(
        root_path="./.gitmem_test_data",
        enable_vectors=True
    )
    
    print(f"✓ Memory store created")
    print(f"  Root path: {store.root_path}")
    print(f"  Vectors enabled: {store.enable_vectors}")
    
    # Add some test memories
    memories_to_add = [
        {
            "content": "User's name is Alice",
            "lossless_restatement": "The user introduced themselves as Alice",
            "keywords": ["name", "Alice", "introduction"],
            "topic": "personal information"
        },
        {
            "content": "User likes Python programming",
            "lossless_restatement": "Alice expressed that she enjoys programming in Python",
            "keywords": ["Python", "programming", "preferences"],
            "topic": "technical preferences"
        },
        {
            "content": "User's favorite color is blue",
            "lossless_restatement": "Alice mentioned that her favorite color is blue",
            "keywords": ["color", "blue", "favorite", "preferences"],
            "topic": "personal preferences"
        },
        {
            "content": "User works at TechCorp",
            "lossless_restatement": "Alice works as a software engineer at TechCorp company",
            "keywords": ["work", "TechCorp", "software engineer", "job"],
            "topic": "employment"
        },
        {
            "content": "User prefers dark mode",
            "lossless_restatement": "Alice prefers using dark mode theme in applications",
            "keywords": ["dark mode", "theme", "preferences", "UI"],
            "topic": "UI preferences"
        }
    ]
    
    print(f"\n  Adding {len(memories_to_add)} test memories...")
    
    for mem in memories_to_add:
        entry_id = store.add_memory(
            agent_id=TEST_AGENT_ID,
            content=mem["content"],
            lossless_restatement=mem["lossless_restatement"],
            keywords=mem["keywords"],
            topic=mem["topic"]
        )
        print(f"  ✓ Added: {entry_id[:8]}... - {mem['topic']}")
    
    return store


def test_keyword_search(store: LocalMemoryStore):
    """Test keyword-based search."""
    print("\n" + "="*60)
    print("Test 2: Keyword Search")
    print("="*60)
    
    test_queries = [
        "Python programming",
        "What is Alice's favorite color?",
        "dark mode preferences",
        "Where does Alice work?"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        results = store.search_memory(TEST_AGENT_ID, query, top_k=3)
        
        if results:
            for i, r in enumerate(results):
                content = r.get("lossless_restatement", "")[:50]
                score = r.get("score", 0)
                print(f"    {i+1}. [{score:.3f}] {content}...")
        else:
            print("    No results found")


def test_hybrid_search(store: LocalMemoryStore):
    """Test hybrid search combining semantic and keyword."""
    print("\n" + "="*60)
    print("Test 3: Hybrid Search (Semantic + Keyword)")
    print("="*60)
    
    test_queries = [
        "user's programming language preference",
        "personal details about the user",
        "what theme does the user like",
        "employment information"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        
        try:
            results = store.hybrid_search_memory(TEST_AGENT_ID, query, top_k=3)
            
            if results:
                for i, r in enumerate(results):
                    content = r.get("lossless_restatement", "")[:50]
                    hybrid_score = r.get("hybrid_score", 0)
                    semantic_score = r.get("semantic_score", 0)
                    keyword_score = r.get("keyword_score", 0)
                    
                    if hybrid_score:
                        print(f"    {i+1}. [H:{hybrid_score:.3f} S:{semantic_score:.3f} K:{keyword_score:.3f}] {content}...")
                    else:
                        score = r.get("score", 0)
                        print(f"    {i+1}. [score:{score:.3f}] {content}... (keyword fallback)")
            else:
                print("    No results found")
        except Exception as e:
            print(f"    Error: {e}")
            print("    (Hybrid search may require numpy and a working embedding API)")


def test_vector_stats(store: LocalMemoryStore):
    """Test vector statistics."""
    print("\n" + "="*60)
    print("Test 4: Vector Statistics")
    print("="*60)
    
    try:
        stats = store.get_vector_stats(TEST_AGENT_ID)
        print(f"  Vector stats: {stats}")
    except Exception as e:
        print(f"  Error getting stats: {e}")


def cleanup_test_data():
    """Clean up test data."""
    import shutil
    test_path = Path("./.gitmem_test_data")
    if test_path.exists():
        shutil.rmtree(test_path)
        print("\n✓ Test data cleaned up")


def main():
    print("\n" + "="*60)
    print("  GitMem Local - Vector Storage Engine Tests")
    print("="*60)
    
    try:
        # Run tests
        store = test_memory_store_with_vectors()
        test_keyword_search(store)
        test_hybrid_search(store)
        test_vector_stats(store)
        
        print("\n" + "="*60)
        print("  All tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ask user before cleanup
        print("\n[Note: Test data is in .gitmem_test_data/]")
        response = input("Clean up test data? (y/n): ").strip().lower()
        if response == 'y':
            cleanup_test_data()


if __name__ == "__main__":
    main()
