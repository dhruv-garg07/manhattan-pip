#!/usr/bin/env python3
"""
GitMem Local - Example Usage

This script demonstrates the core features of the local AI context storage system.
"""

import json
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gitmem.api import LocalAPI


def main():
    print("=" * 60)
    print("GitMem Local - Example Usage")
    print("=" * 60)
    
    # Initialize with custom path
    api = LocalAPI("./.gitmem_example")
    
    agent_id = "demo-agent"
    
    # ---------------------------------------------------------------------
    # 1. Session Start
    # ---------------------------------------------------------------------
    print("\n1. Starting session...")
    result = api.session_start(agent_id)
    print(f"   Session started: {result['session_id'][:8]}...")
    
    # ---------------------------------------------------------------------
    # 2. Add Memories
    # ---------------------------------------------------------------------
    print("\n2. Adding memories...")
    
    memories = [
        {
            "lossless_restatement": "The user's name is Dhruv",
            "keywords": ["name", "Dhruv", "identity"],
            "persons": ["Dhruv"],
            "topic": "personal"
        },
        {
            "lossless_restatement": "Dhruv prefers Python over JavaScript for backend development",
            "keywords": ["Python", "JavaScript", "backend", "preference"],
            "topic": "preferences"
        },
        {
            "lossless_restatement": "Dhruv is building an AI context storage system called gitmem",
            "keywords": ["project", "gitmem", "AI", "context", "storage"],
            "topic": "work"
        }
    ]
    
    result = api.add_memory(agent_id, memories)
    print(f"   Added {result['count']} memories")
    print(f"   Entry IDs: {[id[:8] for id in result['entry_ids']]}")
    
    # ---------------------------------------------------------------------
    # 3. Search Memories
    # ---------------------------------------------------------------------
    print("\n3. Searching memories...")
    
    result = api.search_memory(agent_id, "Python")
    print(f"   Found {result['count']} results for 'Python':")
    for mem in result['results'][:3]:
        content = mem['lossless_restatement'] or mem['content']
        print(f"      - {content[:60]}... (score: {mem['score']:.2f})")
    
    # ---------------------------------------------------------------------
    # 4. Auto-Remember
    # ---------------------------------------------------------------------
    print("\n4. Testing auto-remember...")
    
    test_messages = [
        "My birthday is March 15th",
        "I love coffee in the morning",
        "Just a random comment about the weather"
    ]
    
    for msg in test_messages:
        result = api.auto_remember(agent_id, msg)
        print(f"   '{msg[:30]}...' -> {result['extracted_facts']} facts extracted")
    
    # ---------------------------------------------------------------------
    # 5. List Virtual Directory
    # ---------------------------------------------------------------------
    print("\n5. Browsing virtual file system...")
    
    # List root
    result = api.list_dir(agent_id, "")
    print(f"   Root folders: {[n['name'] for n in result['nodes']]}")
    
    # List context types
    result = api.list_dir(agent_id, "context")
    print(f"   Context types: {[n['name'] for n in result['nodes']]}")
    
    # List semantic memories
    result = api.list_dir(agent_id, "context/semantic")
    print(f"   Semantic memories: {len(result['nodes'])} files")
    
    # ---------------------------------------------------------------------
    # 6. Get Agent Stats
    # ---------------------------------------------------------------------
    print("\n6. Getting agent statistics...")
    
    stats = api.get_agent_stats(agent_id)
    print(f"   Total memories: {stats['memories']['total']}")
    print(f"   - Episodic: {stats['memories']['episodic']}")
    print(f"   - Semantic: {stats['memories']['semantic']}")
    print(f"   - Procedural: {stats['memories']['procedural']}")
    print(f"   Documents: {stats['documents']}")
    print(f"   Checkpoints: {stats['checkpoints']}")
    
    # ---------------------------------------------------------------------
    # 7. Version Control
    # ---------------------------------------------------------------------
    print("\n7. Testing version control...")
    
    # Commit current state
    commit_sha = api.commit(agent_id, "Initial memory load")
    if commit_sha:
        print(f"   Committed: {commit_sha[:8]}")
    
    # Add more memories
    api.add_memory(agent_id, [{
        "lossless_restatement": "User's favorite color is blue",
        "keywords": ["favorite", "color", "blue"],
        "topic": "preferences"
    }])
    
    # Commit again
    commit_sha_2 = api.commit(agent_id, "Added favorite color")
    if commit_sha_2:
        print(f"   Second commit: {commit_sha_2[:8]}")
    
    # View history
    history = api.history(agent_id, limit=5)
    print(f"   History ({len(history)} commits):")
    for h in history:
        print(f"      - {h['sha'][:8]}: {h['message']}")
    
    # ---------------------------------------------------------------------
    # 8. Vector Search & Context Answer
    # ---------------------------------------------------------------------
    print("\n8. Testing Vector Search...")
    
    # Enable vector search
    api.enable_vectors(True)
    
    # 8a. Hybrid Search (combines semantic + keyword)
    print("\n   8a. Hybrid Search (semantic + keyword):")
    result = api.hybrid_search(agent_id, "user preferences programming")
    print(f"       Query: 'user preferences programming'")
    print(f"       Found: {result['count']} results (type: {result.get('search_type', 'fallback')})")
    for mem in result['results'][:3]:
        content = mem.get('lossless_restatement') or mem.get('content', '')
        score = mem.get('hybrid_score') or mem.get('score', 0)
        print(f"         - [{score:.3f}] {content[:50]}...")
    
    # 8b. Semantic Search (pure vector similarity)
    print("\n   8b. Semantic Search (vector similarity):")
    result = api.semantic_search(agent_id, "what does the user enjoy")
    print(f"       Query: 'what does the user enjoy'")
    print(f"       Found: {result['count']} results (type: {result.get('search_type', 'fallback')})")
    for mem in result['results'][:3]:
        content = mem.get('lossless_restatement') or mem.get('content', '')
        score = mem.get('semantic_score') or mem.get('score', 0)
        print(f"         - [{score:.3f}] {content[:50]}...")
    
    # 8c. Vector Statistics
    print("\n   8c. Vector Storage Stats:")
    stats = api.get_vector_stats(agent_id)
    print(f"       Vectors stored: {stats.get('vector_count', 0)}")
    print(f"       Dimension: {stats.get('dimension', 'N/A')}")
    print(f"       Cache size: {stats.get('cache_size', 0)}")
    
    # 8d. Context Answer (uses hybrid search internally)
    print("\n   8d. Context-Aware Answer:")
    result = api.get_context_answer(agent_id, "What does user loves to have in morning?")
    print(f"       Query: 'What does user loves to have in morning?'")
    print(f"       Sources found: {result['source_count']}")
    for src in result['sources'][:3]:
        print(f"         - {src['content'][:50]}...")
    
    # ---------------------------------------------------------------------
    # 9. Memory Summary
    # ---------------------------------------------------------------------
    print("\n9. Getting memory summary...")
    
    result = api.memory_summary(agent_id)
    print(f"   Topics: {result['topics']}")
    print(f"   Total memories: {result['total_memories']}")
    
    # ---------------------------------------------------------------------
    # 10. Export
    # ---------------------------------------------------------------------
    print("\n10. Exporting memories...")
    
    backup = api.export_memories(agent_id)
    print(f"   Exported {len(backup.get('memories', []))} memories")
    print(f"   Export version: {backup.get('version', 'unknown')}")
    
    # Save to file
    export_path = "./.gitmem_example/export.json"
    with open(export_path, 'w') as f:
        json.dump(backup, f, indent=2)
    print(f"   Saved to: {export_path}")
    
    # ---------------------------------------------------------------------
    # 11. Session End
    # ---------------------------------------------------------------------
    print("\n11. Ending session...")
    
    result = api.session_end(
        agent_id,
        conversation_summary="Demo session showing all GitMem features",
        key_points=[
            "User's name is Dhruv",
            "Prefers Python for backend",
            "Building gitmem project"
        ]
    )
    print(f"   Session ended, checkpoint created: {result['checkpoint_created']}")
    
    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(f"\nData stored in: {os.path.abspath('./.gitmem_example')}")
    print("\nYou can now:")
    print("  - Browse the JSON files to see stored data")
    print("  - Explore .gitmem/ for version control objects")
    print("  - Run this script again to see persistence")


if __name__ == "__main__":
    main()
