#!/usr/bin/env python3
"""Test the improved search functionality."""

import sys
import os

# Add parent directory to path so we can import gitmem as a package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from gitmem.api import LocalAPI

# Clean start
import shutil
test_path = "./.gitmem_search_test"
if os.path.exists(test_path):
    shutil.rmtree(test_path)

api = LocalAPI(test_path)

# Add test memories
print("Adding test memories...")
api.add_memory("test", [{
    "lossless_restatement": "I love coffee in the morning",
    "keywords": ["coffee", "morning", "preference", "beverage"],
    "topic": "preferences"
}])
api.add_memory("test", [{
    "lossless_restatement": "Dhruv is building an AI context storage system called gitmem",
    "keywords": ["project", "gitmem", "AI", "storage"],
    "topic": "work"
}])
api.add_memory("test", [{
    "lossless_restatement": "User prefers Python over JavaScript for backend development",
    "keywords": ["Python", "JavaScript", "programming", "backend"],
    "topic": "preferences"
}])
api.add_memory("test", [{
    "lossless_restatement": "The user's favorite color is blue",
    "keywords": ["color", "blue", "favorite"],
    "topic": "preferences"
}])

print("=" * 60)
print("Search Tests")
print("=" * 60)

# Test 1: Query about morning coffee
print("\n1. Query: 'What does user love to have in morning?'")
results = api.search_memory("test", "What does user love to have in morning?")
print(f"   Results: {results['count']}")
for r in results["results"]:
    print(f"   - {r['lossless_restatement'][:50]}... (score: {r['score']})")

# Test 2: Direct coffee query
print("\n2. Query: 'coffee morning'")
results = api.search_memory("test", "coffee morning")
print(f"   Results: {results['count']}")
for r in results["results"]:
    print(f"   - {r['lossless_restatement'][:50]}... (score: {r['score']})")

# Test 3: Query about gitmem project
print("\n3. Query: 'What is the user building?'")
results = api.search_memory("test", "building project gitmem")
print(f"   Results: {results['count']}")
for r in results["results"]:
    print(f"   - {r['lossless_restatement'][:50]}... (score: {r['score']})")

# Test 4: Query about Python
print("\n4. Query: 'Python programming'")
results = api.search_memory("test", "Python programming")
print(f"   Results: {results['count']}")
for r in results["results"]:
    print(f"   - {r['lossless_restatement'][:50]}... (score: {r['score']})")

# Test 5: Query about favorite color
print("\n5. Query: 'favorite color'")
results = api.search_memory("test", "favorite color")
print(f"   Results: {results['count']}")
for r in results["results"]:
    print(f"   - {r['lossless_restatement'][:50]}... (score: {r['score']})")

print("\n" + "=" * 60)
print("Search tests complete!")

# Cleanup
shutil.rmtree(test_path)
