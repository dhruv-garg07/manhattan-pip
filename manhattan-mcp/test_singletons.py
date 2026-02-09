"""
Test global singletons for vector storage and retrieval.
"""
import sys
import os
sys.path.insert(0, '.')

from gitmem.embedding import get_embedding_client, RemoteEmbeddingClient
from gitmem.vector_store import get_vector_store, LocalVectorStore
from gitmem.hybrid_retriever import get_retriever, HybridRetriever
from gitmem.memory_store import LocalMemoryStore

def test_embedding_singleton():
    print("\n--- Testing Embedding Client Singleton ---")
    c1 = get_embedding_client()
    c2 = get_embedding_client()
    
    print(f"Client 1 ID: {id(c1)}")
    print(f"Client 2 ID: {id(c2)}")
    
    if c1 is c2:
        print("SUCCESS: Embedding client is a singleton")
    else:
        print("FAILURE: Embedding client created multiple instances")

def test_vector_store_singleton():
    print("\n--- Testing Vector Store Singleton ---")
    v1 = get_vector_store()
    v2 = get_vector_store()
    
    print(f"Store 1 ID: {id(v1)}")
    print(f"Store 2 ID: {id(v2)}")
    
    if v1 is v2:
        print("SUCCESS: Vector store is a singleton")
    else:
        print("FAILURE: Vector store created multiple instances")
        
    # Check if embedding client in vector store matches global singleton
    ec = get_embedding_client()
    if v1.embedding_client is ec:
        print("SUCCESS: Vector store uses global embedding client")
    else:
        print("FAILURE: Vector store created new embedding client")

def test_retriever_singleton():
    print("\n--- Testing Retriever Singleton ---")
    r1 = get_retriever()
    r2 = get_retriever()
    
    print(f"Retriever 1 ID: {id(r1)}")
    print(f"Retriever 2 ID: {id(r2)}")
    
    if r1 is r2:
        print("SUCCESS: Retriever is a singleton")
    else:
        print("FAILURE: Retriever created multiple instances")

    # Check if vector store matches global singleton
    vs = get_vector_store()
    if r1.vector_store is vs:
        print("SUCCESS: Retriever uses global vector store")
    else:
        print("FAILURE: Retriever created new vector store")

def test_memory_store_integration():
    print("\n--- Testing Memory Store Integration ---")
    ms1 = LocalMemoryStore(root_path="./.gitmem_test_1")
    ms2 = LocalMemoryStore(root_path="./.gitmem_test_2")
    
    print("Accessing ms1.vector_store...")
    vs1 = ms1.vector_store
    
    print("Accessing ms2.vector_store...")
    vs2 = ms2.vector_store
    
    if vs1 is vs2:
        print("SUCCESS: Different MemoryStores share same VectorStore singleton")
    else:
        print("FAILURE: MemoryStores created different VectorStores")

if __name__ == "__main__":
    print("Starting Singleton Tests...")
    test_embedding_singleton()
    test_vector_store_singleton()
    test_retriever_singleton()
    test_memory_store_integration()
    print("\nTests Complete!")
