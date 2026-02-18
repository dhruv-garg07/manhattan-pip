
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "manhattan-mcp" / "src"))

from manhattan_mcp.gitmem.embedding import RemoteEmbeddingClient

def benchmark():
    client = RemoteEmbeddingClient(cache_embeddings=False)
    
    test_texts = [
        f"This is a test sentence number {i} to benchmark embedding generation speed."
        for i in range(20)
    ]
    
    print(f"\nBenchmarking with {len(test_texts)} texts...")
    
    # 1. Sequential
    print("Running sequential embeddings...")
    start_seq = time.time()
    for text in test_texts:
        client.embed(text)
    end_seq = time.time()
    seq_time = end_seq - start_seq
    print(f"Sequential time: {seq_time:.2f}s")
    
    # 2. Parallel (Batch)
    print("\nRunning parallel (batch) embeddings...")
    start_par = time.time()
    client.embed_batch(test_texts, max_workers=10)
    end_par = time.time()
    par_time = end_par - start_par
    print(f"Parallel time: {par_time:.2f}s")
    
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark()
