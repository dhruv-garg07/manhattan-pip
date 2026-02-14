
import os
import sys
import json

sys.path.insert(0, os.path.abspath("manhattan-mcp/src"))

from manhattan_mcp.gitmem_coding.ast_skeleton import ContextTreeBuilder

FILE_TO_TEST = "manhattan-mcp/src/manhattan_mcp/gitmem_coding/coding_store.py"

def analyze():
    if not os.path.exists(FILE_TO_TEST):
        print(f"File not found: {FILE_TO_TEST}")
        return

    with open(FILE_TO_TEST, "r") as f:
        content = f.read()
    
    original_size = len(content)
    print(f"Original Size: {original_size} bytes")
    
    # Chunk it first (mock chunking or use simple lines mapping?)
    # ContextTreeBuilder expects chunks. Ideally we should use the real chunker.
    from manhattan_mcp.gitmem_coding.chunking_engine import ChunkingEngine, detect_language
    
    language = detect_language(FILE_TO_TEST)
    chunker = ChunkingEngine.get_chunker(language)
    chunks_objs = chunker.chunk_file(content, FILE_TO_TEST)
    chunks = [c.to_dict() for c in chunks_objs]
    
    builder = ContextTreeBuilder()
    flow_data = builder.build(chunks, FILE_TO_TEST)
    
    skeleton_json = json.dumps(flow_data)
    skeleton_size = len(skeleton_json)
    
    print(f"Skeleton Size (JSON): {skeleton_size} bytes")
    
    compression_ratio = 1.0 - (skeleton_size / original_size)
    print(f"Compression Ratio: {compression_ratio:.2%}")
    
    # Print a sample of the skeleton tree content
    print("\n--- Skeleton Sample (Tree Structure) ---")
    tree = flow_data.get("tree", {})
    
    def print_node(node, depth=0):
        indent = "  " * depth
        content_preview = node['content'][:50].replace('\n', ' ')
        print(f"{indent}- [{node['type']}] {content_preview}...")
        for child in node.get('children', []):
            print_node(child, depth + 1)
            
    if tree:
        print_node(tree)

if __name__ == "__main__":
    analyze()
