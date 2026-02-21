import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from manhattan_mcp.gitmem_coding.coding_api import CodingAPI

SRC_DIR = os.path.join(os.path.dirname(__file__), "src", "manhattan_mcp")

REAL_FILES = {
    "server.py (Real-world API)": os.path.join(SRC_DIR, "server.py"),
    "coding_api.py (Large Module)": os.path.join(SRC_DIR, "gitmem_coding", "coding_api.py"),
}

def create_worst_case():
    # Hundreds of tiny functions. Structural metadata overhead is maximum.
    content = "# Worst Case: Many tiny functions\n"
    for i in range(250):
        content += f"def tiny_func_{i}(x):\n    return x + {i}\n\n"
    path = os.path.join(tempfile.gettempdir(), "worst_case.py")
    with open(path, "w") as f:
        f.write(content)
    return path

def create_best_case():
    # A few massive functions/classes with lots of internal repetitive logic
    # Context compression (summarization) shines here.
    content = "# Best Case: Massive internal logic\n"
    content += "class DeepProcessor:\n"
    content += "    def process_data_pipeline(self, data: list):\n"
    content += '        """Processes a giant pipeline with complex nested logic."""\n'
    for i in range(150):
        content += f"        data = [x * {i} for x in data if x % 2 == 0]\n"
        content += f"        if len(data) > {i * 10}:\n"
        content += f"            data = data[:100]\n"
    content += "        return data\n\n"
    
    content += "    def analyze_results(self, data: list):\n"
    content += '        """Analyzes results with massive internal repetitive checks."""\n'
    content += "        score = 0\n"
    for i in range(150):
        content += f"        if sum(data) > {i*100}:\n"
        content += f"            score += {i}\n"
    content += "        return score\n"
    
    path = os.path.join(tempfile.gettempdir(), "best_case.py")
    with open(path, "w") as f:
        f.write(content)
    return path

def run_comparison():
    print("=" * 70)
    print("  MANHATTAN MCP: CONTEXT & OUTLINE TOKEN OPTIMIZATION COMPARISON")
    print("=" * 70)
    
    files_to_test = {**REAL_FILES}
    files_to_test["Synthetic Best Case (Few massive functions)"] = create_best_case()
    files_to_test["Synthetic Worst Case (250 tiny functions)"] = create_worst_case()
    
    api = CodingAPI(root_path=tempfile.gettempdir())
    agent = "optimization_tester"
    
    for name, path in files_to_test.items():
        print(f"\nAnalyzing: {name}")
        print("-" * 50)
        
        # 1. Read Context (triggers auto-index)
        ctx = api.read_file_context(agent, path)
        if ctx.get("status") == "error":
            print(f"Error indexing {name}: {ctx}")
            continue
            
        ctx_ti = ctx.get("_token_info", {})
        raw = ctx_ti.get("tokens_if_raw_read", 1)
        ctx_used = ctx_ti.get("tokens_this_call", 0)
        ctx_saved = ctx_ti.get("tokens_saved", 0)
        ctx_ratio = (ctx_used / raw) * 100
        
        # 2. Get Outline
        out = api.get_file_outline(agent, path)
        out_ti = out.get("_token_info", {})
        out_used = out_ti.get("tokens_this_call", 0)
        out_ratio = (out_used / raw) * 100
        
        print(f"Raw File Tokens            : {raw}")
        print(f"Compressed Context Tokens  : {ctx_used} ({ctx_ratio:.1f}% of raw) -> Saved {ctx_saved} tokens")
        print(f"Structural Outline Tokens  : {out_used} ({out_ratio:.1f}% of raw)")

if __name__ == "__main__":
    run_comparison()
