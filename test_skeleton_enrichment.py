#!/usr/bin/env python3
"""Test enriched AST skeleton — direct import, no heavy dependencies."""

import sys
import os
import importlib.util

# Direct-import ast_skeleton.py without going through the full package
SKELETON_PATH = os.path.join(
    os.path.dirname(__file__),
    "manhattan-mcp", "src", "manhattan_mcp", "gitmem_coding", "ast_skeleton.py"
)

spec = importlib.util.spec_from_file_location("ast_skeleton", SKELETON_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ASTSkeletonGenerator = mod.ASTSkeletonGenerator


def test_index_py():
    """Test skeleton generation on real index.py."""
    index_path = os.path.join(os.path.dirname(__file__), "index.py")
    
    with open(index_path, "r") as f:
        content = f.read()
    
    gen = ASTSkeletonGenerator()
    skeleton = gen.generate_skeleton(content, "python", index_path)
    
    original_tokens = len(content.split())
    skeleton_tokens = len(skeleton.split())
    ratio = skeleton_tokens / original_tokens
    reduction = (1 - ratio) * 100
    
    print(f"=== Token Stats ===")
    print(f"Original:  {original_tokens:,} tokens")
    print(f"Skeleton:  {skeleton_tokens:,} tokens")
    print(f"Reduction: {reduction:.1f}%")
    print()
    
    # ── Key operations that must appear ──
    key_operations = [
        ("DB: profiles table", "supabase.table('profiles')"),
        ("DB: select", ".select("),
        ("DB: insert", ".insert("),
        ("DB: execute", ".execute()"),
        ("DB: upsert", ".upsert("),
        ("Auth: sign_up", ".sign_up("),
        ("Auth: sign_in", "sign_in_with_password("),
        ("Auth: get_user", ".get_user("),
        ("Flask: render_template", "render_template("),
        ("Flask: redirect", "redirect("),
        ("Flask: jsonify", "jsonify("),
        ("Async: asyncio.run", "asyncio.run("),
        ("Return stmt", "return "),
    ]
    
    found = []
    missing = []
    for label, op in key_operations:
        if op in skeleton:
            found.append(label)
        else:
            missing.append(label)
    
    print(f"=== Key Operations ({len(found)}/{len(key_operations)}) ===")
    for label in found:
        print(f"  ✅ {label}")
    if missing:
        for label in missing:
            print(f"  ❌ {label}")
    print()
    
    # ── Reduction target ──
    print(f"=== Reduction Target ===")
    if reduction >= 45:
        print(f"  ✅ {reduction:.1f}% >= 45%")
    else:
        print(f"  ❌ {reduction:.1f}% < 45% (skeleton too bloated)")
    print()
    
    # ── Print register() section ──
    print("=== Skeleton: /register ===")
    lines = skeleton.split("\n")
    in_register = False
    for i, line in enumerate(lines):
        if "def register" in line:
            in_register = True
        elif in_register and line and not line.startswith(" ") and not line.startswith("⋮"):
            break
        if in_register:
            print(line)
    
    print()
    all_pass = len(missing) == 0 and reduction >= 45
    if all_pass:
        print("🎉 ALL TESTS PASSED")
    else:
        print(f"⚠️  {len(missing)} missing, reduction={reduction:.1f}%")
    return all_pass


if __name__ == "__main__":
    ok = test_index_py()
    sys.exit(0 if ok else 1)
