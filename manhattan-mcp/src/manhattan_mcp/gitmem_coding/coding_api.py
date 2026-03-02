"""
GitMem Coding - Coding Context API

High-level API for code memory management.
Provides VFS navigation tools + CRUD operations for agent-facing MCP tools.
"""

from typing import Dict, Any, List, Optional
from .coding_store import CodingContextStore
from .coding_vector_store import CodingVectorStore
from .coding_memory_builder import CodingMemoryBuilder
from .coding_hybrid_retriever import CodingHybridRetriever
from .coding_file_system import CodingFileSystem
import os
import json
import hashlib
import time
import re


class CodingAPI:
    """
    High-level API for coding context storage.
    
    VFS Navigation Tools (for MCP):
    1. list_directory(path) — browse indexed files
    2. search_codebase(query) — hybrid semantic+keyword search
    3. get_token_savings() — savings report
    
    CRUD Operations:
    4. index_file(file_path) — index/create
    5. reindex_file(file_path) — re-index/update
    6. remove_index(file_path) — delete
    7. list_indexed_files() — list all
    """
    
    def __init__(self, root_path: str = "./.gitmem_coding"):
        """Initialize the Coding API."""
        self.store = CodingContextStore(root_path=root_path)
        self.vector_store = CodingVectorStore(root_path=root_path)
        self.builder = CodingMemoryBuilder(self.store, self.vector_store)
        self.retriever = CodingHybridRetriever(self.store, self.vector_store)
        self.filesystem = CodingFileSystem(self.store)
        
        # Performance tracking (session-scoped, not persisted)
        self._perf_tracker = {
            "indexing": {"total_ms": 0, "count": 0, "last_ms": 0},
            "retrieval": {"total_ms": 0, "count": 0, "last_ms": 0},
            "search": {"total_ms": 0, "count": 0, "last_ms": 0},
            "embedding": {"total_ms": 0, "count": 0, "last_ms": 0},
        }
    
    def _record_perf(self, op: str, elapsed_ms: float):
        """Record a performance measurement."""
        tracker = self._perf_tracker.get(op)
        if tracker:
            tracker["total_ms"] += elapsed_ms
            tracker["count"] += 1
            tracker["last_ms"] = round(elapsed_ms, 1)
    
    # =========================================================================
    # VFS Navigation Tools (NEW — for agent-facing MCP tools)
    # =========================================================================
    
    # def get_file_outline(self, agent_id: str, file_path: str) -> Dict[str, Any]:
    #     """
    #     Get structural outline of a file — chunk names, types, signatures, line ranges.
    #     No full content. Auto-indexes if not cached or stale.
    #     """
    #     normalized = os.path.normpath(file_path)
    #     
    #     # 1. Check cache via store (includes freshness check)
    #     cached = self.store.retrieve_file_context(agent_id, normalized)
    #     
    #     if cached.get("status") == "cache_miss" or cached.get("freshness") == "stale":
    #         # Auto-index (re-index if stale)
    #         if os.path.exists(normalized):
    #             self.index_file(agent_id, normalized)
    #             # Re-retrieve to get the new chunks
    #             cached = self.store.retrieve_file_context(agent_id, normalized)
    #     
    #     if cached.get("status") != "cache_hit":
    #         return {
    #             "status": "error",
    #             "message": f"File not found and could not be indexed: {normalized}"
    #         }
    #     
    #     # We need the full file context object from the agent data to get the chunks
    #     # retrieve_file_context return 'code_flow' which is a BST/Tree.
    #     # But get_file_outline wants the raw chunks list for a flat overview.
    #     contexts = self.store._load_agent_data(agent_id, "file_contexts")
    #     found = next(
    #         (ctx for ctx in contexts 
    #          if os.path.normpath(ctx.get("file_path", "")) == normalized), 
    #         None
    #     )
    #     
    #     if found is None:
    #          return {
    #             "status": "error",
    #             "message": f"File indexed but not found in store retrieval: {normalized}"
    #         }
    #     
    #     # Extract outline from chunks
    #     chunks = found.get("chunks", [])
    #     outline_items = []
    #     for chunk in chunks:
    #         item = {
    #             "name": chunk.get("name", "unknown"),
    #             "type": chunk.get("type", "unknown"),
    #             "start_line": chunk.get("start_line", 0),
    #             "end_line": chunk.get("end_line", 0),
    #         }
    #         # Include signature (content first line only)
    #         content = chunk.get("content", "")
    #         if content:
    #             first_line = content.split("\n")[0].strip()
    #             item["signature"] = first_line
    #             
    #         outline_items.append(item)
    #     
    #     original_tokens = self._estimate_file_tokens(normalized)
    #     outline_tokens = int(len(json.dumps(outline_items, separators=(',', ':'))) * 0.25)
    #     
    #     # Optimization for worst-case scenarios (many tiny functions)
    #     if original_tokens > 0 and outline_tokens > original_tokens * 0.3:
    #         # 1. Drop signatures, which are usually the longest part
    #         for item in outline_items:
    #             item.pop('signature', None)
    #         outline_tokens = int(len(json.dumps(outline_items, separators=(',', ':'))) * 0.25)
    #         
    #         # 2. If still too large, drop line numbers
    #         if outline_tokens > original_tokens * 0.5:
    #             for item in outline_items:
    #                 item.pop('start_line', None)
    #                 item.pop('end_line', None)
    #             outline_tokens = int(len(json.dumps(outline_items, separators=(',', ':'))) * 0.25)
    #             
    #         # 3. If outline is STILL too large, just group names by type
    #         if outline_tokens > original_tokens * 0.8:
    #             grouped = {}
    #             for item in outline_items:
    #                 t = item.get('type', 'unknown')
    #                 grouped.setdefault(t, []).append(item.get('name', 'unknown'))
    #             
    #             compact_outline = []
    #             for t, names in grouped.items():
    #                 name_str = ", ".join(names)
    #                 if len(name_str) > 800:
    #                     name_str = name_str[:800] + "... (truncated)"
    #                 compact_outline.append({"type": f"grouped {t}s", "names": name_str})
    #             outline_items = compact_outline
    #             outline_tokens = int(len(json.dumps(outline_items, separators=(',', ':'))) * 0.25)
    #     
    #     hint_pct = min(100, max(1, int(outline_tokens / max(1, original_tokens) * 100))) if original_tokens > 0 else 100
    #     
    #     return {
    #         "status": "ok",
    #         "file_path": normalized,
    #         "language": found.get("language", "unknown"),
    #         "total_chunks": len(outline_items),
    #         "outline": outline_items,
    #         "_token_info": {
    #             "tokens_this_call": outline_tokens,
    #             "tokens_if_raw_read": original_tokens,
    #             "tokens_saved": max(0, original_tokens - outline_tokens),
    #             "hint": f"Outline uses ~{hint_pct}% of raw file tokens"
    #         }
    #     }
    
    def list_directory(self, agent_id: str, path: str = "") -> Dict[str, Any]:
        """
        List indexed files through the VFS, organized by language.
        """
        nodes = self.filesystem.list_dir(agent_id, path)
        return {
            "status": "ok",
            "path": path or "/",
            "items": nodes,
            "count": len(nodes)
        }
    
    def get_token_savings(self, agent_id: str) -> Dict[str, Any]:
        """Get the token savings report."""
        return self.store.get_token_savings_report(agent_id)
    
    def _estimate_file_tokens(self, file_path: str) -> int:
        """Estimate tokens for a file on disk."""
        try:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                return int(size * 0.25)  # ~4 chars per token
        except Exception:
            pass
        return 0
    
    # =========================================================================
    # Tier 1 Features: Cross-Reference, Dependency Graph, Delta Update, Stats
    # =========================================================================
    
    # def cross_reference(self, agent_id: str, symbol: str) -> Dict[str, Any]:
    #     """
    #     Find all references to a symbol across indexed files.
    #     Replaces grep_search for symbol usage lookups.
    #     """
    #     # Guard: empty or whitespace-only symbols would match everything
    #     if not symbol or not symbol.strip():
    #         return {
    #             "symbol": symbol,
    #             "total_references": 0,
    #             "files_matched": 0,
    #             "references": [],
    #             "_token_info": {"hint": "Empty symbol — no references to find"}
    #         }
    #     result = self.store.find_symbol_references(agent_id, symbol)
    #     result["_token_info"] = {
    #         "hint": f"Found {result['total_references']} references across {result['files_matched']} files — no grep needed"
    #     }
    #     return result
    
    def dependency_graph(self, agent_id: str, file_paths: List[str], depth: int = 1) -> Dict[str, Any]:
        """
        Build import/dependency graph for a list of files.
        Shows what these files import and what imports them.
        Max 10 files per call.
        """
        if not isinstance(file_paths, list):
            file_paths = [file_paths]
            
        if len(file_paths) > 10:
            return {
                "status": "error",
                "message": "Too many file paths. Please limit to 10 files per call."
            }
            
        contexts = self.store._load_agent_data(agent_id, "file_contexts")
        batch_results = {}
        
        for file_path in file_paths:
            normalized = os.path.normpath(file_path)
            file_basename = os.path.basename(normalized)
            module_name = os.path.splitext(file_basename)[0]
            
            # Get imports FROM this file
            imports_raw = self.store.get_file_imports(agent_id, normalized)
            import_modules = []
            for imp in imports_raw:
                if "module" in imp:
                    import_modules.append(imp["module"])
            
            # Get files that import THIS file
            imported_by = self.store.find_importers(agent_id, module_name)
            
            # Extract cross-file calls from chunks
            calls_to = []
            found = next(
                (ctx for ctx in contexts 
                 if os.path.normpath(ctx.get("file_path", "")) == normalized),
                None
            )
            if found:
                for chunk in found.get("chunks", []):
                    # Use the extracted calls from AST
                    chunk_calls = chunk.get("calls", [])
                    if not chunk_calls:
                        # Fallback for old indices: Regex based
                        if chunk.get("type") in ("method", "function"):
                            content = chunk.get("content", "")
                            ext_calls = re.findall(r'self\.(\w+)\.(\w+)\s*\(', content)
                            for obj_attr, method in ext_calls:
                                chunk_calls.append({
                                    "target": f"{obj_attr}.{method}",
                                    "line": chunk.get("start_line", 0)
                                })
                    
                    for call in chunk_calls:
                        call_entry = {
                            "target": call.get("target"),
                            "from": chunk.get("name", "unknown"),
                            "line": call.get("line", 0),
                            "signature": call.get("signature", ""),
                            "parameters_passed": call.get("parameters_passed", ""),
                            "return_used_as": call.get("return_used_as", "")
                        }
                        calls_to.append(call_entry)
            
            # Inbound calls (called_by)
            called_by = []
            for other_ctx in contexts:
                other_path = other_ctx.get("file_path", "")
                if other_path == normalized: continue
                for other_chunk in other_ctx.get("chunks", []):
                    for call in other_chunk.get("calls", []):
                        target = call.get("target", "")
                        # Try to match target with this file's functions/classes
                        if target == module_name or target.startswith(f"{module_name}."):
                            called_by.append({
                                "caller": other_chunk.get("name", "unknown"),
                                "caller_file": os.path.basename(other_path),
                                "line": call.get("line", 0)
                            })

            # External packages
            external_packages = {}
            # We can use the external_imports collected by chunking_engine
            collected_externals = found.get("external_imports", []) if found else []
            for ext in collected_externals:
                # Mocking version/installed check for now as we don't have a reliable way to check package versions easily without pip
                external_packages[ext] = {"version": "unknown", "installed": True}

            # Unused functions (dead code detection)
            # A simple heuristic: if a function is NOT in called_by and NOT called within the same file (self calls)
            # This is complex to do perfectly without a full symbol resolver, but we can do a best effort.
            unused_functions = []
            if found:
                all_calls_to_this_file = [c["caller"] for c in called_by]
                for chunk in found.get("chunks", []):
                    if chunk.get("type") in ("function", "method"):
                        cname = chunk.get("name", "")
                        # Check if cname is called in any chunk of the SAME file
                        is_called_internally = False
                        for other_chunk in found.get("chunks", []):
                            for call in other_chunk.get("calls", []):
                                if call.get("target") == cname or call.get("target") == f"self.{cname.split('.')[-1]}":
                                    is_called_internally = True
                                    break
                        
                        if not is_called_internally and cname not in all_calls_to_this_file:
                            unused_functions.append({
                                "function": cname,
                                "line": chunk.get("start_line", 0),
                                "reason": "no_external_callers"
                            })

            # Depth > 1: follow transitive imports
            transitive_imports = []
            if depth > 1:
                for mod in import_modules:
                    mod_basename = mod.split(".")[-1]
                    # Try to find this module in indexed files
                    for ctx in contexts:
                        ctx_basename = os.path.splitext(os.path.basename(ctx.get("file_path", "")))[0]
                        if ctx_basename == mod_basename:
                            sub_imports = self.store.get_file_imports(agent_id, ctx.get("file_path", ""))
                            for si in sub_imports:
                                if "module" in si:
                                    transitive_imports.append({
                                        "via": mod_basename,
                                        "module": si["module"]
                                    })
                            break
            
            result = {
                "status": "ok",
                "file": file_basename,
                "file_path": normalized,
                "imports": import_modules,
                "imported_by": [ib["file_path"] for ib in imported_by],
                "calls_to": calls_to,
                "called_by": called_by,
                "external_packages": external_packages,
                "unused_functions": unused_functions,
                "graph_summary": f"{file_basename} depends on {len(import_modules)} modules and is used by {len(imported_by)} modules"
            }
            
            if transitive_imports:
                result["transitive_imports"] = transitive_imports
            
            batch_results[file_path] = result
            
        return {
            "status": "ok",
            "results": batch_results
        }
    
    def delta_update(self, agent_id: str, file_path: str) -> Dict[str, Any]:
        """
        Incrementally re-index a file, only processing changed chunks.
        Compares old vs new chunks by hash_id, skips unchanged, removes stale.
        """
        normalized = os.path.normpath(file_path)
        
        if not os.path.exists(normalized):
            return {"status": "error", "message": f"File not found: {normalized}"}
        
        # 1. Get OLD chunks from cache
        contexts = self.store._load_agent_data(agent_id, "file_contexts")
        old_ctx = next(
            (ctx for ctx in contexts 
             if os.path.normpath(ctx.get("file_path", "")) == normalized),
            None
        )
        old_hashes = {}
        if old_ctx:
            for chunk in old_ctx.get("chunks", []):
                hid = chunk.get("hash_id")
                if hid:
                    old_hashes[hid] = chunk
        
        # 2. Generate NEW chunks from current file
        from .chunking_engine import ChunkingEngine, detect_language
        try:
            with open(normalized, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            lang = detect_language(normalized)
            chunker = ChunkingEngine.get_chunker(lang)
            code_chunks = chunker.chunk_file(content, normalized)
            
            new_chunks = []
            for cc in code_chunks:
                c_dict = cc.to_dict()
                if not c_dict.get("summary"):
                    c_dict["summary"] = f"Code unit: {c_dict.get('name', 'unnamed')}"
                new_chunks.append(c_dict)
        except Exception as e:
            return {"status": "error", "message": f"Chunking failed: {str(e)}"}
        
        # 3. Compute hashes for new chunks (same logic as builder)
        import re as _re
        import hashlib as _hashlib
        new_hashes = {}
        for chunk in new_chunks:
            if not chunk.get("hash_id") and chunk.get("content"):
                norm_content = _re.sub(r'\s+', ' ', chunk["content"]).strip()
                chunk["hash_id"] = _hashlib.sha256(norm_content.encode('utf-8')).hexdigest()
            hid = chunk.get("hash_id")
            if hid:
                new_hashes[hid] = chunk
        
        # 4. Diff: unchanged, added, removed, modified
        old_set = set(old_hashes.keys())
        new_set = set(new_hashes.keys())
        
        unchanged_ids = old_set & new_set
        added_ids = new_set - old_set
        removed_ids = old_set - new_set
        
        # 5. Remove stale vectors
        if removed_ids:
            self.vector_store.delete_vectors(agent_id, list(removed_ids))
        
        # 6. Only embed & store the new/changed chunks
        # We still store ALL chunks for the file context, but builder will skip
        # embedding for chunks with existing vectors (hash_id match)
        result = self.builder.process_file_chunks(
            agent_id=agent_id,
            file_path=normalized,
            chunks=new_chunks,
            language=lang,
        )
        
        delta_info = {
            "status": "delta_applied",
            "file_path": normalized,
            "chunks_added": len(added_ids),
            "chunks_removed": len(removed_ids), 
            "chunks_unchanged": len(unchanged_ids),
            "total_chunks": len(new_chunks)
        }
        return delta_info
    
    def cache_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics with per-file freshness and recommendations.
        Enhanced replacement for get_token_savings.
        """
        return self.store.get_detailed_cache_stats(agent_id)
    
    # =========================================================================
    # Tier 2 Features: Invalidation, Summaries, Snapshots, Analytics, Perf
    # =========================================================================
    
    def invalidate_cache(
        self, agent_id: str, file_path: str = None, scope: str = "file"
    ) -> Dict[str, Any]:
        """
        Explicitly invalidate cache entries. scope: 'file', 'stale', or 'all'.
        """
        if scope == "file" and file_path:
            result = self.store.invalidate_with_vectors(agent_id, file_path)
            if result["removed"] and result["hash_ids"]:
                self.vector_store.delete_vectors(agent_id, result["hash_ids"])
            return {
                "status": "invalidated" if result["removed"] else "not_found",
                "scope": "file",
                "file_path": file_path,
                "vectors_cleaned": len(result.get("hash_ids", [])),
            }
        
        elif scope == "stale":
            result = self.store.invalidate_stale(agent_id)
            if result["hash_ids"]:
                self.vector_store.delete_vectors(agent_id, result["hash_ids"])
            return {
                "status": "invalidated",
                "scope": "stale",
                "invalidated": result["invalidated"],
                "files": result["files"],
                "vectors_cleaned": len(result.get("hash_ids", [])),
            }
        
        elif scope == "all":
            contexts = self.store._load_agent_data(agent_id, "file_contexts")
            all_hash_ids = []
            for ctx in contexts:
                all_hash_ids.extend(
                    ch.get("hash_id") for ch in ctx.get("chunks", []) if ch.get("hash_id")
                )
            if all_hash_ids:
                self.vector_store.delete_vectors(agent_id, all_hash_ids)
            count = len(contexts)
            self.store._save_agent_data(agent_id, "file_contexts", [])
            # Clear global index
            self.store._global_index = {}
            self.store._save_global_index()
            return {
                "status": "invalidated",
                "scope": "all",
                "invalidated": count,
                "vectors_cleaned": len(all_hash_ids),
            }
        
        return {"status": "error", "message": f"Invalid scope '{scope}'. Use 'file', 'stale', or 'all'."}
    
    def summarize_context(
        self, agent_id: str, file_path: str, verbosity: str = "brief"
    ) -> Dict[str, Any]:
        """
        Return a file's context at configurable verbosity levels.
        brief (~50 tokens), normal (code_flow), detailed (full chunks).
        """
        normalized = os.path.normpath(file_path)
        
        # Ensure indexed
        cached = self.store.retrieve_file_context(agent_id, normalized)
        if cached.get("status") == "cache_miss" or cached.get("freshness") == "stale":
            if os.path.exists(normalized):
                self.index_file(agent_id, normalized)
                cached = self.store.retrieve_file_context(agent_id, normalized)
        
        if cached.get("status") != "cache_hit":
            return {"status": "error", "message": f"Could not retrieve context: {normalized}"}
        
        # Get full context data
        contexts = self.store._load_agent_data(agent_id, "file_contexts")
        found = next(
            (ctx for ctx in contexts
             if os.path.normpath(ctx.get("file_path", "")) == normalized),
            None
        )
        if not found:
            return {"status": "error", "message": f"Context not found in store: {normalized}"}
        
        chunks = found.get("chunks", [])
        file_name = os.path.basename(normalized)
        language = found.get("language", "unknown")
        
        if verbosity == "brief":
            # Ultra-compact: file + language + chunk type counts + key names
            type_counts = {}
            key_names = []
            for ch in chunks:
                ct = ch.get("type", "block")
                type_counts[ct] = type_counts.get(ct, 0) + 1
                if ct in ("class", "function") and ch.get("name"):
                    key_names.append(ch["name"])
            
            type_summary = ", ".join(f"{v} {k}s" for k, v in type_counts.items())
            names_str = ", ".join(key_names[:8])
            
            return {
                "status": "ok",
                "verbosity": "brief",
                "file": file_name,
                "language": language,
                "summary": f"{file_name} ({language}): {len(chunks)} chunks — {type_summary}. Key: {names_str}"
            }
        
        elif verbosity == "detailed":
            # Full chunks with content and summaries
            detailed_chunks = []
            for ch in chunks:
                detailed_chunks.append({
                    "name": ch.get("name", ""),
                    "type": ch.get("type", ""),
                    "content": ch.get("content", ""),
                    "summary": ch.get("summary", ""),
                    "keywords": ch.get("keywords", []),
                })
            return {
                "status": "ok",
                "verbosity": "detailed",
                "file": file_name,
                "language": language,
                "chunks": detailed_chunks,
                "total_chunks": len(detailed_chunks),
            }
        
        else:  # "normal" — default: return code_flow tree
            return {
                "status": "ok",
                "verbosity": "normal",
                "file": file_name,
                "language": language,
                "code_flow": cached.get("code_flow", {}),
                "freshness": cached.get("freshness", "unknown"),
            }
    
    def create_snapshot(self, agent_id: str, message: str = "Snapshot") -> Dict[str, Any]:
        """
        Create an immutable snapshot of all cached contexts.
        Wraps CodingFileSystem.commit_snapshot.
        """
        sha = self.filesystem.commit_snapshot(agent_id, message)
        if sha:
            return {
                "status": "ok",
                "sha": sha,
                "message": message,
            }
        return {
            "status": "error",
            "message": "Snapshot failed. Is .gitmem initialized? (Requires gitmem DAG backend)",
        }
    
    def compare_snapshots(
        self, agent_id: str, sha_a: str, sha_b: str
    ) -> Dict[str, Any]:
        """
        Compare two snapshots and return the diff.
        Wraps CodingFileSystem.get_diff.
        """
        diff = self.filesystem.get_diff(agent_id, sha_a, sha_b)
        if diff is not None:
            return {
                "status": "ok",
                "sha_a": sha_a,
                "sha_b": sha_b,
                "diff": diff,
            }
        return {
            "status": "error",
            "message": f"Could not compare snapshots {sha_a} vs {sha_b}. Check SHAs or .gitmem availability.",
        }
    
    def usage_report(self, agent_id: str) -> Dict[str, Any]:
        """
        Get aggregate usage analytics: access counts, tokens, indexing stats.
        """
        return self.store.get_usage_report(agent_id)
    
    def performance_profile(self, agent_id: str) -> Dict[str, Any]:
        """
        Get performance timing data for key operations (session-scoped).
        """
        profile = {}
        for op, data in self._perf_tracker.items():
            count = data["count"]
            profile[op] = {
                "avg_ms": round(data["total_ms"] / count, 1) if count > 0 else 0,
                "count": count
            }
        return {"status": "ok", "profile": profile}

    def usage_analysis(self, agent_id: str, file_path: str) -> Dict[str, Any]:
        """
        Analyze code usage within a file: defined vs used functions, unused params.
        """
        normalized = os.path.normpath(file_path)
        contexts = self.store._load_agent_data(agent_id, "file_contexts")
        found = next((ctx for ctx in contexts if os.path.normpath(ctx.get("file_path", "")) == normalized), None)
        
        if not found:
            return {"status": "error", "message": f"File not indexed: {file_path}"}
            
        functions_defined = []
        functions_used_externally = []
        functions_unused = []
        parameters_unused = []
        
        # 1. Identify all defined functions/methods
        for chunk in found.get("chunks", []):
            if chunk.get("type") in ("function", "method"):
                functions_defined.append({
                    "name": chunk.get("name"),
                    "line": chunk.get("start_line")
                })
                
        # 2. Check external usage (called_by)
        module_name = os.path.splitext(os.path.basename(normalized))[0]
        for other_ctx in contexts:
            if os.path.normpath(other_ctx.get("file_path", "")) == normalized: continue
            for other_chunk in other_ctx.get("chunks", []):
                for call in other_chunk.get("calls", []):
                    target = call.get("target", "")
                    if target == module_name or target.startswith(f"{module_name}."):
                        func_name = target.split('.')[-1] if '.' in target else target
                        if func_name not in functions_used_externally:
                            functions_used_externally.append(func_name)
                            
        # 3. Identify unused functions (simplistic check)
        used_internal = set()
        for chunk in found.get("chunks", []):
            for call in chunk.get("calls", []):
                target = call.get("target", "")
                if target.startswith("self."):
                    used_internal.add(target.split('.')[-1])
                elif target in [f["name"] for f in functions_defined]:
                    used_internal.add(target)
                    
        for f in functions_defined:
            name = f["name"].split('.')[-1] if '.' in f["name"] else f["name"]
            if name not in functions_used_externally and name not in used_internal:
                functions_unused.append(f)
                
        # 4. Parameters Unused (very simplistic AST-based check)
        # Check if param name appears in function body content (excluding signature)
        for chunk in found.get("chunks", []):
            if chunk.get("type") in ("function", "method"):
                params = chunk.get("parameters", [])
                full_content = chunk.get("content", "").lower()
                
                # Numbered content format: "line:1 def foo(x): line:2   return x"
                # We want to skip the signature part. 
                # A simple way: find the first line number after the start line.
                start_match = re.search(fr'line:{chunk.get("start_line")}\s', full_content)
                if start_match:
                    next_line = chunk.get("start_line") + 1
                    body_match = re.search(fr'line:{next_line}\s', full_content)
                    if body_match:
                        body_content = full_content[body_match.start():]
                    else:
                        # Only one line (the signature itself, e.g. one-liner)
                        body_content = ""
                else:
                    body_content = full_content

                for p in params:
                    pname = p["name"].lower()
                    if pname in ("self", "cls"): continue
                    # check if pname is a word in body
                    if not re.search(fr'\b{pname}\b', body_content):
                        parameters_unused.append({
                            "function": chunk.get("name"),
                            "param": pname,
                            "line": chunk.get("start_line")
                        })
                        
        return {
            "functions_defined": functions_defined,
            "functions_used_externally": list(functions_used_externally),
            "functions_unused": functions_unused,
            "parameters_unused": parameters_unused
        }

    def circular_dependency_check(self, agent_id: str) -> Dict[str, Any]:
        """
        Check for circular dependencies in the indexed codebase.
        """
        contexts = self.store._load_agent_data(agent_id, "file_contexts")
        graph = {} # module -> list of imported modules
        
        for ctx in contexts:
            path = ctx.get("file_path", "")
            mod_name = os.path.splitext(os.path.basename(path))[0]
            imports = self.store.get_file_imports(agent_id, path)
            graph[mod_name] = [imp["module"].split('.')[0] for imp in imports]
            
        cycles = []
        def find_cycles(node, visited, stack, path):
            visited.add(node)
            stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    find_cycles(neighbor, visited, stack, path)
                elif neighbor in stack:
                    # Cycle found
                    cycle_start_idx = path.index(neighbor)
                    cycles.append(path[cycle_start_idx:] + [neighbor])
            
            stack.remove(node)
            path.pop()

        visited = set()
        for node in graph:
            if node not in visited:
                find_cycles(node, visited, set(), [])
                
        return {
            "has_circular_deps": len(cycles) > 0,
            "cycles_found": cycles,
            "graph_type": "DAG" if not cycles else "DCG",
            "module_count": len(graph)
        }
    
    # =========================================================================
    # CRUD Operations (renamed for agent-facing tools)
    # =========================================================================
    
    
    def index_file(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Index a file (create Code Mem). Alias: create_mem.
        If chunks are not provided, falls back to local AST parsing.
        """
        return self._ingest_file(agent_id, file_path, chunks)

    def reindex_file(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Re-index a file (update Code Mem). Alias: update_mem.
        If chunks are not provided, falls back to local AST parsing.
        """
        return self._ingest_file(agent_id, file_path, chunks)

    def search_codebase(self, agent_id: str, query: str, top_k: int = 5, file_paths: List[str] = None, trace_calls: bool = False, group_by: str = None) -> Dict[str, Any]:
        """
        Search the indexed codebase with hybrid semantic + keyword search.
        Alias: get_mem.
        """
        start_t = time.perf_counter()
        results = self.retriever.search(agent_id, query, top_k=top_k, file_paths_filter=file_paths)
        self._record_perf("search", (time.perf_counter() - start_t) * 1000)
        
        # Call tracing
        if trace_calls:
            call_chain = []
            for res in results.get("results", []):
                chunk = res.get("chunk", {})
                file_path = res.get("file_path", "")
                fname = os.path.basename(file_path)
                cname = chunk.get("name", "unknown")
                line = chunk.get("start_line", 0)
                call_chain.append(f"{fname}:{cname} [line {line}]")
                
                # Trace one level of calls from this chunk
                for call in chunk.get("calls", []):
                    call_chain.append(f"  → {call.get('target')} [line {call.get('line')}]")
            results["call_chain"] = call_chain
            
        # Semantic grouping
        if group_by == "semantic_purpose":
            grouped = {
                "storage_operations": [],
                "search_operations": [],
                "encoding_operations": [],
                "llm_interactions": [],
                "other": []
            }
            
            purpose_map = {
                "storage": ["storage", "store", "save", "db", "database", "insert", "update", "delete", "filesystem"],
                "search": ["search", "find", "query", "lookup", "retrieval", "retriever"],
                "encoding": ["encoding", "encode", "embed", "vector", "token", "hash"],
                "llm": ["llm", "chat", "model", "prompt", "completion", "generate"]
            }
            
            for res in results.get("results", []):
                chunk = res.get("chunk", {})
                content = (chunk.get("content", "") + " " + chunk.get("summary", "")).lower()
                
                found_group = False
                for group, keywords in purpose_map.items():
                    if any(kw in content for kw in keywords):
                        if group == "storage": grouped["storage_operations"].append(res)
                        elif group == "search": grouped["search_operations"].append(res)
                        elif group == "encoding": grouped["encoding_operations"].append(res)
                        elif group == "llm": grouped["llm_interactions"].append(res)
                        found_group = True
                        break
                
                if not found_group:
                    grouped["other"].append(res)
            
            results["grouped_results"] = grouped

        return results

    def remove_index(self, agent_id: str, file_path: str) -> bool:
        """Remove a file's index. Alias: delete_mem."""
        return self.store.delete_code_mem(agent_id, file_path)

    def list_indexed_files(self, agent_id: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List all indexed files. Alias: list_mems."""
        return self.store.list_code_mems(agent_id, limit, offset)

    # =========================================================================
    # Backward-compatible aliases
    # =========================================================================
    
    def create_mem(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Backward-compatible alias for index_file."""
        return self.index_file(agent_id, file_path, chunks)
    
    def get_mem(self, agent_id: str, query: str, top_k: int = 5, file_paths: List[str] = None) -> Dict[str, Any]:
        """Backward-compatible alias for search_codebase."""
        return self.search_codebase(agent_id, query, top_k, file_paths)
    
    def update_mem(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Backward-compatible alias for reindex_file."""
        return self.reindex_file(agent_id, file_path, chunks)
    
    def delete_mem(self, agent_id: str, file_path: str) -> bool:
        """Backward-compatible alias for remove_index."""
        return self.remove_index(agent_id, file_path)
    
    def list_mems(self, agent_id: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Backward-compatible alias for list_indexed_files."""
        return self.list_indexed_files(agent_id, limit, offset)
    
    # =========================================================================
    # Internal helpers
    # =========================================================================
    
    def _ingest_file(self, agent_id: str, file_path: str, chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Shared logic for index_file and reindex_file."""
        from .chunking_engine import detect_language
        
        # Detect language for VFS categorization
        lang = detect_language(file_path)
        
        if chunks is None:
            # Fallback to local AST parsing
            if not os.path.exists(file_path):
                return {"status": "error", "message": f"File not found for auto-chunking: {file_path}"}
            
            from .chunking_engine import ChunkingEngine
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                chunker = ChunkingEngine.get_chunker(lang)
                code_chunks = chunker.chunk_file(content, file_path)
                
                # Convert CodeChunk objects to dicts
                chunks = []
                for cc in code_chunks:
                    c_dict = cc.to_dict()
                    if not c_dict.get("summary"):
                        c_dict["summary"] = f"Code unit: {c_dict.get('name', 'unnamed')}"
                    chunks.append(c_dict)
                    
            except Exception as e:
                return {"status": "error", "message": f"Auto-chunking failed: {str(e)}"}
            
        start_t = time.perf_counter()
        result = self.builder.process_file_chunks(
            agent_id=agent_id,
            file_path=file_path,
            chunks=chunks,
            language=lang,
        )
        self._record_perf("indexing", (time.perf_counter() - start_t) * 1000)
        return result
