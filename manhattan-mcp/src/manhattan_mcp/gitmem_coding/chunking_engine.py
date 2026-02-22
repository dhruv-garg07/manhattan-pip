"""
GitMem Coding - Semantic Chunking Engine

Splits source code into semantic units (functions, classes, etc.) to enable
granular context retrieval and deduplication.
"""

import ast
import hashlib
import re
import os
from typing import List, Optional, Tuple, Any, Dict
from .models import CodeChunk, FileLanguage, TOKENS_PER_CHAR_RATIO

def detect_language(file_path: str) -> str:
    """Detect language from file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
        ".sh": "shell",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".md": "markdown",
        ".xml": "xml"
    }
    
    return mapping.get(ext, "other")

class ChunkingEngine:
    """Base class for language-specific chunkers."""
    
    @staticmethod
    def get_chunker(language: str) -> 'ChunkingEngine':
        """Factory method to get appropriate chunker."""
        if language == FileLanguage.PYTHON:
            return PythonChunker()
        # Fallback for others
        return TextChunker()

    def chunk_file(self, content: str, file_path: str = "") -> List[CodeChunk]:
        """Split file content into CodeChunks."""
        raise NotImplementedError

    def _create_chunk(self, content: str, chunk_type: str, name: str, 
                      start_line: int, end_line: int, language: str) -> CodeChunk:
        """Helper to create a normalized CodeChunk."""
        # Normalize for consistent hashing (strip heavy whitespace)
        normalized = re.sub(r'\s+', ' ', content).strip()
        hash_id = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        
        return CodeChunk(
            hash_id=hash_id,
            content=content,
            type=chunk_type,
            name=name,
            start_line=start_line,
            end_line=end_line,
            language=language,
            token_count=int(len(content) * TOKENS_PER_CHAR_RATIO)
        )


class TextChunker(ChunkingEngine):
    """Regex-based chunker for non-Python files (C, JS, Go, etc.).
    
    Detects function/method boundaries using common patterns.
    Falls back to splitting every ~80 lines if no patterns found.
    """
    
    # Patterns for function-like definitions across languages
    _FUNC_PATTERNS = [
        # C/C++/Java/Go: type name(args) {
        re.compile(r'^[\w\s\*]+\s+(\w+)\s*\([^)]*\)\s*\{?\s*$', re.MULTILINE),
        # JavaScript/TypeScript: function name(args) / const name = (args) =>
        re.compile(r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)', re.MULTILINE),
        re.compile(r'^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(', re.MULTILINE),
        # Go: func name(args)
        re.compile(r'^\s*func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(', re.MULTILINE),
        # Rust: fn name(args)
        re.compile(r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)', re.MULTILINE),
        # Ruby: def name
        re.compile(r'^\s*def\s+(\w+)', re.MULTILINE),
        # Class declarations
        re.compile(r'^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)', re.MULTILINE),
        # Struct/enum declarations
        re.compile(r'^\s*(?:typedef\s+)?(?:struct|enum)\s+(\w+)', re.MULTILINE),
    ]
    
    def chunk_file(self, content: str, file_path: str = "") -> List[CodeChunk]:
        lines = content.splitlines()
        total_lines = len(lines)
        
        if total_lines < 50:
            # Small files: return as single chunk
            return [self._create_chunk(
                content=content,
                chunk_type="file",
                name=file_path.split("/")[-1] if file_path else "unknown",
                start_line=1,
                end_line=total_lines,
                language="text"
            )]
        
        # Find function/class boundaries
        boundaries = []  # list of (line_number, name, type)
        for pattern in self._FUNC_PATTERNS:
            for match in pattern.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                name = match.group(1) if match.lastindex else "block"
                match_text = match.group(0).strip()
                chunk_type = "class" if "class " in match_text or "struct " in match_text else "function"
                boundaries.append((line_num, name, chunk_type))
        
        # Deduplicate and sort by line number
        boundaries = sorted(set(boundaries), key=lambda x: x[0])
        
        if not boundaries:
            # No patterns found: split every ~80 lines
            chunks = []
            chunk_size = 80
            for start in range(0, total_lines, chunk_size):
                end = min(start + chunk_size, total_lines)
                segment = "\n".join(lines[start:end])
                chunks.append(self._create_chunk(
                    content=segment,
                    chunk_type="block",
                    name=f"block_{start+1}_{end}",
                    start_line=start + 1,
                    end_line=end,
                    language="text"
                ))
            return chunks
        
        # Split file at function boundaries
        chunks = []
        
        # Add header chunk if first boundary isn't at the start
        if boundaries[0][0] > 5:
            header_end = boundaries[0][0] - 1
            header_content = "\n".join(lines[:header_end])
            chunks.append(self._create_chunk(
                content=header_content,
                chunk_type="block",
                name="header",
                start_line=1,
                end_line=header_end,
                language="text"
            ))
        
        # Create chunks between boundaries
        for i, (line_num, name, chunk_type) in enumerate(boundaries):
            if i + 1 < len(boundaries):
                end_line = boundaries[i + 1][0] - 1
            else:
                end_line = total_lines
            
            start_idx = line_num - 1
            end_idx = end_line
            segment = "\n".join(lines[start_idx:end_idx])
            
            chunks.append(self._create_chunk(
                content=segment,
                chunk_type=chunk_type,
                name=name,
                start_line=line_num,
                end_line=end_line,
                language="text"
            ))
        
        return chunks if chunks else [self._create_chunk(
            content=content,
            chunk_type="file",
            name=file_path.split("/")[-1] if file_path else "unknown",
            start_line=1,
            end_line=total_lines,
            language="text"
        )]


class PythonChunker(ChunkingEngine):
    """AST-based chunker for Python code."""
    
    def chunk_file(self, content: str, file_path: str = "") -> List[CodeChunk]:
        chunks = []
        try:
            tree = ast.parse(content)
            lines = content.splitlines()
            
            def get_segment(start_line, end_line) -> str:
                return "\n".join(lines[start_line-1:end_line])

            def process_nodes(nodes, prefix=""):
                buffer = []
                
                def flush():
                    if not buffer:
                        return
                    start = buffer[0].lineno
                    end = max(n.end_lineno for n in buffer)
                    
                    types = set()
                    names = []
                    for n in buffer:
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            types.add("function" if not prefix else "method")
                            names.append(n.name)
                        elif isinstance(n, ast.ClassDef):
                            types.add("class")
                            names.append(n.name)
                        elif isinstance(n, (ast.Import, ast.ImportFrom)):
                            types.add("import")
                        elif isinstance(n, ast.Assign):
                            types.add("assignment")
                    
                    if not types:
                        types.add("block")
                    
                    t_list = list(types)
                    chunk_type = t_list[0] if len(t_list) == 1 else "mixed"
                    if not names:
                        names = ["block"]
                        
                    name = ", ".join(names)
                    if len(name) > 60:
                        name = name[:57] + "..."
                    
                    if prefix:
                        name = f"{prefix}.[{name}]" if len(names) > 1 else f"{prefix}.{name}"
                        
                    chunks.append(self._create_chunk(
                        content=get_segment(start, end),
                        chunk_type=chunk_type,
                        name=name,
                        start_line=start,
                        end_line=end,
                        language="python"
                    ))
                    buffer.clear()

                for node in nodes:
                    if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
                        continue
                        
                    if isinstance(node, ast.ClassDef):
                        flush()
                        chunks.append(self._create_chunk(
                            content=get_segment(node.lineno, node.end_lineno),
                            chunk_type="class",
                            name=node.name,
                            start_line=node.lineno,
                            end_line=node.end_lineno,
                            language="python"
                        ))
                        process_nodes(node.body, prefix=node.name)
                    else:
                        buffer.append(node)
                        current_lines = buffer[-1].end_lineno - buffer[0].lineno + 1
                        if current_lines >= 100:
                            flush()
                            
                flush()

            process_nodes(tree.body)
            
            if not chunks:
                return TextChunker().chunk_file(content, file_path)
                
            return chunks

        except SyntaxError:
            return TextChunker().chunk_file(content, file_path)
