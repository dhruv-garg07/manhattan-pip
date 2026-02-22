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
    """Fallback chunker treating the whole file as one chunk or splitting by paragraphs."""
    
    def chunk_file(self, content: str, file_path: str = "") -> List[CodeChunk]:
        # For now, just return the whole file as a single 'file' chunk
        return [self._create_chunk(
            content=content,
            chunk_type="file",
            name=file_path.split("/")[-1] if file_path else "unknown",
            start_line=1,
            end_line=content.count('\n') + 1,
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
