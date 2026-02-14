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
            
            # Helper to extract source segment
            def get_segment(node) -> str:
                # ast lines are 1-indexed
                start = node.lineno - 1
                end = node.end_lineno # inclusive
                return "\n".join(lines[start:end])

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk_content = get_segment(node)
                    chunks.append(self._create_chunk(
                        content=chunk_content,
                        chunk_type="function",
                        name=node.name,
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        language="python"
                    ))
                elif isinstance(node, ast.ClassDef):
                    chunk_content = get_segment(node)
                    chunks.append(self._create_chunk(
                        content=chunk_content,
                        chunk_type="class",
                        name=node.name,
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        language="python"
                    ))
                    # Chunk methods inside the class
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_content = get_segment(item)
                            chunks.append(self._create_chunk(
                                content=method_content,
                                chunk_type="method",
                                name=f"{node.name}.{item.name}",
                                start_line=item.lineno,
                                end_line=item.end_lineno,
                                language="python"
                            ))
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Group imports? For now just take individual
                    chunk_content = get_segment(node)
                    chunks.append(self._create_chunk(
                        content=chunk_content,
                        chunk_type="import",
                        name="imports",
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        language="python"
                    ))
                elif isinstance(node, ast.Assign):
                    # Capture top-level assignments (constants/config)
                    # Only if simple (target is Name)
                    if any(isinstance(t, ast.Name) for t in node.targets):
                        chunk_content = get_segment(node)
                        chunks.append(self._create_chunk(
                            content=chunk_content,
                            chunk_type="assignment",
                            name="constant",
                            start_line=node.lineno,
                            end_line=node.end_lineno,
                            language="python"
                        ))
                
                # We can also capture imports as a block?
                
            # If no structural chunks found (e.g. script), fallback to file
            if not chunks:
                return TextChunker().chunk_file(content, file_path)
                
            return chunks

        except SyntaxError:
            # Fallback if parse fails
            return TextChunker().chunk_file(content, file_path)
