"""
GitMem Coding - Semantic Chunking Engine

Splits source code into semantic units (functions, classes, etc.) to enable
granular context retrieval and deduplication.
"""

import ast
import hashlib
import re
from typing import List, Optional, Tuple, Any
from .models import CodeChunk, FileLanguage, TOKENS_PER_CHAR_RATIO

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
                    
                    # Optional: Also chunk methods inside the class?
                    # For now, let's keep Class as the unit, or maybe both?
                    # "Refined Strategy": Store Class signature as one chunk + methods as children?
                    # Simple approach first: Just top-level chunks.
                
                # We can also capture imports as a block?
                
            # If no structural chunks found (e.g. script), fallback to file
            if not chunks:
                return TextChunker().chunk_file(content, file_path)
                
            return chunks

        except SyntaxError:
            # Fallback if parse fails
            return TextChunker().chunk_file(content, file_path)
