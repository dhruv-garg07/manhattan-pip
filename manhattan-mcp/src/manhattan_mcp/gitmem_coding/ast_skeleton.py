"""
GitMem Coding - Code Flow Logic & Indexing

Generates a structured Code Flow Tree (Start, Input, Processing, Decision, Iteration, Output, End)
and a BST-based Symbol Index for efficient O(log n) retrieval.
"""

import ast
import json
import os
from typing import List, Dict, Any, Optional, Tuple

# ─── Data Structures ────────────────────────────────────────────────────────

class FlowNode:
    """Represents a node in the Code Flow Tree."""
    def __init__(
        self,
        node_type: str,  # "start", "input", "processing", "decision", "iteration", "output", "end"
        content: str,
        line_number: int,
        scope_variables: List[str] = None
    ):
        self.id = str(uuid.uuid4())[:8]
        self.type = node_type
        self.content = content
        self.line_number = line_number
        self.children: List['FlowNode'] = []
        self.parent_id: Optional[str] = None
        self.scope_variables = scope_variables or []

    def add_child(self, child: 'FlowNode'):
        child.parent_id = self.id
        self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "line": self.line_number,
            "scope": self.scope_variables,
            "children": [c.to_dict() for c in self.children]
        }

class BSTNode:
    """Node for the Binary Search Tree Index."""
    def __init__(self, key: str, value: Any):
        self.key = key  # Symbol name (variable, function, class)
        self.value = value  # List of FlowNode IDs where this symbol appears
        self.left: Optional['BSTNode'] = None
        self.right: Optional['BSTNode'] = None

class BSTIndex:
    """Binary Search Tree for O(log n) symbol lookup."""
    def __init__(self):
        self.root: Optional[BSTNode] = None

    def insert(self, key: str, node_id: str):
        if not self.root:
            self.root = BSTNode(key, [node_id])
        else:
            self._insert_recursive(self.root, key, node_id)

    def _insert_recursive(self, node: BSTNode, key: str, node_id: str):
        if key < node.key:
            if node.left:
                self._insert_recursive(node.left, key, node_id)
            else:
                node.left = BSTNode(key, [node_id])
        elif key > node.key:
            if node.right:
                self._insert_recursive(node.right, key, node_id)
            else:
                node.right = BSTNode(key, [node_id])
        else:
            # Key exists, append node_id if not present
            if node_id not in node.value:
                node.value.append(node_id)

    def search(self, key: str) -> List[str]:
        """Return list of FlowNode IDs for the given symbol."""
        return self._search_recursive(self.root, key)

    def _search_recursive(self, node: Optional[BSTNode], key: str) -> List[str]:
        if not node:
            return []
        if key == node.key:
            return node.value
        elif key < node.key:
            return self._search_recursive(node.left, key)
        else:
            return self._search_recursive(node.right, key)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize BST to dictionary."""
        return self._node_to_dict(self.root)

    def _node_to_dict(self, node: Optional[BSTNode]) -> Optional[Dict[str, Any]]:
        if not node:
            return None
        return {
            "key": node.key,
            "value": node.value,
            "left": self._node_to_dict(node.left),
            "right": self._node_to_dict(node.right)
        }

# ─── AST Analysis & Generation ──────────────────────────────────────────────

import uuid

class CodeFlowGenerator:
    """
    Parses source code into a Code Flow Tree and builds a BST Index.
    """
    
    def __init__(self):
        self.bst = BSTIndex()
        self.node_map: Dict[str, FlowNode] = {}  # ID -> Node for quick lookup during path reconstruction

    def generate(self, source: str, file_path: str = "") -> Dict[str, Any]:
        """
        Generate the Code Flow structure and Index.
        Returns a dict with 'tree' and 'index'.
        """
        try:
            tree = ast.parse(source)
            lines = source.splitlines()
            
            root_node = FlowNode("start", f"File: {os.path.basename(file_path)}", 1)
            self.node_map[root_node.id] = root_node
            
            # Process module body
            self._process_block(tree.body, root_node, lines)
            
            # Add end node
            end_node = FlowNode("end", "End of File", len(lines))
            root_node.add_child(end_node)
            self.node_map[end_node.id] = end_node
            
            return {
                "tree": root_node.to_dict(),
                "index": self.bst.to_dict(),
                "node_map": {k: v.to_dict() for k, v in self.node_map.items()} # Serialize map for storage? Or just reconstruct?
                # Storing full map might be redundant if we have the tree. 
                # But for O(1) node lookup during retrieval from ID, we need it.
                # For compression, we might re-build it on load or store it flat.
                # Let's return flattened nodes for storage efficiency if needed, 
                # but for now, the user wants the tree structure.
            }
        except Exception as e:
            return {
                "error": str(e),
                "tree": None,
                "index": None
            }

    def _process_block(self, block: List[ast.AST], parent: FlowNode, lines: List[str]):
        """Recursively process a block of AST nodes."""
        for node in block:
            node_type = "processing"
            content = self._get_source_segment(node, lines)
            truncated_content = self._truncate_content(content)
            
            # Determine specific types
            if isinstance(node, (ast.If, ast.Try)):
                node_type = "decision"
            elif isinstance(node, (ast.For, ast.While)):
                node_type = "iteration"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                node_type = "processing" # Definitions are processing steps that define structure
            elif isinstance(node, ast.Return):
                node_type = "output"
            elif self._is_input(node):
                node_type = "input"
            elif self._is_output(node):
                node_type = "output"

            # Create node
            flow_node = FlowNode(node_type, truncated_content, getattr(node, 'lineno', 0))
            self.node_map[flow_node.id] = flow_node
            parent.add_child(flow_node)

            # Index symbols defined/used
            self._index_node_symbols(node, flow_node.id)

            # Recurse for children
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._process_block(node.body, flow_node, lines)
            elif isinstance(node, ast.If):
                # True branch
                if_branch = FlowNode("decision_branch", "True", node.lineno)
                self.node_map[if_branch.id] = if_branch
                flow_node.add_child(if_branch)
                self._process_block(node.body, if_branch, lines)
                # False branch
                if node.orelse:
                    else_branch = FlowNode("decision_branch", "False", node.orelse[0].lineno)
                    self.node_map[else_branch.id] = else_branch
                    flow_node.add_child(else_branch)
                    self._process_block(node.orelse, else_branch, lines)
            elif isinstance(node, (ast.For, ast.While)):
                loop_body = FlowNode("iteration_body", "Loop Body", node.lineno)
                self.node_map[loop_body.id] = loop_body
                flow_node.add_child(loop_body)
                self._process_block(node.body, loop_body, lines)
            elif isinstance(node, ast.Try):
                try_block = FlowNode("processing", "Try Block", node.lineno)
                self.node_map[try_block.id] = try_block
                flow_node.add_child(try_block)
                self._process_block(node.body, try_block, lines)
                for handler in node.handlers:
                    except_block = FlowNode("decision", f"Except {self._get_name(handler.type)}", handler.lineno)
                    self.node_map[except_block.id] = except_block
                    flow_node.add_child(except_block)
                    self._process_block(handler.body, except_block, lines)

    def _index_node_symbols(self, node: ast.AST, node_id: str):
        """Extract and index variable/function names."""
        # Defined names
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.bst.insert(node.name, node_id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.bst.insert(target.id, node_id)
        
        # Used names (simple traversal)
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                 self.bst.insert(child.id, node_id)

    def _get_source_segment(self, node: ast.AST, lines: List[str]) -> str:
        try:
            return "\n".join(lines[node.lineno-1 : node.end_lineno])
        except (AttributeError, IndexError):
            return str(node)

    def _truncate_content(self, content: str, max_length: int = 150) -> str:
        """Keep content concise for ~50% compression."""
        cleaned = " ".join(content.split())
        if len(cleaned) > max_length:
            return cleaned[:max_length] + "..."
        return cleaned

    def _is_input(self, node: ast.AST) -> bool:
        # Heuristic for input
        code = ast.dump(node)
        return "input" in code or "argv" in code or "request" in code

    def _is_output(self, node: ast.AST) -> bool:
        # Heuristic for output
        code = ast.dump(node)
        return "print" in code or "write" in code or "return" in code or "send" in code

    def _get_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        return "Exception"

# ─── Retrieval Helper ───────────────────────────────────────────────────────

def retrieve_path(tree_data: Dict[str, Any], index_data: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """
    Reconstruct path from Root to matching Nodes using the BST Index.
    Returns a list of node chains (one chain per match).
    """
    # 1. Rebuild BST helper
    # (Simplified: we just need to search the dict structure or rebuild object)
    # Using recursive search on dict for simplicity
    
    matches = _search_bst_dict(index_data, query) # Returns list of node_ids
    
    # 2. Flatten tree map to parent pointers
    # We need a map of id -> parent_id and id -> node_content
    # This assumes we have access to the full tree or a flattened map.
    # In `generate`, we returned `node_map`. If we stored that, great. 
    # If not, we re-traverse the tree.
    
    # For now, let's assume we can traverse the tree dict to find paths.
    
    paths = []
    for node_id in matches:
        path = _find_path_to_node(tree_data, node_id)
        if path:
            paths.append(path)
            
    return paths

def _search_bst_dict(node: Optional[Dict[str, Any]], key: str) -> List[str]:
    if not node:
        return []
    if key == node["key"]:
        return node["value"]
    elif key < node["key"]:
        return _search_bst_dict(node["left"], key)
    else:
        return _search_bst_dict(node["right"], key)

def _find_path_to_node(current_node: Dict[str, Any], target_id: str, current_path: List[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
    if current_path is None:
        current_path = []
    
    # Add current node summary to path
    node_summary = {
        "type": current_node["type"],
        "content": current_node["content"],
        "line": current_node["line"]
    }
    new_path = current_path + [node_summary]
    
    if current_node["id"] == target_id:
        return new_path
    
    for child in current_node.get("children", []):
        result = _find_path_to_node(child, target_id, new_path)
        if result:
            return result
    
    return None

def detect_language(file_path: str) -> str:
    """Auto-detect language from file extension."""
    _, ext = os.path.splitext(file_path)
    # Simple mapping
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript", 
        ".html": "html"
    }.get(ext.lower(), "other")
