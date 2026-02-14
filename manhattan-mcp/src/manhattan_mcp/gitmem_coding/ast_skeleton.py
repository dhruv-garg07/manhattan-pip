"""
GitMem Coding - Code Flow Logic & Indexing

Generates a structured Code Flow Tree (Start, Input, Processing, Decision, Iteration, Output, End)
and a BST-based Symbol Index for efficient O(log n) retrieval.
"""

import uuid
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

    def to_dict(self) -> Dict[str, List[str]]:
        """Serialize BST to flat dictionary (Symbol -> NodeIDs)."""
        flat_index = {}
        self._flatten(self.root, flat_index)
        return flat_index

    def _flatten(self, node: Optional[BSTNode], flat_index: Dict[str, List[str]]):
        if not node:
            return
        flat_index[node.key] = node.value
        self._flatten(node.left, flat_index)
        self._flatten(node.right, flat_index)

# ─── Context Tree Builder ──────────────────────────────────────────────────

class ContextTreeBuilder:
    """
    Builds a Code Flow Tree and BST Index from pre-chunked data.
    """
    
    def __init__(self):
        self.bst = BSTIndex()
        self.node_map: Dict[str, FlowNode] = {}

    def build(self, chunks: List[Dict[str, Any]], file_path: str = "") -> Dict[str, Any]:
        """
        Build the Code Flow structure and Index from chunks.
        Returns a dict with 'tree' and 'index'.
        """
        try:
            root_node = FlowNode("start", f"File: {os.path.basename(file_path)}", 1)
            self.node_map[root_node.id] = root_node
            
            # Process chunks into nodes
            for chunk in chunks:
                # Determine type based on chunk data or default to processing
                node_type = "processing"
                content = chunk.get("content", "")
                summary = chunk.get("summary", "")
                display_content = summary if summary else self._truncate_content(content)
                
                line_number = chunk.get("start_line", 0)
                
                # Create node
                flow_node = FlowNode(node_type, display_content, line_number)
                self.node_map[flow_node.id] = flow_node
                root_node.add_child(flow_node)
                
                # Index keywords
                keywords = chunk.get("keywords", [])
                for kw in keywords:
                    self.bst.insert(kw, flow_node.id)
                    
                # Also index the name if present
                name = chunk.get("name")
                if name:
                    self.bst.insert(name, flow_node.id)

            # Add end node
            end_line = chunks[-1].get("end_line", 0) if chunks else 1
            end_node = FlowNode("end", "End of File", end_line)
            root_node.add_child(end_node)
            
            return {
                "tree": root_node.to_dict(),
                "index": self.bst.to_dict()
            }
        except Exception as e:
            return {
                "error": str(e),
                "tree": None,
                "index": None
            }

    def _truncate_content(self, content: str, max_length: int = 500) -> str:
        """Keep content concise but informative."""
        if not content:
            return ""
        
        # Strip internal whitespace but keep structure if it looks like a signature
        lines = content.splitlines()
        first_line = lines[0].strip()
        
        # If it's a docstring or single line, just take it
        if len(lines) == 1:
            return first_line[:max_length]

        # For multi-line, take first line (signature) + a bit of docstring if present
        result = first_line
        if len(lines) > 1 and '"""' in lines[1]:
            result += " " + lines[1].strip()
        
        if len(result) > max_length:
            return result[:max_length] + "..."
        return result


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
