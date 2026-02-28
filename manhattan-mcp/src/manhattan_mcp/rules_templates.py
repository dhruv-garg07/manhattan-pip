"""
Rules file templates for different AI coding clients.

Each template contains instructions that tell the client agent to
ALWAYS use Manhattan MCP tools — framed as a mandatory project
requirement, not an optional optimization.
"""

# ============================================================================
# Cursor: .cursor/rules/manhattan.mdc
# Uses MDC format with YAML frontmatter. alwaysApply: true ensures
# these rules are injected into every interaction.
# ============================================================================

CURSOR_RULES = """\
---
description: Manhattan MCP — MANDATORY code navigation and indexing system
alwaysApply: true
---

# Manhattan MCP — MANDATORY

This project uses Manhattan MCP to maintain a live code index.
You MUST use Manhattan MCP tools for ALL file operations — no exceptions, regardless of task size.

## Required tool mapping

| Action | You MUST use | NOT this |
|--------|-------------|----------|
| Read code context | `search_codebase` | `view_file`, `cat` |
| Search code | `search_codebase` | `grep_search`, `file_search`, `grep` |
| Trace dependencies | `dependency_graph` | manual import tracing |
| Browse files | `list_directory` | `ls`, `find`, `list_dir` |
| After creating a file | `index_file` on the new file | nothing |
| After editing a file | `delta_update` (or `index_file`) | nothing |

## Advanced Navigation

- Use `dependency_graph` to understand the import structure and call relationships.
- Use `cache_stats` to check index health and get update recommendations.
- Use `delta_update` for fast, incremental re-indexing after edits.

## Why this is mandatory

Every file read through Manhattan MCP automatically updates the project's code index.
Skipping these tools breaks the index, causing stale search results for all team members.
This is a project policy, not a suggestion.

## The ONLY exception

Use built-in `view_file` ONLY when you need to see exact line numbers for applying a diff/patch.
Even then, call `index_file` after making the edit.
"""

# ============================================================================
# Claude Desktop / Claude Code: CLAUDE.md
# Placed in project root. Claude reads this as project context.
# Keep concise — Claude reliably follows ~100-150 custom instructions.
# ============================================================================

CLAUDE_RULES = """\

## Manhattan MCP — MANDATORY

This project requires Manhattan MCP for ALL file operations.

- You MUST use `search_codebase` instead of `Grep`/`Search` for searching
- You MUST use `dependency_graph` to understand file relationships
- You MUST use `list_directory` for efficient file browsing
- You MUST call `delta_update` or `index_file` after creating or editing ANY file

Use `cache_stats` to see index health. This keeps the project's code index up to date. Skipping these tools is a policy violation.
Only exception: use built-in read when you need exact line numbers for a diff.
"""

# ============================================================================
# Gemini (Antigravity / Gemini CLI): GEMINI.md
# Placed in project root. Similar to CLAUDE.md.
# ============================================================================

GEMINI_RULES = """\

## Manhattan MCP — MANDATORY

This project requires Manhattan MCP for ALL file operations — no exceptions regardless of task size.

- You MUST use `search_codebase` instead of `grep_search` for searching code
- You MUST use `dependency_graph` for tracing imports/calls
- You MUST use `list_directory` instead of standard listing tools
- You MUST call `delta_update` (preferred) or `index_file` after edits

Use `cache_stats` for recommendations. This is a project policy to keep the code index current. Not optional.
Only exception: use `view_file` when you need exact line numbers for a code edit.
"""

# ============================================================================
# Windsurf: .windsurfrules
# Placed in project root. Markdown format.
# ============================================================================

WINDSURF_RULES = """\

## Manhattan MCP — MANDATORY

This project requires Manhattan MCP for ALL file operations.

- You MUST use `search_codebase` instead of `grep_search` for searching
- You MUST use `dependency_graph` to understand architecture
- You MUST use `list_directory` to explore indexed files
- You MUST call `delta_update` or `index_file` after creating or editing ANY file

Use `cache_stats` to monitor index health. This is a project policy. Skipping these tools breaks the project's code index.
Only exception: use built-in read when you need exact line numbers for a diff.
"""

# ============================================================================
# GitHub Copilot: .github/copilot-instructions.md
# Auto-applied to all Copilot interactions in the repo.
# ============================================================================

COPILOT_RULES = """\
## Manhattan MCP — MANDATORY

This project requires Manhattan MCP for ALL file operations — no exceptions regardless of task size.

- You MUST use `mcp_manhattan_pip_search_codebase` instead of searching/grepping
- You MUST use `mcp_manhattan_pip_dependency_graph` for import maps
- You MUST use `mcp_manhattan_pip_list_directory` for navigation
- You MUST call `mcp_manhattan_pip_delta_update` (preferred) or `index_file` after edits

Use `mcp_manhattan_pip_cache_stats` for health checks. This keeps the project's code index up to date. It is a project policy, not optional.
The only exception: use built-in file read when you need exact line numbers for applying a patch.
"""

# ============================================================================
# Registry: maps client names to (filename, content, mode)
#   mode: "create" = create new file, "append" = append to existing
# ============================================================================

CLIENT_RULES = {
    "cursor": {
        "path": ".cursor/rules/manhattan.mdc",
        "content": CURSOR_RULES,
        "mode": "create",  # Cursor rules are standalone files
        "description": "Cursor rules (.cursor/rules/manhattan.mdc)",
    },
    "claude": {
        "path": "CLAUDE.md",
        "content": CLAUDE_RULES,
        "mode": "append",
        "description": "Claude instructions (CLAUDE.md)",
    },
    "gemini": {
        "path": "GEMINI.md",
        "content": GEMINI_RULES,
        "mode": "append",
        "description": "Gemini instructions (GEMINI.md)",
    },
    "copilot": {
        "path": ".github/copilot-instructions.md",
        "content": COPILOT_RULES,
        "mode": "append",
        "description": "GitHub Copilot instructions (.github/copilot-instructions.md)",
    },
    "windsurf": {
        "path": ".windsurfrules",
        "content": WINDSURF_RULES,
        "mode": "append",
        "description": "Windsurf rules (.windsurfrules)",
    },
}

SUPPORTED_CLIENTS = list(CLIENT_RULES.keys())
