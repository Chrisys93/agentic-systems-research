"""
tools.py — Tool wrappers for the tool-selection agent.

Each tool is a callable that takes structured arguments and returns a string result.
Tools are registered in TOOL_REGISTRY for the agent to reference by name.

Security model:
  - All shell tools run inside the container against the mounted repo volume only.
  - Command allowlist is enforced — no arbitrary shell execution.
  - File paths are validated to stay within REPO_PATH before any operation.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Path safety — all tools enforce this
# ---------------------------------------------------------------------------

def _safe_path(repo_path: str, relative_or_absolute: str) -> str:
    """Resolve a path and assert it stays within repo_path. Raises ValueError otherwise."""
    repo = Path(repo_path).resolve()
    target = (repo / relative_or_absolute).resolve()
    if not str(target).startswith(str(repo)):
        raise ValueError(f"Path escape attempt: {relative_or_absolute!r} resolves outside repo")
    return str(target)


def _run(cmd: list[str], cwd: str, timeout: int = 30) -> tuple[str, bool]:
    """Run a subprocess, return (stdout+stderr, success)."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + (f"\n[stderr]: {result.stderr}" if result.stderr.strip() else "")
        return output.strip(), result.returncode == 0
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s]", False
    except Exception as e:
        return f"[error]: {e}", False


# ---------------------------------------------------------------------------
# Shell tools
# ---------------------------------------------------------------------------

def tool_grep(repo_path: str, pattern: str, path: str = ".", flags: str = "-rn",
              include: str = "", max_lines: int = 100) -> dict[str, Any]:
    """
    Grep for a pattern within the repo.

    Args:
        repo_path:  Mounted repo root (e.g. /data/repos/myrepo)
        pattern:    Search pattern (regex supported)
        path:       Subdirectory to search (relative to repo_path, default ".")
        flags:      grep flags (restricted to safe subset)
        include:    File glob filter, e.g. "*.py"
        max_lines:  Truncate output after this many lines
    """
    # Allowlist flags — no --exec, no -z, no dangerous flags
    allowed_flags = {"-r", "-n", "-i", "-l", "-c", "-w", "-rn", "-ri", "-rin", "-rni"}
    if flags not in allowed_flags:
        return {"result": f"Disallowed flags: {flags}", "success": False}

    safe_path = _safe_path(repo_path, path)
    cmd = ["grep", flags, pattern, safe_path]
    if include:
        cmd += [f"--include={include}"]

    output, success = _run(cmd, cwd=repo_path)
    lines = output.splitlines()
    truncated = len(lines) > max_lines
    result = "\n".join(lines[:max_lines])
    if truncated:
        result += f"\n[... truncated at {max_lines} lines]"

    return {"result": result, "success": success, "truncated": truncated}


def tool_cat(repo_path: str, file_path: str, start_line: int = 1,
             end_line: int = -1) -> dict[str, Any]:
    """
    Read a file (or a line range) from the repo.

    Args:
        repo_path:  Mounted repo root
        file_path:  Path relative to repo_path
        start_line: First line to include (1-indexed)
        end_line:   Last line to include (-1 = end of file)
    """
    safe_file = _safe_path(repo_path, file_path)
    try:
        with open(safe_file) as f:
            all_lines = f.readlines()

        end = len(all_lines) if end_line == -1 else end_line
        selected = all_lines[start_line - 1:end]
        result = "".join(selected)
        return {"result": result, "success": True, "total_lines": len(all_lines)}
    except Exception as e:
        return {"result": str(e), "success": False}


def tool_find(repo_path: str, name_pattern: str = "", file_type: str = "f",
              path: str = ".") -> dict[str, Any]:
    """
    Find files matching a pattern within the repo.

    Args:
        repo_path:     Mounted repo root
        name_pattern:  Filename glob (e.g. "*.py", "config*")
        file_type:     "f" (files) | "d" (directories)
        path:          Subdirectory to search
    """
    safe_path = _safe_path(repo_path, path)
    cmd = ["find", safe_path, "-type", file_type]
    if name_pattern:
        cmd += ["-name", name_pattern]

    output, success = _run(cmd, cwd=repo_path)
    files = [line.replace(repo_path, "").lstrip("/") for line in output.splitlines() if line]
    return {"result": "\n".join(files), "files": files, "success": success}


def tool_git_log(repo_path: str, file_path: str = "", n: int = 10) -> dict[str, Any]:
    """
    Get recent git log, optionally scoped to a specific file.

    Args:
        repo_path:  Mounted repo root (must be a git repo)
        file_path:  Optional file to scope the log to
        n:          Number of commits to return
    """
    cmd = ["git", "log", f"-{n}", "--oneline", "--no-merges"]
    if file_path:
        safe = _safe_path(repo_path, file_path)
        cmd += ["--", safe]

    output, success = _run(cmd, cwd=repo_path)
    return {"result": output, "success": success}


def tool_git_blame(repo_path: str, file_path: str, start_line: int = 1,
                   end_line: int = 20) -> dict[str, Any]:
    """Git blame for a line range in a file."""
    safe = _safe_path(repo_path, file_path)
    cmd = ["git", "blame", f"-L{start_line},{end_line}", safe]
    output, success = _run(cmd, cwd=repo_path)
    return {"result": output, "success": success}


def tool_stat(repo_path: str, file_path: str) -> dict[str, Any]:
    """
    Get file metadata (size, modification time, permissions).
    Useful for checking staleness before embedding.
    """
    safe = _safe_path(repo_path, file_path)
    cmd = ["stat", "-c", "%n %s %y %A", safe]
    output, success = _run(cmd, cwd=repo_path)
    return {"result": output, "success": success}


# ---------------------------------------------------------------------------
# Semantic / AST tools
# ---------------------------------------------------------------------------

def tool_vector_search(query: str, chroma_host: str, collection_name: str = "codebase",
                       top_k: int = 5, score_threshold: float = 0.3,
                       filter_file: str = "") -> dict[str, Any]:
    """
    Semantic vector search against ChromaDB.

    Args:
        query:            Natural language or code query
        chroma_host:      ChromaDB HTTP host (e.g. http://chromadb:8000)
        collection_name:  Collection to search
        top_k:            Number of results to return
        score_threshold:  Minimum similarity score (0–1)
        filter_file:      Optional: restrict to chunks from this file path
    """
    try:
        import chromadb
        client = chromadb.HttpClient(host=chroma_host.replace("http://", "").split(":")[0],
                                     port=int(chroma_host.split(":")[-1]))
        collection = client.get_collection(collection_name)

        where = {"source_file": {"$eq": filter_file}} if filter_file else None
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            score = 1 - dist  # ChromaDB returns L2 distance; invert for similarity
            if score >= score_threshold:
                chunks.append({
                    "content": doc,
                    "source_file": meta.get("source_file", "unknown"),
                    "start_line": meta.get("start_line"),
                    "end_line": meta.get("end_line"),
                    "chunk_type": meta.get("chunk_type", "text"),
                    "confidence": round(score, 4)
                })

        return {"chunks": chunks, "success": True, "count": len(chunks)}
    except Exception as e:
        return {"chunks": [], "success": False, "error": str(e)}


def tool_ast_parse(repo_path: str, file_path: str) -> dict[str, Any]:
    """
    Parse a source file with tree-sitter and return the symbol table:
    functions, classes, imports, and their line ranges.

    Falls back to a simple regex scan if tree-sitter parsing fails.
    """
    safe = _safe_path(repo_path, file_path)
    try:
        from tree_sitter_language_pack import get_parser
        import re

        ext = Path(safe).suffix.lstrip(".")
        lang_map = {"py": "python", "js": "javascript", "ts": "typescript",
                    "go": "go", "rs": "rust", "java": "java", "cpp": "cpp", "c": "c"}
        lang = lang_map.get(ext)

        with open(safe, "rb") as f:
            source = f.read()

        if lang:
            parser = get_parser(lang)
            tree = parser.parse(source)

            symbols = []
            def _walk(node):
                if node.type in ("function_definition", "class_definition",
                                 "function_declaration", "method_definition"):
                    name_node = node.child_by_field_name("name")
                    name = name_node.text.decode() if name_node else "<anonymous>"
                    symbols.append({
                        "type": node.type,
                        "name": name,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                    })
                for child in node.children:
                    _walk(child)
            _walk(tree.root_node)

            result_str = "\n".join(
                f"{s['type']} `{s['name']}` (lines {s['start_line']}–{s['end_line']})"
                for s in symbols
            )
            return {"symbols": symbols, "result": result_str, "success": True,
                    "method": "tree-sitter"}

    except Exception:
        pass

    # Fallback: regex-based symbol extraction
    try:
        import re
        with open(safe) as f:
            lines = f.readlines()
        symbols = []
        for i, line in enumerate(lines, 1):
            m = re.match(r"^\s*(def|class|function|func)\s+(\w+)", line)
            if m:
                symbols.append({"type": m.group(1), "name": m.group(2), "start_line": i})
        result_str = "\n".join(f"{s['type']} `{s['name']}` (line {s['start_line']})" for s in symbols)
        return {"symbols": symbols, "result": result_str, "success": True, "method": "regex-fallback"}
    except Exception as e:
        return {"symbols": [], "result": str(e), "success": False}


def tool_github_fetch(owner: str, repo: str, file_path: str,
                      ref: str = "main", github_token: str = "") -> dict[str, Any]:
    """
    Fetch a file directly from the GitHub API.
    Useful when the repo is not locally mounted or for fetching a specific commit/branch.

    Args:
        owner:        GitHub org or username
        repo:         Repository name
        file_path:    Path within the repo (e.g. "src/app.py")
        ref:          Branch, tag, or commit SHA
        github_token: Optional PAT for private repos (read from env if not provided)
    """
    import urllib.request
    import base64, json

    token = github_token or os.environ.get("GITHUB_TOKEN", "")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={ref}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return {"result": content, "success": True, "sha": data.get("sha")}
    except Exception as e:
        return {"result": str(e), "success": False}


# ---------------------------------------------------------------------------
# Tool registry — maps name → callable + metadata for the agent
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "grep": {
        "fn": tool_grep,
        "description": "Search for patterns across the codebase. Best for: locating definitions, finding usages of a symbol, tracing imports.",
        "required_args": ["repo_path", "pattern"],
        "optional_args": ["path", "flags", "include", "max_lines"],
    },
    "cat": {
        "fn": tool_cat,
        "description": "Read a specific file or line range. Best for: fetching a known file, reading a specific function after grep located it.",
        "required_args": ["repo_path", "file_path"],
        "optional_args": ["start_line", "end_line"],
    },
    "find": {
        "fn": tool_find,
        "description": "Find files by name pattern. Best for: discovering all config files, finding test files, locating entry points.",
        "required_args": ["repo_path"],
        "optional_args": ["name_pattern", "file_type", "path"],
    },
    "git_log": {
        "fn": tool_git_log,
        "description": "Get recent commit history, optionally for a specific file. Best for: understanding recent changes, finding when something was introduced.",
        "required_args": ["repo_path"],
        "optional_args": ["file_path", "n"],
    },
    "git_blame": {
        "fn": tool_git_blame,
        "description": "Get authorship and commit info for a line range. Best for: understanding who changed what and when.",
        "required_args": ["repo_path", "file_path"],
        "optional_args": ["start_line", "end_line"],
    },
    "stat": {
        "fn": tool_stat,
        "description": "Get file metadata (size, modified time, permissions). Best for: checking if a file is recent before deciding to re-embed.",
        "required_args": ["repo_path", "file_path"],
        "optional_args": [],
    },
    "vector_search": {
        "fn": tool_vector_search,
        "description": "Semantic similarity search over embedded code chunks. Best for: conceptual questions ('how does caching work?'), cross-file relationships, architectural questions.",
        "required_args": ["query", "chroma_host"],
        "optional_args": ["collection_name", "top_k", "score_threshold", "filter_file"],
    },
    "ast_parse": {
        "fn": tool_ast_parse,
        "description": "Parse a file's AST to get a symbol table (functions, classes, line ranges). Best for: understanding a file's structure before fetching specific sections.",
        "required_args": ["repo_path", "file_path"],
        "optional_args": [],
    },
    "github_fetch": {
        "fn": tool_github_fetch,
        "description": "Fetch a file directly from GitHub API. Best for: repos not locally mounted, fetching a specific branch/commit, cross-referencing with upstream.",
        "required_args": ["owner", "repo", "file_path"],
        "optional_args": ["ref", "github_token"],
    },
}


def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a registered tool by name. Returns its result dict."""
    if name not in TOOL_REGISTRY:
        return {"result": f"Unknown tool: {name}", "success": False}
    start = time.time()
    result = TOOL_REGISTRY[name]["fn"](**args)
    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    return result
