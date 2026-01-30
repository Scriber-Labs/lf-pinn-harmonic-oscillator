#!/usr/bin/env python3
"""
repo-to-txt.py

Create a compact, LLM-friendly snapshot of a repository:
- Depth-limited directory tree
- Prioritized key files
- Truncated or header-only source files
- Config files summarized (not dumped)
- Notebook summaries instead of raw JSON
- Global size guard to avoid context blowup
"""

from pathlib import Path

# =====================
# Configuration
# =====================

OUTPUT_FILE = "snapshot.txt"

TREE_DEPTH = 2

MAX_LINES_PER_FILE = 150
MAX_OUTPUT_BYTES = 250_000

CODE_EXTS = {".py", ".md", ".txt"}
CONFIG_EXTS = {".json", ".yml", ".yaml", ".toml"}

IMPORTANT_FILES = {
    "README.md",
    "main.py",
    "app.py",
    "__init__.py",
}

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    ".ipynb_checkpoints",
    ".egg-info",
    "build",
    "dist",
}

EXCLUDE_FILES = {
    "snapshot.txt",
    "prompt.txt",
    "repo_snapshot.txt",
}

REPO_INTENT_FILE = "REPO_INTENT.txt"

NOTEBOOK_SUMMARIES = {
    # "notebooks/demo.ipynb": "Notebook demonstrating training loop and visualization."
}

PY_HEADER_ONLY = True  # extract imports / defs / classes only

# =====================
# Helpers
# =====================


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def iter_files(root: Path, max_depth: int):
    for path in root.rglob("*"):
        if is_excluded(path):
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) <= max_depth:
            yield path


def load_repo_intent() -> str:
    path = Path(REPO_INTENT_FILE)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def print_tree(root: Path, depth: int) -> str:
    lines = []
    for path in sorted(iter_files(root, depth)):
        rel = path.relative_to(root)
        indent = "  " * (len(rel.parts) - 1)
        lines.append(f"{indent}{path.name}")
    return "\n".join(lines)


def extract_py_headers(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith(("import ", "from ", "class ", "def ")):
            lines.append(line)
    return "\n".join(lines)


def write_section(out, title: str):
    out.write(f"## {title}\n\n")


# =====================
# Main
# =====================


def main():
    repo = Path(".").resolve()
    written_bytes = 0

    def safe_write(out, text: str) -> bool:
        nonlocal written_bytes
        b = len(text.encode("utf-8"))
        if written_bytes + b > MAX_OUTPUT_BYTES:
            out.write("\n\n[Snapshot truncated: size limit reached]\n")
            return False
        out.write(text)
        written_bytes += b
        return True

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        safe_write(out, "# Repository Snapshot\n\n")

        write_section(out, "Repo Intent (author-provided)")
        intent = load_repo_intent()
        safe_write(out, (intent if intent else "[No repo intent provided]") + "\n\n")

        write_section(out, "Directory Tree")
        safe_write(out, print_tree(repo, TREE_DEPTH) + "\n\n")

        write_section(out, "Files")

        files = sorted(iter_files(repo, TREE_DEPTH))
        files = sorted(files, key=lambda p: p.name not in IMPORTANT_FILES)

        for file in files:
            if not file.is_file():
                continue
            if file.name in EXCLUDE_FILES:
                continue

            rel = file.relative_to(repo)

            # Notebook handling
            if file.suffix == ".ipynb":
                summary = NOTEBOOK_SUMMARIES.get(
                    str(rel),
                    "Jupyter notebook (content omitted; JSON format).",
                )
                if not safe_write(
                    out, f"\n\n=== {rel} (summary) ===\n\n{summary}\n"
                ):
                    break
                continue

            # Config files (summarize only)
            if file.suffix in CONFIG_EXTS:
                if not safe_write(
                    out,
                    f"\n\n=== {rel} (config summary) ===\n\n"
                    f"{file.suffix} configuration file; content omitted.\n",
                ):
                    break
                continue

            # Code / text files
            if file.suffix in CODE_EXTS:
                if not safe_write(out, f"\n\n=== {rel} ===\n\n"):
                    break
                try:
                    text = file.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    if not safe_write(out, "[Skipped: encoding error]\n"):
                        break
                    continue

                if file.suffix == ".py" and PY_HEADER_ONLY:
                    headers = extract_py_headers(text)
                    content = headers if headers else "[No top-level definitions]"
                else:
                    lines = text.splitlines()
                    if len(lines) > MAX_LINES_PER_FILE:
                        content = (
                            "\n".join(lines[:MAX_LINES_PER_FILE])
                            + "\n\n[... truncated ...]"
                        )
                    else:
                        content = text

                if not safe_write(out, content):
                    break

    print(f"Saved snapshot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
