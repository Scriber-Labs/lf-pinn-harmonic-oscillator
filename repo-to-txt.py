from pathlib import Path

# --- config ---
MAX_FILE_SIZE = 200_000  # bytes
INCLUDE_EXTS = {
    ".md", ".py", ".txt", ".yml", ".yaml", ".json", ".toml"
}
EXCLUDE_DIRS = {
    ".git", "__pycache__", ".venv",
    "node_modules", ".ipynb_checkpoints",
    ".egg-info",
}
OUTPUT_FILE = "snapshot.txt"

# --- helpers ---
def is_text_file(path: Path) -> bool:
    return path.suffix in INCLUDE_EXTS and path.stat().st_size < MAX_FILE_SIZE

def print_tree(root: Path, depth=3):
    lines = []
    for path in sorted(root.rglob("*")):
        if any(p in EXCLUDE_DIRS for p in path.parts):
            continue
        rel = path.relative_to(root)
        if len(rel.parts) <= depth:
            indent = "  " * (len(rel.parts) - 1)
            lines.append(f"{indent}{path.name}")
    return "\n".join(lines)

# --- main ---
repo = Path(".").resolve()

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    out.write("# Repository Snapshot\n\n")
    out.write("## Directory Tree\n\n")
    out.write(print_tree(repo))
    out.write("\n\n## Files\n\n")

    for file in sorted(repo.rglob("*")):
        if any(p in EXCLUDE_DIRS for p in file.parts):
            continue
        if file.is_file() and is_text_file(file):
            out.write(f"\n\n=== {file.relative_to(repo)} ===\n\n")
            try:
                out.write(file.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                out.write("[Skipped: encoding error]")

print(f"Saved snapshot to {OUTPUT_FILE}")