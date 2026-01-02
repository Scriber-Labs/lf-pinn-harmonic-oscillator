from pathlib import Path
from typing import Iterable

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_FILE = "snapshot.txt"

MAX_FILE_SIZE = 200_000  # bytes

INCLUDE_EXTS = {
    ".md",
    ".py",
    ".txt",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
}

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    ".ipynb_checkpoints",
    ".egg-info",
}

# Optional: notebook summaries you want injected
NOTEBOOK_SUMMARIES = {
    "notebooks/demo.ipynb": (
        "Notebook demonstrating training of the low-fidelity PINN, "
        "with visualizations of learned trajectory x(t) and energy conservation."
    )
}

REPO_INTENT = """\
This repository explores low-fidelity physics-informed neural networks (PINNs)
for the 1-D harmonic oscillator, emphasizing interpretability and physical
structure over numerical accuracy.
"""


# =============================================================================
# Helpers
# =============================================================================

def is_text_file(path: Path) -> bool:
    return (
        path.suffix in INCLUDE_EXTS
        and path.stat().st_size < MAX_FILE_SIZE
    )


def should_exclude(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def print_tree(root: Path, depth: int = 3) -> str:
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if should_exclude(path):
            continue
        rel = path.relative_to(root)
        if len(rel.parts) <= depth:
            indent = "  " * (len(rel.parts) - 1)
            lines.append(f"{indent}{path.name}")
    return "\n".join(lines)


def write_section(out, title: str) -> None:
    out.write(f"\n## {title}\n\n")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    repo = Path(".").resolve()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        # ---------------------------------------------------------------------
        # Header
        # ---------------------------------------------------------------------
        out.write("# Repository Snapshot\n\n")

        write_section(out, "Repo Intent (author-provided)")
        out.write(REPO_INTENT.strip() + "\n\n")

        # ---------------------------------------------------------------------
        # Directory tree
        # ---------------------------------------------------------------------
        write_section(out, "Directory Tree")
        out.write(print_tree(repo))
        out.write("\n")

        # ---------------------------------------------------------------------
        # Files
        # ---------------------------------------------------------------------
        write_section(out, "Files")

        for file in sorted(repo.rglob("*")):
            if should_exclude(file):
                continue

            if not file.is_file():
                continue

            # Skip the output snapshot itself
            if file.name == OUTPUT_FILE:
                continue

            rel_path = file.relative_to(repo)

            # Notebook handling (summary instead of JSON spam)
            if file.suffix == ".ipynb":
                summary = NOTEBOOK_SUMMARIES.get(
                    str(rel_path),
                    "Jupyter notebook (summary not provided)."
                )
                out.write(f"\n=== {rel_path} (summary) ===\n\n")
                out.write(summary + "\n")
                continue

            # Regular text files
            if is_text_file(file):
                out.write(f"\n=== {rel_path} ===\n\n")
                try:
                    out.write(file.read_text(encoding="utf-8"))
                except UnicodeDecodeError:
                    out.write("[Skipped: encoding error]\n")

    print(f"Saved repository snapshot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
