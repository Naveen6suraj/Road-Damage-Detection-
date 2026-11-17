"""
ROADGUARD AI - Clean Submission Package Generator
Script: package_submission.py

Bundles the complete source code, models, static assets, templates, documentation,
and test suite into a clean ZIP archive for academic submission.
Excludes Python bytecode, temporary caches, and large transient files.
"""

import os
import zipfile

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ZIP_OUTPUT = os.path.join(BASE_DIR, "RoadGuard-AI-Final-Submission.zip")

EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".idea",
    ".vscode",
    "runs",
    "static - Copy"
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".part",
    ".tmp",
    ".db-wal",
    ".db-shm"
}


def should_include_file(rel_path: str) -> bool:
    # Check directory exclusions
    parts = rel_path.replace("\\", "/").split("/")
    for p in parts:
        if p in EXCLUDE_DIRS:
            return False

    # Check extension exclusions
    _, ext = os.path.splitext(rel_path)
    if ext.lower() in EXCLUDE_EXTS:
        return False

    # Avoid bundling the output zip itself
    if os.path.basename(rel_path) == os.path.basename(ZIP_OUTPUT):
        return False

    return True


def create_submission_zip():
    print(f"[*] Packaging RoadGuard AI into: {ZIP_OUTPUT}")
    file_count = 0

    with zipfile.ZipFile(ZIP_OUTPUT, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(BASE_DIR):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, BASE_DIR)

                if should_include_file(rel_path):
                    # Ensure top-level archive directory name
                    arc_name = os.path.join("RoadGuard-AI", rel_path)
                    z.write(full_path, arc_name)
                    file_count += 1

    zip_size_mb = os.path.getsize(ZIP_OUTPUT) / (1024 * 1024)
    print(f"[+] Archive generated successfully!")
    print(f"    Total files bundled: {file_count}")
    print(f"    Archive size: {zip_size_mb:.2f} MB")
    print(f"    Location: {ZIP_OUTPUT}")


if __name__ == "__main__":
    create_submission_zip()
