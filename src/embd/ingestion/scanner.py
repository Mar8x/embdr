"""
Recursive folder scanner with SHA-256-based change detection.

Walks ``documents_dir`` recursively, skipping directories and files that
match ``ignore_patterns``.  Files are keyed by their **relative path**
from ``documents_dir`` (e.g. ``papers/report.pdf``, ``my-repo/README.md``),
which lets subdirectories and cloned repos coexist without filename collisions.
"""
from __future__ import annotations

import fnmatch
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)


def hash_file(path: Path, block_size: int = 65536) -> str:
    """Compute the SHA-256 digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def _is_ignored(rel: PurePosixPath, ignore_patterns: list[str]) -> bool:
    """Return True if any component of *rel* matches an ignore pattern."""
    for part in rel.parts:
        for pat in ignore_patterns:
            if fnmatch.fnmatch(part, pat):
                return True
    return False


def _walk_files(
    root: Path,
    supported_extensions: set[str],
    ignore_patterns: list[str],
) -> dict[str, Path]:
    """Recursively collect files, keyed by POSIX relative path from *root*."""
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in supported_extensions:
            continue
        rel = PurePosixPath(path.relative_to(root))
        if _is_ignored(rel, ignore_patterns):
            continue
        found[str(rel)] = path
    return found


@dataclass
class ScanResult:
    to_add: list[Path] = field(default_factory=list)
    to_update: list[Path] = field(default_factory=list)
    to_remove: list[str] = field(default_factory=list)


def scan_documents(
    documents_dir: Path,
    known_files: dict[str, str],
    supported_extensions: set[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> ScanResult:
    """Diff files on disk against the vector store's known state.

    *known_files* is keyed by relative POSIX path (``subdir/file.pdf``) or
    bare filename for backward-compatible flat layouts.

    *ignore_patterns* is a list of fnmatch globs tested against each path
    component (directory name or filename), e.g. ``[".git", "*.pyc"]``.
    """
    if supported_extensions is None:
        supported_extensions = {".pdf", ".epub"}
    if ignore_patterns is None:
        ignore_patterns = []

    on_disk = _walk_files(documents_dir, supported_extensions, ignore_patterns)
    result = ScanResult()

    for rel_key, abs_path in sorted(on_disk.items()):
        current_hash = hash_file(abs_path)
        if rel_key not in known_files:
            logger.info("New file detected: %s", rel_key)
            result.to_add.append(abs_path)
        elif known_files[rel_key] != current_hash:
            logger.info("Changed file detected: %s", rel_key)
            result.to_update.append(abs_path)
        else:
            logger.debug("Unchanged, skipping: %s", rel_key)

    result.to_remove = sorted(key for key in known_files if key not in on_disk)
    for name in result.to_remove:
        logger.info("Deleted file detected: %s", name)

    return result
