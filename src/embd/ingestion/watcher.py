"""Filesystem watcher that auto-ingests new/changed/deleted documents.

Uses ``watchdog`` to monitor ``documents_dir`` **recursively** for changes
to any file type enabled in ``cfg.ingestion.enabled_types``.  A short
debounce timer prevents double-processing when editors write temporary files
before the final save.

Paths are stored in the vector index as POSIX relative paths from
``documents_dir`` (e.g. ``papers/report.pdf``), matching the recursive
scanner.  Directories and files matching ``cfg.ingestion.ignore_patterns``
are silently skipped.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path, PurePosixPath

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..config import Config
from ..embedding.encoder import Encoder
from ..store.vector_store import VectorStore
from .ingest import ingest_file
from .registry import get_supported_extensions
from .scanner import _is_ignored

logger = logging.getLogger(__name__)

_DEBOUNCE_SECONDS = 2.0


class _IngestHandler(FileSystemEventHandler):
    """React to file-system events by ingesting or deleting documents."""

    def __init__(
        self,
        store: VectorStore,
        encoder: Encoder,
        cfg: Config,
    ) -> None:
        super().__init__()
        self._store = store
        self._encoder = encoder
        self._cfg = cfg
        self._docs_dir = cfg.paths.documents_dir.resolve()
        self._supported_extensions = get_supported_extensions(cfg.ingestion.enabled_types)
        self._ignore_patterns = cfg.ingestion.ignore_patterns
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _rel_key(self, path: Path) -> str | None:
        """Return the POSIX relative path from documents_dir, or None if ignored."""
        try:
            rel = PurePosixPath(path.resolve().relative_to(self._docs_dir))
        except ValueError:
            return None
        if _is_ignored(rel, self._ignore_patterns):
            return None
        return str(rel)

    def _is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in self._supported_extensions

    def _debounced_ingest(self, abs_path: Path, source_key: str, *, delete_first: bool = False) -> None:
        with self._lock:
            existing = self._timers.pop(source_key, None)
            if existing is not None:
                existing.cancel()
            t = threading.Timer(
                _DEBOUNCE_SECONDS,
                self._do_ingest,
                args=(abs_path, source_key, delete_first),
            )
            t.daemon = True
            self._timers[source_key] = t
            t.start()

    def _do_ingest(self, abs_path: Path, source_key: str, delete_first: bool) -> None:
        if not abs_path.exists():
            return
        try:
            result = ingest_file(
                abs_path,
                self._store,
                self._encoder,
                self._cfg,
                delete_existing=delete_first,
                source_name=source_key,
            )
            if result:
                logger.info(
                    "[watcher] Ingested %s → %d chunks (%.1fs)",
                    result.filename,
                    result.n_chunks,
                    result.extract_s + result.embed_s + result.upsert_s,
                )
            else:
                logger.warning("[watcher] %s: no extractable text", source_key)
        except Exception:
            logger.exception("[watcher] Failed to ingest %s", source_key)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        p = Path(event.src_path)
        if not self._is_supported(p):
            return
        key = self._rel_key(p)
        if key:
            logger.info("[watcher] New file detected: %s", key)
            self._debounced_ingest(p, key)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        p = Path(event.src_path)
        if not self._is_supported(p):
            return
        key = self._rel_key(p)
        if key:
            logger.info("[watcher] Modified file: %s", key)
            self._debounced_ingest(p, key, delete_first=True)

    def on_moved(self, event: FileSystemEvent) -> None:
        src = Path(event.src_path)
        dst = Path(event.dest_path)
        if src.suffix.lower() in self._supported_extensions:
            src_key = self._rel_key(src)
            if src_key:
                logger.info("[watcher] File moved away: %s", src_key)
                self._store.delete_file(src_key)
        if dst.suffix.lower() in self._supported_extensions:
            dst_key = self._rel_key(dst)
            if dst_key:
                logger.info("[watcher] File moved in: %s", dst_key)
                self._debounced_ingest(dst, dst_key)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        p = Path(event.src_path)
        if not self._is_supported(p):
            return
        key = self._rel_key(p)
        if key:
            logger.info("[watcher] File deleted: %s — removing chunks", key)
            try:
                self._store.delete_file(key)
            except Exception:
                logger.exception("[watcher] Failed to delete chunks for %s", key)


def start_watcher(
    store: VectorStore,
    encoder: Encoder,
    cfg: Config,
) -> Observer:
    """Start a background observer on ``cfg.paths.documents_dir`` (recursive).

    Returns the running Observer so the caller can ``observer.stop()`` on shutdown.
    """
    watch_dir = cfg.paths.documents_dir
    watch_dir.mkdir(parents=True, exist_ok=True)
    handler = _IngestHandler(store, encoder, cfg)
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=True)
    observer.daemon = True
    observer.start()
    exts = get_supported_extensions(cfg.ingestion.enabled_types)
    logger.info(
        "[watcher] Watching %s (recursive) for %s changes",
        watch_dir, ", ".join(sorted(exts)),
    )
    return observer
