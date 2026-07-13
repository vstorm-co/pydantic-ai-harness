"""Memory stores: path-addressed persistence backends for the `Memory` capability."""

from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

import anyio.to_thread

_VALID_SEGMENT_RE = re.compile(r'[A-Za-z0-9_.-]{1,200}')
_T = TypeVar('_T')


def validate_store_path(path: str) -> None:
    r"""Reject path strings that could escape a store's root directory.

    Every segment must match `[A-Za-z0-9_.-]{1,200}` and must not contain `..`;
    `/` is only valid as a separator. Empty paths, absolute paths, `\`, and
    traversal sequences are all rejected. Called by `FileStore` on every access
    as defense in depth -- the toolset already validates slugs and scope
    segments before a path string is ever built.
    """
    if not all(_VALID_SEGMENT_RE.fullmatch(segment) and '..' not in segment for segment in path.split('/')):
        raise ValueError(f'invalid memory path: {path!r}')


@runtime_checkable
class MemoryStore(Protocol):
    """Async, path-addressed storage for agent memories.

    Paths are relative POSIX-style strings such as `'main/MEMORY.md'`.

    Contract: `read` returns `None` ONLY when the path does not exist -- access
    and IO failures must RAISE, so callers can distinguish "no memory yet" from
    "storage broken" instead of silently reporting an empty memory. `delete` is
    idempotent (deleting a missing path is a no-op).
    """

    async def read(self, path: str) -> str | None:
        """Return the content at `path`, or `None` if it does not exist."""
        ...  # pragma: no cover

    async def write(self, path: str, content: str) -> None:
        """Write `content` at `path`, creating parents as needed."""
        ...  # pragma: no cover

    async def delete(self, path: str) -> None:
        """Delete `path` if it exists (idempotent)."""
        ...  # pragma: no cover

    async def list_paths(self, prefix: str = '') -> list[str]:
        """Return all stored paths starting with `prefix`, sorted."""
        ...  # pragma: no cover


@dataclass
class InMemoryStore:
    """Default store: a process-lifetime dict.

    Memories survive across `Agent.run` calls within one process but NOT across
    restarts. Pass a `FileStore` (or a database-backed implementation of
    `MemoryStore`) to persist across sessions.
    """

    files: dict[str, str] = field(default_factory=dict[str, str])

    async def read(self, path: str) -> str | None:
        """Return the content at `path`, or `None` if it does not exist."""
        return self.files.get(path)

    async def write(self, path: str, content: str) -> None:
        """Write `content` at `path`."""
        self.files[path] = content

    async def delete(self, path: str) -> None:
        """Delete `path` if it exists (idempotent)."""
        self.files.pop(path, None)

    async def list_paths(self, prefix: str = '') -> list[str]:
        """Return all stored paths starting with `prefix`, sorted."""
        return sorted(path for path in self.files if path.startswith(prefix))


class FileStore:
    """Real-disk store for CLI/local/single-host agents.

    Every access validates the path (`validate_store_path`) and additionally
    jails the resolved location inside `directory` after symlink resolution, so
    a bug in any layer above cannot escape the root. The jail defends against
    hostile path *values* (the model is the adversary); it does not defend
    against a concurrent local process rewriting the store tree between check
    and use -- restrict the directory with OS permissions if local actors are
    untrusted (same threat model as the `FileSystem` capability). Writes are atomic
    (`mkstemp` + `os.replace`); blocking IO runs in a worker thread via
    `anyio.to_thread` so capability hooks never stall the event loop.

    Multi-process note: concurrent index updates from separate processes are
    last-write-wins; the index is self-healing so correctness survives, but
    apps needing strict cross-process consistency should use a database-backed
    `MemoryStore` (transactions) instead.
    """

    def __init__(self, directory: str | Path) -> None:
        self._root = Path(directory)

    def _resolve(self, path: str) -> Path:
        validate_store_path(path)
        real_root = Path(os.path.realpath(self._root))
        resolved = Path(os.path.realpath(real_root / path))
        if not resolved.is_relative_to(real_root):
            raise ValueError(f'memory path {path!r} resolves outside the store directory')
        return resolved

    async def read(self, path: str) -> str | None:
        """Return the content at `path`, or `None` if it does not exist."""
        return await anyio.to_thread.run_sync(self._sync_read, path)

    def _sync_read(self, path: str) -> str | None:
        target = self._resolve(path)
        if not target.is_file():
            return None
        return target.read_text(encoding='utf-8')

    async def write(self, path: str, content: str) -> None:
        """Atomically write `content` at `path`, creating parent directories."""
        await anyio.to_thread.run_sync(self._sync_write, path, content)

    def _sync_write(self, path: str, content: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(dir=target.parent, prefix='.memory-tmp-')
        try:
            with os.fdopen(handle, 'w', encoding='utf-8') as tmp_file:
                tmp_file.write(content)
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):  # pragma: no cover - only on a failed replace
                os.unlink(tmp_name)

    async def delete(self, path: str) -> None:
        """Delete `path` if it exists (idempotent)."""
        await anyio.to_thread.run_sync(self._sync_delete, path)

    def _sync_delete(self, path: str) -> None:
        target = self._resolve(path)
        target.unlink(missing_ok=True)

    async def list_paths(self, prefix: str = '') -> list[str]:
        """Return all stored file paths starting with `prefix`, sorted."""
        return await anyio.to_thread.run_sync(self._sync_list_paths, prefix)

    def _sync_list_paths(self, prefix: str) -> list[str]:
        root = Path(os.path.realpath(self._root))
        if not root.is_dir():
            return []
        found: list[str] = []
        for item in root.rglob('*'):
            if item.is_file():
                relative = item.relative_to(root).as_posix()
                if relative.startswith(prefix):
                    found.append(relative)
        return sorted(found)


_MEMORY_SCHEMA = 'CREATE TABLE IF NOT EXISTS memory_files (path TEXT PRIMARY KEY, content TEXT NOT NULL)'


class SqliteMemoryStore:
    """SQLite-backed store: a single file holds every memory across namespaces.

    Pass either `database=` (path; connections opened short-lived per call,
    with WAL enabled) or `connection=` (caller-owned `sqlite3.Connection`).
    `database=':memory:'` is rejected -- every per-call connection would get a
    fresh empty database; use `InMemoryStore` or a caller-owned connection.
    A caller-owned connection **must** be created with
    `check_same_thread=False` -- store methods dispatch SQL onto worker
    threads via `anyio.to_thread`, so the stdlib default raises
    `sqlite3.ProgrammingError` on first use. The `database=` path sets this
    internally; `connection=` cannot, so it is the caller's responsibility.

    Schema: `memory_files(path TEXT PRIMARY KEY, content TEXT NOT NULL)`,
    created lazily on first use (`CREATE TABLE IF NOT EXISTS` is idempotent,
    so the ready-flag race between worker threads is benign). Paths are
    opaque keys -- always bound as parameters, never interpolated.
    """

    def __init__(
        self,
        *,
        database: str | Path | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if (database is None) == (connection is None):
            raise ValueError('provide exactly one of `database=` or `connection=`')
        if database is not None and str(database) in ('', ':memory:'):
            raise ValueError(
                'an in-memory SQLite database does not work with per-call connections -- '
                'use `InMemoryStore`, or pass a caller-owned `connection=`'
            )
        self._database = database
        self._connection = connection
        self._schema_ready = False

    def _run(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run `operation` on a schema-ready connection, committing on success."""
        if self._connection is not None:
            connection = self._connection
        else:
            assert self._database is not None
            connection = sqlite3.connect(self._database, check_same_thread=False)
            connection.execute('PRAGMA journal_mode=WAL')
        try:
            if not self._schema_ready:
                connection.execute(_MEMORY_SCHEMA)
                self._schema_ready = True
            result = operation(connection)
            connection.commit()
            return result
        finally:
            if connection is not self._connection:
                connection.close()

    async def read(self, path: str) -> str | None:
        """Return the content at `path`, or `None` if it does not exist."""
        return await anyio.to_thread.run_sync(self._sync_read, path)

    def _sync_read(self, path: str) -> str | None:
        def op(connection: sqlite3.Connection) -> str | None:
            row = connection.execute('SELECT content FROM memory_files WHERE path = ?', (path,)).fetchone()
            return None if row is None else str(row[0])

        return self._run(op)

    async def write(self, path: str, content: str) -> None:
        """Upsert `content` at `path`."""
        await anyio.to_thread.run_sync(self._sync_write, path, content)

    def _sync_write(self, path: str, content: str) -> None:
        def op(connection: sqlite3.Connection) -> None:
            connection.execute(
                'INSERT INTO memory_files (path, content) VALUES (?, ?) '
                'ON CONFLICT (path) DO UPDATE SET content = excluded.content',
                (path, content),
            )

        self._run(op)

    async def delete(self, path: str) -> None:
        """Delete `path` if it exists (idempotent)."""
        await anyio.to_thread.run_sync(self._sync_delete, path)

    def _sync_delete(self, path: str) -> None:
        def op(connection: sqlite3.Connection) -> None:
            connection.execute('DELETE FROM memory_files WHERE path = ?', (path,))

        self._run(op)

    async def list_paths(self, prefix: str = '') -> list[str]:
        """Return all stored paths starting with `prefix`, sorted."""
        return await anyio.to_thread.run_sync(self._sync_list_paths, prefix)

    def _sync_list_paths(self, prefix: str) -> list[str]:
        def op(connection: sqlite3.Connection) -> list[str]:
            # substr comparison, not LIKE: SQLite LIKE is case-insensitive for
            # ASCII by default, which would leak listings across case-variant
            # namespaces (`Alice/` vs `alice/`); it also treats `%`/`_` as
            # wildcards, and scope segments may legally contain `_`.
            # Plain qmark placeholders: mixing numbered `?1` with a parameter
            # sequence is deprecated on 3.12 and an error on 3.14.
            rows = connection.execute(
                'SELECT path FROM memory_files WHERE substr(path, 1, length(?)) = ? ORDER BY path',
                (prefix, prefix),
            ).fetchall()
            return [str(row[0]) for row in rows]

        return self._run(op)
