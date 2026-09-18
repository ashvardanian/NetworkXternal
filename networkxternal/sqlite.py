"""SQLite backend: the same four tables, with the page, journal and synchronization settings tuned for bulk writes."""

from __future__ import annotations

from sqlalchemy import text

from networkxternal.base_sql import SQLDiGraph, SQLGraph, SQLMultiDiGraph, SQLMultiGraph

PRAGMAS = (
    "PRAGMA page_size=4096",
    "PRAGMA cache_size=10000",
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA mmap_size=268435456",
)
"""Applied once per connection; a `PRAGMA` SQLite cannot honour is a silent no-op rather than an error."""


class SQLiteGraph(SQLGraph):
    """An undirected simple graph in a single SQLite file, which beats every server below a gigabyte."""

    PAGE = 10_000
    """SQLite compiles at most 32766 bound parameters into one statement, and a row here binds three."""

    def tune(self) -> None:
        with self.engine.begin() as connection:
            for pragma in PRAGMAS:
                connection.execute(text(pragma))


class SQLiteDiGraph(SQLiteGraph, SQLDiGraph):
    """A directed simple graph in a single SQLite file."""


class SQLiteMultiGraph(SQLiteGraph, SQLMultiGraph):
    """An undirected multigraph in a single SQLite file."""


class SQLiteMultiDiGraph(SQLiteGraph, SQLMultiDiGraph):
    """A directed multigraph in a single SQLite file."""
