"""PostgreSQL backend: the same four tables, with commits relaxed and upserts written as `ON CONFLICT`."""

from __future__ import annotations

from sqlalchemy import text

from networkxternal.base_sql import SQLDiGraph, SQLGraph, SQLMultiDiGraph, SQLMultiGraph


class PostgresGraph(SQLGraph):
    """An undirected simple graph in PostgreSQL, whose B-tree indexes read well and update slowly."""

    PAGE = 20_000
    """Postgres binds at most 65535 parameters per statement, and a row here binds three."""

    def tune(self) -> None:
        # Durability of a benchmark load is worth less than the throughput it costs.
        with self.engine.begin() as connection:
            connection.execute(text("SET synchronous_commit=off"))


class PostgresDiGraph(PostgresGraph, SQLDiGraph):
    """A directed simple graph in PostgreSQL."""


class PostgresMultiGraph(PostgresGraph, SQLMultiGraph):
    """An undirected multigraph in PostgreSQL."""


class PostgresMultiDiGraph(PostgresGraph, SQLMultiDiGraph):
    """A directed multigraph in PostgreSQL."""
