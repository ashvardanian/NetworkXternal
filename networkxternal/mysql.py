"""MySQL backend: the same four tables, with per-table files and `INSERT IGNORE` for vertex upserts."""

from __future__ import annotations

from sqlalchemy import text

from networkxternal.base_sql import SQLDiGraph, SQLGraph, SQLMultiDiGraph, SQLMultiGraph


class MySQLGraph(SQLGraph):
    """An undirected simple graph in MySQL, whose InnoDB tables are clustered by the primary key."""

    PAGE = 10_000
    """A statement travels as one packet, and the server's default `max_allowed_packet` is 64 MB."""

    def tune(self) -> None:
        # Zero is a legitimate vertex identifier, so the auto-value substitution has to go.
        with self.engine.begin() as connection:
            connection.execute(text("SET SESSION sql_mode='NO_AUTO_VALUE_ON_ZERO'"))


class MySQLDiGraph(MySQLGraph, SQLDiGraph):
    """A directed simple graph in MySQL."""


class MySQLMultiGraph(MySQLGraph, SQLMultiGraph):
    """An undirected multigraph in MySQL."""


class MySQLMultiDiGraph(MySQLGraph, SQLMultiDiGraph):
    """A directed multigraph in MySQL."""
