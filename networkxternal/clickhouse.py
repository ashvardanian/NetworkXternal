"""ClickHouse backend: append-only parts merged by identity, with a mirrored table for reverse lookups.

ClickHouse has no synchronous update or delete, so every write is an `INSERT` carrying a rising
version, and a removal is a tombstone row that `ReplacingMergeTree` collapses at merge time. Reads
resolve the surviving row with `FINAL`. Degrees, counts and histograms are pure aggregations, which
is where a column store beats every row store here by an order of magnitude.

There is no secondary index for the reverse direction either, so edges are written once more into a
table ordered by target, and a lookup picks whichever table its role wants.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from time import time_ns
from urllib.parse import urlparse

import clickhouse_connect

from networkxternal.base_api import (
    Attributes,
    AttributeStore,
    BaseDiGraph,
    BaseGraph,
    BaseMultiDiGraph,
    BaseMultiGraph,
    Role,
    Triple,
)

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS nodes (
        node UInt64, version UInt64, is_deleted UInt8 DEFAULT 0
    ) ENGINE = ReplacingMergeTree(version, is_deleted) ORDER BY node
    """,
    """
    CREATE TABLE IF NOT EXISTS edges (
        edge UInt64, source UInt64, target UInt64, version UInt64, is_deleted UInt8 DEFAULT 0
    ) ENGINE = ReplacingMergeTree(version, is_deleted) ORDER BY (source, target, edge)
    """,
    """
    CREATE TABLE IF NOT EXISTS edges_by_target (
        edge UInt64, source UInt64, target UInt64, version UInt64, is_deleted UInt8 DEFAULT 0
    ) ENGINE = ReplacingMergeTree(version, is_deleted) ORDER BY (target, source, edge)
    """,
    """
    CREATE MATERIALIZED VIEW IF NOT EXISTS edges_mirrored TO edges_by_target AS SELECT * FROM edges
    """,
    """
    CREATE TABLE IF NOT EXISTS node_attributes (
        node UInt64, document String, version UInt64, is_deleted UInt8 DEFAULT 0
    ) ENGINE = ReplacingMergeTree(version, is_deleted) ORDER BY node
    """,
    """
    CREATE TABLE IF NOT EXISTS edge_attributes (
        edge UInt64, document String, version UInt64, is_deleted UInt8 DEFAULT 0
    ) ENGINE = ReplacingMergeTree(version, is_deleted) ORDER BY edge
    """,
)
"""The tables a graph lives in; every one collapses to the newest row per key at merge time."""


class ClickHouseGraph(BaseGraph):
    """An undirected simple graph stored in ClickHouse, as `networkx.Graph` is in RAM."""

    PAGE = 1 << 20
    """A column store punishes small inserts with too many parts, so pages here are far larger than a row store's."""

    def __init__(self, url: str = "clickhouse://graph:graph@localhost:8123/graph") -> None:
        super().__init__()
        address = urlparse(url)
        database = address.path.strip("/") or "graph"
        self.client = clickhouse_connect.get_client(
            host=address.hostname or "localhost",
            port=address.port or 8123,
            username=address.username or "default",
            password=address.password or "",
        )
        if not database.replace("-", "_").isidentifier():
            raise ValueError(f"A database name must be a plain identifier, got {database!r}")
        self.client.command(f"CREATE DATABASE IF NOT EXISTS `{database}`")
        self.client.database = database
        for statement in SCHEMA:
            self.client.command(statement)

    @staticmethod
    def _version() -> int:
        """The version a write carries, so the newest row of a key wins however the parts merge."""
        return time_ns()

    def _attributes_table(self, store: AttributeStore) -> tuple[str, str]:
        if store is AttributeStore.NODES:
            return "node_attributes", "node"
        return "edge_attributes", "edge"

    # region Storage Verbs

    def scan_nodes(self) -> Iterator[int]:
        query = "SELECT node FROM nodes FINAL WHERE is_deleted = 0 ORDER BY node"
        with self.client.query_row_block_stream(query) as stream:
            for block in stream:
                for row in block:
                    yield row[0]

    def has_node(self, node: int) -> bool:
        found = self.client.query(
            "SELECT count() FROM nodes FINAL WHERE node = {node:UInt64} AND is_deleted = 0",
            parameters={"node": node},
        )
        return bool(found.result_rows[0][0])

    def number_of_nodes(self) -> int:
        return self.client.query("SELECT count() FROM nodes FINAL WHERE is_deleted = 0").result_rows[0][0]

    def upsert_nodes(self, nodes: Sequence[int]) -> None:
        keys = list(dict.fromkeys(nodes))
        if not keys:
            return
        version = self._version()
        self.client.insert(
            "nodes", [[int(key), version, 0] for key in keys], column_names=["node", "version", "is_deleted"]
        )

    def drop_nodes(self, nodes: Sequence[int]) -> None:
        keys = list(dict.fromkeys(nodes))
        if not keys:
            return
        stored = [triple for triples in self.find_edges(keys, Role.ANY) for triple in triples]
        if stored:
            self.drop_edges(*zip(*stored, strict=True))
        version = self._version()
        self.client.insert(
            "nodes", [[int(key), version, 1] for key in keys], column_names=["node", "version", "is_deleted"]
        )

    def find_edges(self, nodes: Sequence[int], role: Role) -> list[list[Triple]]:
        keys = list(nodes)
        if not keys:
            return []
        grouped: dict[int, list[Triple]] = {key: [] for key in keys}
        for table, column in self._tables_for(role):
            query = f"""
            SELECT source, target, edge FROM {table} FINAL
            WHERE {column} IN {{keys:Array(UInt64)}} AND is_deleted = 0
            """
            for source, target, edge in self.client.query(query, parameters={"keys": keys}).result_rows:
                end = source if column == "source" else target
                if end in grouped and (source, target, edge) not in grouped[end]:
                    grouped[end].append((source, target, edge))
        return [grouped[key] for key in keys]

    def _tables_for(self, role: Role) -> tuple[tuple[str, str], ...]:
        """Which table answers a lookup in that role: the one ordered by the end being looked up."""
        if role is Role.SOURCE:
            return (("edges", "source"),)
        if role is Role.TARGET:
            return (("edges_by_target", "target"),)
        return ("edges", "source"), ("edges_by_target", "target")

    def degrees(self, nodes: Sequence[int], role: Role) -> list[int]:
        keys = list(nodes)
        if not keys:
            return []
        counts = dict.fromkeys(keys, 0)
        for table, column in self._tables_for(role):
            query = f"""
            SELECT {column} AS end, count() AS degree FROM {table} FINAL
            WHERE {column} IN {{keys:Array(UInt64)}} AND is_deleted = 0
            GROUP BY end
            """
            for end, degree in self.client.query(query, parameters={"keys": keys}).result_rows:
                if end in counts:
                    counts[end] += degree
        return [counts[key] for key in keys]

    def upsert_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        triples = list(zip(sources, targets, edges, strict=True))
        if not triples:
            return
        version = self._version()
        rows = [[int(edge), int(source), int(target), version, 0] for source, target, edge in triples]
        self.client.insert("edges", rows, column_names=["edge", "source", "target", "version", "is_deleted"])
        self.upsert_nodes([end for source, target, _ in triples for end in (source, target)])

    def drop_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        triples = list(zip(sources, targets, edges, strict=True))
        if not triples:
            return
        version = self._version()
        rows = [[int(edge), int(source), int(target), version, 1] for source, target, edge in triples]
        self.client.insert("edges", rows, column_names=["edge", "source", "target", "version", "is_deleted"])

    def count_edges(self) -> int:
        return self.client.query("SELECT count() FROM edges FINAL WHERE is_deleted = 0").result_rows[0][0]

    def biggest_edge_id(self) -> int:
        found = self.client.query("SELECT max(edge) FROM edges FINAL WHERE is_deleted = 0").result_rows[0][0]
        return found or 0

    def read_documents(self, store: AttributeStore, keys: Sequence[int]) -> list[Attributes]:
        keys = list(keys)
        if not keys:
            return []
        table, column = self._attributes_table(store)
        query = f"""
        SELECT {column}, document FROM {table} FINAL
        WHERE {column} IN {{keys:Array(UInt64)}} AND is_deleted = 0
        """
        found = {
            key: json.loads(document)
            for key, document in self.client.query(query, parameters={"keys": keys}).result_rows
        }
        return [found.get(key, {}) for key in keys]

    def merge_documents(self, store: AttributeStore, keys: Sequence[int], entries: Sequence[Attributes]) -> None:
        keys = list(keys)
        if not keys:
            return
        table, column = self._attributes_table(store)
        stored = self.read_documents(store, keys)
        merged: dict[int, Attributes] = {}
        for index, (key, held) in enumerate(zip(keys, stored, strict=True)):
            entry = entries[0] if len(entries) == 1 else entries[index]
            merged[int(key)] = {**merged.get(int(key), held), **entry}
        version = self._version()
        rows = [[key, json.dumps(entry), version, 0] for key, entry in merged.items()]
        self.client.insert(table, rows, column_names=[column, "document", "version", "is_deleted"])

    def drop_documents(self, store: AttributeStore, keys: Sequence[int]) -> None:
        keys = list(dict.fromkeys(keys))
        if not keys:
            return
        table, column = self._attributes_table(store)
        version = self._version()
        rows = [[int(key), "{}", version, 1] for key in keys]
        self.client.insert(table, rows, column_names=[column, "document", "version", "is_deleted"])

    def clear(self) -> None:
        """Truncation is the one removal a column store does cheaply, so clearing skips the tombstones."""
        for table in ("nodes", "edges", "edges_by_target", "node_attributes", "edge_attributes"):
            self.client.command(f"TRUNCATE TABLE {table}")
        self.next_edge_id = None

    def close(self) -> None:
        self.client.close()

    # endregion Storage Verbs


class ClickHouseDiGraph(ClickHouseGraph, BaseDiGraph):
    """A directed simple graph stored in ClickHouse, as `networkx.DiGraph` is in RAM."""


class ClickHouseMultiGraph(ClickHouseGraph, BaseMultiGraph):
    """An undirected multigraph stored in ClickHouse, as `networkx.MultiGraph` is in RAM."""


class ClickHouseMultiDiGraph(ClickHouseGraph, BaseMultiDiGraph):
    """A directed multigraph stored in ClickHouse, as `networkx.MultiDiGraph` is in RAM."""
