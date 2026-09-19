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
from itertools import batched
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
    Cursor,
    EdgeLayout,
    NetworkXternalError,
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
    CREATE TABLE IF NOT EXISTS graph_meta (
        graph String, directed UInt8, multigraph UInt8, version UInt64
    ) ENGINE = ReplacingMergeTree(version) ORDER BY graph
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

    LAYOUT = EdgeLayout.MIRRORED
    """An undirected edge is two rows sharing one identifier, one ordered by each end, since there is no index."""

    PAGE = 1 << 20
    """A column store punishes small inserts with too many parts, so pages here are far larger than a row store's."""

    WRITE = 1 << 16
    """How many rows one tombstone or upsert statement carries."""

    QUERY_KEYS = 2_048
    """How many keys one query binds; a parameter travels in the URL, which the server caps at 128 KiB."""

    QUERY_PAIRS = 512
    """How many pairs one query binds, an undirected lookup asking both orientations of each."""

    SHARES_IDENTIFIERS = False
    """No linearizable counter without Keeper, so one writer at a time; two would hand out one identifier."""

    CONCURRENT = False
    """A `clickhouse_connect` client runs one query per session, so a thread needs a client of its own."""

    def __init__(self, url: str = "clickhouse://graph:graph@localhost:8123/graph") -> None:
        super().__init__()
        address = urlparse(url)
        database = address.path.strip("/") or "graph"
        self.client = clickhouse_connect.get_client(
            host=address.hostname or "localhost",
            port=address.port or 8123,
            username=address.username or "default",
            password=address.password or "",
            # Zero means as many threads as the server has, which is where a `FINAL` read spends itself.
            # Asynchronous inserts stay off: the simple-graph path reads its own writes back before it
            # allocates, and an insert that has not landed yet would hand out a second edge for one pair.
            settings={"max_threads": 0},
        )
        if not database.replace("-", "_").isidentifier():
            raise ValueError(f"A database name must be a plain identifier, got {database!r}")
        self.client.command(f"CREATE DATABASE IF NOT EXISTS `{database}`")
        self.client.database = database
        for statement in SCHEMA:
            self.client.command(statement)
        try:
            self._seed_meta()
        except Exception:
            # A refused database leaves no object to close, so its client is released here.
            self.client.close()
            raise

    GRAPH = "graph"
    """The name the meta row is keyed by; one database holds one graph."""

    def _seed_meta(self) -> None:
        """Records the shape this graph was written as, and refuses a database written as another.

        The counter stays in this process, since a versioned append cannot be claimed atomically.
        """
        held = self.client.query(
            "SELECT directed, multigraph FROM graph_meta FINAL WHERE graph = {graph:String}",
            parameters={"graph": self.GRAPH},
        ).result_rows
        if not held:
            row = [[self.GRAPH, int(self.DIRECTED), int(self.MULTIGRAPH), self._version()]]
            self.client.insert("graph_meta", row, column_names=["graph", "directed", "multigraph", "version"])
            return
        directed, multigraph = bool(held[0][0]), bool(held[0][1])
        if (directed, multigraph) != (self.DIRECTED, self.MULTIGRAPH):
            raise NetworkXternalError(
                f"This database was written as directed={directed}, multigraph={multigraph}, "
                f"and is being opened as directed={self.DIRECTED}, multigraph={self.MULTIGRAPH}"
            )

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
        """Tombstones the vertices and every edge they take part in, which costs their degree in memory.

        The incident edges are read whole before the first tombstone is written: a client serves one
        query per session, so writing into an open block stream is refused mid-stream.
        """
        keys = list(dict.fromkeys(nodes))
        if not keys:
            return
        stored = [triple for _, triple in self.adjacent_edges(keys, Role.ANY)]
        for page in batched(stored, self.WRITE):
            self.drop_edges(*zip(*page, strict=True))
        version = self._version()
        self.client.insert(
            "nodes", [[int(key), version, 1] for key in keys], column_names=["node", "version", "is_deleted"]
        )

    def scan_edges(self, after: Cursor = None, limit: int | None = None) -> Iterator[Triple]:
        """Streams whole blocks out of the part ordered by `(source, target, edge)`, never a whole result set."""
        beyond = "" if after is None else "AND (source, target, edge) > {after:Tuple(UInt64, UInt64, UInt64)}"
        taken = "" if limit is None else f"LIMIT {int(limit)}"
        query = f"""
        SELECT source, target, edge FROM edges FINAL
        WHERE is_deleted = 0 {beyond} ORDER BY source, target, edge {taken}
        """
        parameters = {} if after is None else {"after": tuple(after)}
        with self.client.query_row_block_stream(query, parameters=parameters) as stream:
            for block in stream:
                for row in block:
                    yield (row[0], row[1], row[2])

    def adjacent_edges(self, nodes: Sequence[int], role: Role) -> Iterator[tuple[int, Triple]]:
        keys = list(dict.fromkeys(int(node) for node in nodes))
        if not keys:
            return
        loops: set[Triple] = set()
        for table, column in self._tables_for(role):
            query = f"""
            SELECT source, target, edge FROM {table} FINAL
            WHERE {column} IN {{keys:Array(UInt64)}} AND is_deleted = 0 ORDER BY {column}, edge
            """
            for page in batched(keys, self.QUERY_KEYS):
                yield from self._adjacent_page(query, list(page), column, role, loops)

    def _adjacent_page(
        self, query: str, keys: list[int], column: str, role: Role, loops: set[Triple]
    ) -> Iterator[tuple[int, Triple]]:
        """One query's worth of incident edges, streamed block by block."""
        with self.client.query_row_block_stream(query, parameters={"keys": keys}) as stream:
            for block in stream:
                for source, target, edge in block:
                    # A self-loop sits in both tables, so the second one skips what the first reported.
                    if source == target and role is Role.ANY:
                        if (source, target, edge) in loops:
                            continue
                        loops.add((source, target, edge))
                    yield (source if column == "source" else target), (source, target, edge)

    def find_pairs(self, sources: Sequence[int], targets: Sequence[int]) -> Iterator[tuple[int, Triple]]:
        positions: dict[tuple[int, int], list[int]] = {}
        for position, (source, target) in enumerate(zip(sources, targets, strict=True)):
            positions.setdefault(self.canonical_pair(int(source), int(target)), []).append(position)
        if not positions:
            return
        query = """
        SELECT source, target, edge FROM edges FINAL
        WHERE (source, target) IN {pairs:Array(Tuple(UInt64, UInt64))} AND is_deleted = 0
        """
        for page in batched(positions, self.QUERY_PAIRS):
            asked = list(page) if self.DIRECTED else [*page, *((target, source) for source, target in page)]
            for source, target, edge in self.client.query(query, parameters={"pairs": asked}).result_rows:
                for position in positions[self.canonical_pair(source, target)]:
                    yield position, (source, target, edge)

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
            for page in batched(keys, self.QUERY_KEYS):
                for end, degree in self.client.query(query, parameters={"keys": list(page)}).result_rows:
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
        """Tombstones the edge in every orientation, since a part collapses on `(source, target, edge)` alone."""
        triples = list(zip(sources, targets, edges, strict=True))
        if not triples:
            return
        version = self._version()
        rows = [[int(edge), int(source), int(target), version, 1] for source, target, edge in triples]
        if not self.DIRECTED:
            rows += [[int(edge), int(target), int(source), version, 1] for source, target, edge in triples]
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
        found: dict[int, Attributes] = {}
        for page in batched(keys, self.QUERY_KEYS):
            rows = self.client.query(query, parameters={"keys": list(page)}).result_rows
            found.update({key: json.loads(document) for key, document in rows})
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
        # The shape a graph was written as outlives its rows, so the meta table is not truncated.
        for table in ("nodes", "edges", "edges_by_target", "node_attributes", "edge_attributes"):
            self.client.command(f"TRUNCATE TABLE {table}")
        self.forget_edge_ids()

    def close(self) -> None:
        self.client.close()

    # endregion Storage Verbs


class ClickHouseDiGraph(ClickHouseGraph, BaseDiGraph):
    """A directed simple graph stored in ClickHouse, as `networkx.DiGraph` is in RAM."""


class ClickHouseMultiGraph(ClickHouseGraph, BaseMultiGraph):
    """An undirected multigraph stored in ClickHouse, as `networkx.MultiGraph` is in RAM."""


class ClickHouseMultiDiGraph(ClickHouseGraph, BaseMultiDiGraph):
    """A directed multigraph stored in ClickHouse, as `networkx.MultiDiGraph` is in RAM."""
