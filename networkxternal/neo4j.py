"""Neo4J backend: vertices and relationships carry their attributes as properties, so nothing is repacked.

Community Neo4J serves one database per instance, so a graph is named by its labels rather than by a
database: vertices carry `<name>` and relationships `<name>_EDGE`, which keeps disjoint graphs apart
in one server. Labels cannot be bound as parameters, so the name is quoted once and interpolated.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import batched
from urllib.parse import urlparse

from neo4j import GraphDatabase

from networkxternal.base_api import (
    Attributes,
    AttributeStore,
    BaseDiGraph,
    BaseGraph,
    BaseMultiDiGraph,
    BaseMultiGraph,
    Cursor,
    Role,
    Triple,
)

BATCH = 1_000
"""Larger batches exhaust the server's Java heap long before they pay off in throughput."""


def graph_name(url: str, default: str = "Graph") -> str:
    """The label a connection string names; a backtick would end the quoting the label is written in."""
    parts = [part for part in urlparse(url).path.split("/") if part]
    name = parts[0] if parts else default
    if "`" in name:
        raise ValueError(f"A graph name may not hold a backtick, got {name!r}")
    return name[0].upper() + name[1:]


class Neo4JGraph(BaseGraph):
    """An undirected simple graph stored in Neo4J, as `networkx.Graph` is in RAM."""

    PAGE = BATCH

    def __init__(self, url: str = "bolt://localhost:7687/graph") -> None:
        super().__init__()
        address = urlparse(url)
        self.driver = GraphDatabase.driver(
            f"{address.scheme}://{address.hostname}:{address.port}",
            auth=(address.username, address.password) if address.username else None,
        )
        self.vertex = f"`{graph_name(url)}`"
        self.edge = f"`{graph_name(url).upper()}_EDGE`"
        self._create_indexes()

    def _create_indexes(self) -> None:
        with self.driver.session() as session:
            name = self.vertex.strip("`")
            session.run(f"CREATE INDEX `{name}_id` IF NOT EXISTS FOR (v:{self.vertex}) ON (v.id)")
            session.run(f"CREATE INDEX `{name}_edge` IF NOT EXISTS FOR ()-[e:{self.edge}]-() ON (e.id)")

    def _pattern(self, role: Role) -> str:
        """The match pattern that binds `v` to the given vertex in that role."""
        if role is Role.SOURCE:
            return f"(v:{self.vertex} {{id: key}})-[e:{self.edge}]->(other:{self.vertex})"
        if role is Role.TARGET:
            return f"(other:{self.vertex})-[e:{self.edge}]->(v:{self.vertex} {{id: key}})"
        return f"(v:{self.vertex} {{id: key}})-[e:{self.edge}]-(other:{self.vertex})"

    # region Storage Verbs

    def scan_nodes(self) -> Iterator[int]:
        with self.driver.session() as session:
            result = session.run(f"MATCH (v:{self.vertex}) RETURN v.id AS id ORDER BY id")
            for record in result:
                yield record["id"]

    def has_node(self, node: int) -> bool:
        with self.driver.session() as session:
            found = session.run(f"MATCH (v:{self.vertex} {{id: $id}}) RETURN count(v) > 0 AS found", id=node).single()
        return bool(found["found"])

    def number_of_nodes(self) -> int:
        with self.driver.session() as session:
            return session.run(f"MATCH (v:{self.vertex}) RETURN count(v) AS count").single()["count"]

    def upsert_nodes(self, nodes: Sequence[int]) -> None:
        if not len(nodes):
            return
        with self.driver.session() as session:
            session.run(f"UNWIND $keys AS key MERGE (:{self.vertex} {{id: key}})", keys=list(nodes))

    def drop_nodes(self, nodes: Sequence[int]) -> None:
        if not len(nodes):
            return
        with self.driver.session() as session:
            session.run(f"UNWIND $keys AS key MATCH (v:{self.vertex} {{id: key}}) DETACH DELETE v", keys=list(nodes))

    def scan_edges(self, after: Cursor = None, limit: int | None = None) -> Iterator[Triple]:
        """Walks the relationships in identifier order, one page per query, resumed by the last identifier."""
        remaining = limit
        cursor = after
        while remaining is None or remaining > 0:
            width = self.PAGE if remaining is None else min(self.PAGE, remaining)
            beyond = "" if cursor is None else "WHERE e.id > $after"
            query = f"""
            MATCH (source:{self.vertex})-[e:{self.edge}]->(target:{self.vertex}) {beyond}
            RETURN source.id AS source, target.id AS target, e.id AS edge ORDER BY e.id LIMIT $width
            """
            with self.driver.session() as session:
                page = [
                    (record["source"], record["target"], record["edge"])
                    for record in session.run(query, after=None if cursor is None else cursor[2], width=width)
                ]
            yield from page
            if len(page) < width:
                return
            cursor = page[-1]
            if remaining is not None:
                remaining -= len(page)

    def adjacent_edges(self, nodes: Sequence[int], role: Role) -> Iterator[tuple[int, Triple]]:
        keys = list(dict.fromkeys(int(node) for node in nodes))
        query = f"""
        UNWIND $keys AS key
        MATCH {self._pattern(role)}
        RETURN key AS key, startNode(e).id AS source, endNode(e).id AS target, e.id AS edge
        """
        for page in batched(keys, self.PAGE):
            with self.driver.session() as session:
                found = [
                    (record["key"], (record["source"], record["target"], record["edge"]))
                    for record in session.run(query, keys=list(page))
                ]
            # An undirected match reaches a self-loop from both of its ends.
            yield from dict.fromkeys(found)

    def find_pairs(self, sources: Sequence[int], targets: Sequence[int]) -> Iterator[tuple[int, Triple]]:
        pairs = [[int(source), int(target)] for source, target in zip(sources, targets, strict=True)]
        if not pairs:
            return
        arrow = "->" if self.DIRECTED else "-"
        query = f"""
        UNWIND range(0, size($pairs) - 1) AS position
        MATCH (a:{self.vertex} {{id: $pairs[position][0]}})-[e:{self.edge}]{arrow}(b:{self.vertex})
        WHERE b.id = $pairs[position][1]
        RETURN position AS position, startNode(e).id AS source, endNode(e).id AS target, e.id AS edge
        """
        for page in batched(enumerate(pairs), self.PAGE):
            with self.driver.session() as session:
                found = [
                    (record["position"], (record["source"], record["target"], record["edge"]))
                    for record in session.run(query, pairs=[pair for _, pair in page])
                ]
            offset = page[0][0]
            # A self-loop matches in both directions, so the same edge is reported once per position.
            for position, triple in dict.fromkeys(found):
                yield offset + position, triple

    def degrees(self, nodes: Sequence[int], role: Role) -> list[int]:
        keys = list(nodes)
        if not keys:
            return []
        query = f"""
        UNWIND $keys AS key
        RETURN key AS key, COUNT {{ MATCH {self._pattern(role)} }} AS degree
        """
        with self.driver.session() as session:
            counts = {record["key"]: record["degree"] for record in session.run(query, keys=keys)}
        return [counts.get(key, 0) for key in keys]

    def upsert_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        rows = [
            {"source": int(source), "target": int(target), "edge": int(edge)}
            for source, target, edge in zip(sources, targets, edges, strict=True)
        ]
        if not rows:
            return
        query = f"""
        UNWIND $rows AS row
        MERGE (source:{self.vertex} {{id: row.source}})
        MERGE (target:{self.vertex} {{id: row.target}})
        MERGE (source)-[:{self.edge} {{id: row.edge}}]->(target)
        """
        with self.driver.session() as session:
            session.run(query, rows=rows)

    def drop_edges(self, sources: Sequence[int], targets: Sequence[int], edges: Sequence[int]) -> None:
        keys = list(edges)
        if not keys:
            return
        query = f"UNWIND $keys AS key MATCH ()-[e:{self.edge} {{id: key}}]-() DELETE e"
        with self.driver.session() as session:
            session.run(query, keys=keys)

    def count_edges(self) -> int:
        with self.driver.session() as session:
            return session.run(f"MATCH ()-[e:{self.edge}]->() RETURN count(e) AS count").single()["count"]

    def biggest_edge_id(self) -> int:
        with self.driver.session() as session:
            found = session.run(f"MATCH ()-[e:{self.edge}]->() RETURN max(e.id) AS biggest").single()
        return found["biggest"] or 0

    def read_documents(self, store: AttributeStore, keys: Sequence[int]) -> list[Attributes]:
        """Attributes are properties of the vertex or relationship itself, minus the `id` we key by."""
        keys = list(keys)
        if not keys:
            return []
        if store is AttributeStore.NODES:
            query = f"UNWIND $keys AS key MATCH (v:{self.vertex} {{id: key}}) RETURN key AS key, properties(v) AS held"
        else:
            query = (
                f"UNWIND $keys AS key MATCH ()-[e:{self.edge} {{id: key}}]-() RETURN key AS key, properties(e) AS held"
            )
        with self.driver.session() as session:
            found = {record["key"]: dict(record["held"]) for record in session.run(query, keys=keys)}
        return [{name: value for name, value in found.get(key, {}).items() if name != "id"} for key in keys]

    def merge_documents(self, store: AttributeStore, keys: Sequence[int], entries: Sequence[Attributes]) -> None:
        keys = list(keys)
        if not keys:
            return
        shared = entries[0] if len(entries) == 1 else None
        rows = [
            {"key": int(key), "held": shared if shared is not None else entries[index]}
            for index, key in enumerate(keys)
        ]
        if store is AttributeStore.NODES:
            query = f"UNWIND $rows AS row MATCH (v:{self.vertex} {{id: row.key}}) SET v += row.held"
        else:
            query = f"UNWIND $rows AS row MATCH ()-[e:{self.edge} {{id: row.key}}]-() SET e += row.held"
        with self.driver.session() as session:
            session.run(query, rows=rows)

    def drop_documents(self, store: AttributeStore, keys: Sequence[int]) -> None:
        """A relationship's properties leave with the relationship; a vertex keeps only its `id`."""
        keys = list(keys)
        if not keys or store is AttributeStore.EDGES:
            return
        query = f"UNWIND $keys AS key MATCH (v:{self.vertex} {{id: key}}) SET v = {{id: key}}"
        with self.driver.session() as session:
            session.run(query, keys=keys)

    def clear(self) -> None:
        with self.driver.session() as session:
            session.run(f"MATCH (v:{self.vertex}) CALL (v) {{ DETACH DELETE v }} IN TRANSACTIONS OF {BATCH} ROWS")
        self.next_edge_id = None

    def close(self) -> None:
        self.driver.close()

    # endregion Storage Verbs


class Neo4JDiGraph(Neo4JGraph, BaseDiGraph):
    """A directed simple graph stored in Neo4J, as `networkx.DiGraph` is in RAM."""


class Neo4JMultiGraph(Neo4JGraph, BaseMultiGraph):
    """An undirected multigraph stored in Neo4J, as `networkx.MultiGraph` is in RAM."""


class Neo4JMultiDiGraph(Neo4JGraph, BaseMultiDiGraph):
    """A directed multigraph stored in Neo4J, as `networkx.MultiDiGraph` is in RAM."""
