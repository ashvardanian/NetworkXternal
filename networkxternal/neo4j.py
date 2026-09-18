"""Neo4J backend: vertices and relationships carry their attributes as properties, so nothing is repacked.

Community Neo4J serves one database per instance, so a graph is named by its labels rather than by a
database: vertices carry `<name>` and relationships `<name>_EDGE`, which keeps disjoint graphs apart
in one server. Labels cannot be bound as parameters, so the name is validated once and interpolated.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from urllib.parse import urlparse

from neo4j import GraphDatabase

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

BATCH = 1_000
"""Larger batches exhaust the server's Java heap long before they pay off in throughput."""


def graph_name(url: str, default: str = "Graph") -> str:
    """The label a connection string names, restricted to the identifiers Cypher accepts unquoted."""
    parts = [part for part in urlparse(url).path.split("/") if part]
    name = parts[0] if parts else default
    if not name.isidentifier():
        raise ValueError(f"A graph name must be a plain identifier, got {name!r}")
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
        self.vertex = graph_name(url)
        self.edge = f"{self.vertex.upper()}_EDGE"
        self._create_indexes()

    def _create_indexes(self) -> None:
        with self.driver.session() as session:
            session.run(f"CREATE INDEX {self.vertex}_id IF NOT EXISTS FOR (v:{self.vertex}) ON (v.id)")
            session.run(f"CREATE INDEX {self.vertex}_edge IF NOT EXISTS FOR ()-[e:{self.edge}]-() ON (e.id)")

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

    def find_edges(self, nodes: Sequence[int], role: Role) -> list[list[Triple]]:
        keys = list(nodes)
        if not keys:
            return []
        query = f"""
        UNWIND $keys AS key
        MATCH {self._pattern(role)}
        RETURN key AS key, startNode(e).id AS source, endNode(e).id AS target, e.id AS edge
        """
        grouped: dict[int, list[Triple]] = {key: [] for key in keys}
        with self.driver.session() as session:
            for record in session.run(query, keys=keys):
                grouped[record["key"]].append((record["source"], record["target"], record["edge"]))
        return [grouped[key] for key in keys]

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

    # endregion Storage Verbs


class Neo4JDiGraph(Neo4JGraph, BaseDiGraph):
    """A directed simple graph stored in Neo4J, as `networkx.DiGraph` is in RAM."""


class Neo4JMultiGraph(Neo4JGraph, BaseMultiGraph):
    """An undirected multigraph stored in Neo4J, as `networkx.MultiGraph` is in RAM."""


class Neo4JMultiDiGraph(Neo4JGraph, BaseMultiDiGraph):
    """A directed multigraph stored in Neo4J, as `networkx.MultiDiGraph` is in RAM."""
