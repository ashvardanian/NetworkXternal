"""Memgraph backend: the same Bolt protocol and Cypher as Neo4J, with the dialect differences overridden.

Memgraph keeps its working set in RAM and spills to disk, so it answers the same Cypher far faster
while holding the same shape. Three verbs differ: indexes are declared with the older `CREATE INDEX
ON :Label(property)` form, degrees are counted by an aggregation rather than a `COUNT {}` subquery,
and a whole-graph delete has no batching clause.
"""

from __future__ import annotations

from collections.abc import Sequence

from networkxternal.base_api import BaseDiGraph, BaseMultiDiGraph, BaseMultiGraph, Role
from networkxternal.neo4j import Neo4JGraph


class MemgraphGraph(Neo4JGraph):
    """An undirected simple graph stored in Memgraph, as `networkx.Graph` is in RAM."""

    def __init__(self, url: str = "bolt://localhost:7687/graph") -> None:
        super().__init__(url)

    def _create_indexes(self) -> None:
        with self.driver.session() as session:
            session.run(f"CREATE INDEX ON :{self.vertex}(id)")
            session.run(f"CREATE EDGE INDEX ON :{self.edge}(id)")

    def degrees(self, nodes: Sequence[int], role: Role) -> list[int]:
        keys = list(nodes)
        if not keys:
            return []
        query = f"""
        UNWIND $keys AS key
        MATCH {self._pattern(role)}
        RETURN key AS key, count(e) AS degree
        """
        with self.driver.session() as session:
            counts = {record["key"]: record["degree"] for record in session.run(query, keys=keys)}
        return [counts.get(key, 0) for key in keys]

    def clear(self) -> None:
        with self.driver.session() as session:
            session.run(f"MATCH (v:{self.vertex}) DETACH DELETE v")
        self.next_edge_id = None


class MemgraphDiGraph(MemgraphGraph, BaseDiGraph):
    """A directed simple graph stored in Memgraph, as `networkx.DiGraph` is in RAM."""


class MemgraphMultiGraph(MemgraphGraph, BaseMultiGraph):
    """An undirected multigraph stored in Memgraph, as `networkx.MultiGraph` is in RAM."""


class MemgraphMultiDiGraph(MemgraphGraph, BaseMultiDiGraph):
    """A directed multigraph stored in Memgraph, as `networkx.MultiDiGraph` is in RAM."""
