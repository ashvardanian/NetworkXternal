"""Memgraph backend: the same Bolt protocol and Cypher as Neo4J, with the dialect differences overridden.

Memgraph keeps its working set in RAM and spills to disk, so it answers the same Cypher far faster
while holding the same shape. Three verbs differ: indexes are declared with the older `CREATE INDEX
ON :Label(property)` form, degrees are counted by an aggregation rather than a `COUNT {}` subquery,
and a whole-graph delete has no batching clause.
"""

from __future__ import annotations

from dataclasses import replace

from networkxternal.base_api import BaseDiGraph, BaseMultiDiGraph, BaseMultiGraph, Role
from networkxternal.neo4j import PATTERNS, CypherText, Neo4JGraph, Queries


class MemgraphGraph(Neo4JGraph):
    """An undirected simple graph stored in Memgraph, as `networkx.Graph` is in RAM."""

    def __init__(self, url: str = "bolt://localhost:7687/graph") -> None:
        super().__init__(url)

    def _create_indexes(self) -> None:
        """The older declaration form, which is the one Memgraph takes."""
        with self.driver.session() as session:
            session.run(f"CREATE INDEX ON :{self.vertex}(id)")
            session.run(f"CREATE EDGE INDEX ON :{self.edge}(id)")

    def _build_queries(self) -> Queries:
        """The same statements, with the three Memgraph spells differently rendered over them."""
        labels = {"vertex": self.vertex, "edge": self.edge, "meta": self.meta}
        held = super()._build_queries()
        patterns = {role: CypherText(body).substitute(labels) for role, body in PATTERNS.items()}
        # Memgraph has no `COUNT {}` subquery, and an undirected pattern finds a self-loop once.
        degrees = {
            role: (
                "UNWIND $keys AS key MATCH "
                + pattern
                + " RETURN key AS key, count(e)"
                + (" + count(CASE WHEN startNode(e) = endNode(e) THEN 1 END)" if role is Role.ANY else "")
                + " AS degree"
            )
            for role, pattern in patterns.items()
        }
        # Nor a batched delete clause, so the whole graph goes in one transaction.
        clear = CypherText("MATCH (v:%vertex) DETACH DELETE v").substitute(labels)
        return replace(held, degrees=degrees, clear=clear)

    def clear_storage(self) -> None:
        with self.driver.session() as session:
            session.run(self.queries.clear)


class MemgraphDiGraph(MemgraphGraph, BaseDiGraph):
    """A directed simple graph stored in Memgraph, as `networkx.DiGraph` is in RAM."""


class MemgraphMultiGraph(MemgraphGraph, BaseMultiGraph):
    """An undirected multigraph stored in Memgraph, as `networkx.MultiGraph` is in RAM."""


class MemgraphMultiDiGraph(MemgraphGraph, BaseMultiDiGraph):
    """A directed multigraph stored in Memgraph, as `networkx.MultiDiGraph` is in RAM."""
