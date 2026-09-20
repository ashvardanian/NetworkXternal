"""The k-truss: every edge whose ends share enough neighbours once the weaker edges are gone."""

from __future__ import annotations

from typing import Any

from algorithms.base import Bounds, EdgeOriented
from networkxternal.base_api import BaseGraph, Triple


def adjacency_of(kept: set[Triple]) -> dict[int, set[int]]:
    """Who each vertex still reaches, from the surviving edges alone."""
    neighbours: dict[int, set[int]] = {}
    for source, target, _ in kept:
        neighbours.setdefault(source, set()).add(target)
        neighbours.setdefault(target, set()).add(source)
    return neighbours


def shared_count(neighbours: dict[int, set[int]], source: int, target: int) -> int:
    """How many neighbours two vertices share, walking the smaller side rather than building the overlap."""
    one, other = neighbours[source], neighbours[target]
    if len(one) > len(other):
        one, other = other, one
    return sum(1 for node in one if node in other)


def weak_edges(kept: set[Triple], support: int) -> set[Triple]:
    """Every surviving edge whose ends share fewer than `support` neighbours."""
    neighbours = adjacency_of(kept)
    return {triple for triple in kept if shared_count(neighbours, triple[0], triple[1]) < support}


class KTruss(EdgeOriented[set[Triple]]):
    """Every edge whose ends share at least `k - 2` neighbours once the weaker edges are gone.

    Peeled a round at a time until nothing falls. This is the one algorithm here holding state per
    edge rather than per vertex, which is why its bound is quoted that way.
    """

    BOUNDS = Bounds(retained_per_vertex=0, working_per_vertex=0, passes="per round", retained_per_edge=176)

    def __init__(self, graph: BaseGraph, *, k: int) -> None:
        super().__init__(graph)
        self.k = k
        """How many neighbours the two ends of a surviving edge must share, less two."""

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """The triangle truss, which is the smallest one that peels anything."""
        return {"k": 3}

    def run(self) -> set[Triple]:
        kept = {triple for triple in self.graph.scan_edges() if triple[0] != triple[1]}
        while True:
            weak = weak_edges(kept, self.k - 2)
            if not weak:
                return kept
            kept -= weak


def k_truss(graph: BaseGraph, k: int) -> set[Triple]:
    """Every edge surviving the peel, as `networkx.k_truss` reports its edges."""
    return KTruss.on(graph, k=k).run()
