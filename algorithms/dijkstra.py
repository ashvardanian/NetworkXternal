"""Dijkstra's shortest paths, one settled vertex and one adjacency lookup at a time."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Any

from algorithms.base import Bounds, VertexOriented
from algorithms.results import VertexMap, sparsely
from networkxternal.base_api import BaseGraph


class DijkstraLengths(VertexOriented[VertexMap[float]]):
    """The weight of the lightest path from `source` to every vertex it reaches.

    One adjacency lookup per settled vertex, a round trip per vertex — the reference walk, not the
    one for a large graph; `DeltaSteppingLengths` is the one for that.
    """

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=16, passes="per settled vertex")

    def __init__(
        self,
        graph: BaseGraph,
        *,
        source: int,
        weight: str | None = "weight",
        cutoff: float | None = None,
    ) -> None:
        super().__init__(graph)
        self.source = source
        """The vertex every distance is measured from."""

        self.weight = weight
        """The edge attribute the walk relaxes along, or nothing to count every edge as one."""

        self.cutoff = cutoff
        """The distance beyond which a vertex is never settled, or nothing to reach every vertex."""

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """The first vertex the store holds as the source, which the declared bound assumes."""
        return {"source": next(iter(graph.scan_nodes()), 0)}

    def run(self) -> VertexMap[float]:
        settled: dict[int, float] = {}
        frontier = [(0.0, self.source)]
        while frontier:
            held, node = heappop(frontier)
            if node in settled:
                continue
            settled[node] = held
            for _, other, along in self.arcs_of([node], self.graph.outgoing_role, self.weight):
                reached = held + along
                if other not in settled and (self.cutoff is None or reached <= self.cutoff):
                    heappush(frontier, (reached, other))
        return sparsely(settled, "d")


def dijkstra_lengths(
    graph: BaseGraph,
    source: int,
    weight: str | None = "weight",
    cutoff: float | None = None,
) -> VertexMap[float]:
    """The weight of the lightest path from `source` to every vertex it reaches."""
    return DijkstraLengths.on(graph, source=source, weight=weight, cutoff=cutoff).run()
