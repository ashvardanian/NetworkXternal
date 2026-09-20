"""Delta-stepping shortest paths, relaxed a bucket of vertices at a time."""

from __future__ import annotations

from math import inf
from typing import Any

from algorithms.base import Bounds, VertexOriented
from algorithms.results import VertexMap, sparsely
from networkxternal.base_api import BaseGraph


class DeltaSteppingLengths(VertexOriented[VertexMap[float]]):
    """The same answer as `DijkstraLengths`, relaxed a whole bucket of vertices in one round trip.

    Relaxations inside one bucket are independent, which is the shape a parallel run would take.
    """

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=16, passes="per bucket")

    def __init__(
        self,
        graph: BaseGraph,
        *,
        source: int,
        weight: str | None = "weight",
        delta: float = 1.0,
        cutoff: float | None = None,
    ) -> None:
        super().__init__(graph)
        self.source = source
        """The vertex every distance is measured from."""

        self.weight = weight
        """The edge attribute the walk relaxes along, or nothing to count every edge as one."""

        self.delta = delta
        """The bucket width the distances are grouped into."""

        self.cutoff = cutoff
        """The distance beyond which a vertex is never settled, or nothing to reach every vertex."""

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """The first vertex the store holds as the source, which the declared bound assumes."""
        return {"source": next(iter(graph.scan_nodes()), 0)}

    def run(self) -> VertexMap[float]:
        settled: dict[int, float] = {self.source: 0.0}
        buckets: dict[int, set[int]] = {0: {self.source}}
        index = 0
        while buckets:
            index = min(buckets)
            wave = buckets.pop(index)
            while wave:
                reached: dict[int, float] = {}
                for node, other, along in self.arcs_of(wave, self.graph.outgoing_role, self.weight):
                    candidate = settled[node] + along
                    if self.cutoff is not None and candidate > self.cutoff:
                        continue
                    if candidate < settled.get(other, inf) and candidate < reached.get(other, inf):
                        reached[other] = candidate
                wave = set()
                for other, candidate in reached.items():
                    if candidate >= settled.get(other, inf):
                        continue
                    settled[other] = candidate
                    into = int(candidate // self.delta)
                    if into == index:
                        wave.add(other)
                    else:
                        buckets.setdefault(into, set()).add(other)
        return sparsely(settled, "d")


def delta_stepping_lengths(
    graph: BaseGraph,
    source: int,
    weight: str | None = "weight",
    delta: float = 1.0,
    cutoff: float | None = None,
) -> VertexMap[float]:
    """The same answer as `dijkstra_lengths`, relaxed a whole bucket of vertices in one round trip."""
    return DeltaSteppingLengths.on(graph, source=source, weight=weight, delta=delta, cutoff=cutoff).run()
