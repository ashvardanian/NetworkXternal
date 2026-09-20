"""Shortest path lengths in edge count, built once over a breadth-first walk and returned sparse."""

from __future__ import annotations

from typing import Any

from algorithms.base import Bounds, VertexOriented
from algorithms.breadth_first import BreadthFirstLayers
from algorithms.results import VertexMap, sparsely
from networkxternal.base_api import BaseGraph


class ShortestPathLengths(VertexOriented[VertexMap[int]]):
    """The number of edges from `source` to every vertex it reaches, within `cutoff` steps."""

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=32, passes="per layer")

    def __init__(self, graph: BaseGraph, *, source: int, cutoff: int | None = None) -> None:
        super().__init__(graph)
        self.source = source
        """The vertex the walk starts from."""

        self.cutoff = cutoff
        """The deepest layer to reach, or nothing to walk until the frontier empties."""

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """The first vertex the store holds as the source, which the declared bound assumes."""
        return {"source": next(iter(graph.scan_nodes()), 0)}

    def run(self) -> VertexMap[int]:
        lengths: dict[int, int] = {}
        walk = BreadthFirstLayers(self.graph, sources=self.source, cutoff=self.cutoff)
        for depth, layer in enumerate(walk.layers()):
            lengths.update(dict.fromkeys(layer, depth))
        return sparsely(lengths, "q")


def shortest_path_lengths(graph: BaseGraph, source: int, cutoff: int | None = None) -> VertexMap[int]:
    """The number of edges from `source` to every vertex it reaches, within `cutoff` steps."""
    return ShortestPathLengths.on(graph, source=source, cutoff=cutoff).run()
