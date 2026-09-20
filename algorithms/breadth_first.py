"""Breadth-first layers, pulled by adjacency until the frontier is wide enough to scan instead."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from algorithms.base import Bounds, VertexOriented
from algorithms.streams import adjacency, scan_arcs
from networkxternal.base_api import BaseGraph

BOTTOM_UP_FRACTION = 0.1
"""The share of the vertex count a frontier must pass before a layer is found by scanning every edge instead."""

BOTTOM_UP_FLOOR = 4_096
"""How large a frontier must be before the vertex count is worth asking for, which some stores scan to answer."""


def expand_top_down(graph: BaseGraph, frontier: set[int], visited: set[int]) -> set[int]:
    """The unvisited neighbours of a small frontier, pulled by adjacency, one page of rows at a time."""
    return {
        neighbour for _, neighbour, _ in adjacency(graph, frontier, graph.outgoing_role) if neighbour not in visited
    }


def expand_bottom_up(graph: BaseGraph, frontier: set[int], visited: set[int]) -> set[int]:
    """The unvisited neighbours of a wide frontier, found in one sequential pass over every edge."""
    return {target for source, target, _ in scan_arcs(graph, None) if source in frontier and target not in visited}


class BreadthFirstLayers(VertexOriented[list[set[int]]]):
    """The vertices at distance 0, 1, 2 … from `sources`, one layer at a time.

    Which side expands a layer is chosen from that layer's own frontier size, so the class stays
    `VertexOriented` and reaches for `scan_arcs` directly for its bottom-up step.
    """

    BOUNDS = Bounds(retained_per_vertex=40, working_per_vertex=32, passes="per layer")

    def __init__(self, graph: BaseGraph, *, sources: int | Iterable[int], cutoff: int | None = None) -> None:
        super().__init__(graph)
        self.sources = sources
        """The vertex, or vertices, the walk starts from."""

        self.cutoff = cutoff
        """The deepest layer to yield, or nothing to walk until the frontier empties."""

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """One source, which is what the declared bound assumes."""
        return {"sources": next(iter(graph.scan_nodes()), 0)}

    def layers(self) -> Iterator[set[int]]:
        """Yields the vertices at distance 0, 1, 2 … from `sources`, one round-trip page per layer."""
        frontier = {self.sources} if isinstance(self.sources, int) else set(self.sources)
        visited = set(frontier)
        threshold: float | None = None
        depth = 0
        while frontier:
            yield frontier
            depth += 1
            if self.cutoff is not None and depth > self.cutoff:
                return
            # The vertex count is asked for only once a frontier is large enough for the answer to matter,
            # since a store without a counter answers it by scanning every key.
            if threshold is None and len(frontier) >= BOTTOM_UP_FLOOR:
                threshold = BOTTOM_UP_FRACTION * self.graph.number_of_nodes()
            large = threshold is not None and len(frontier) > threshold
            expand = expand_bottom_up if large else expand_top_down
            reached = expand(self.graph, frontier, visited)
            visited |= reached
            frontier = reached

    def run(self) -> list[set[int]]:
        return list(self.layers())


def breadth_first_layers(
    graph: BaseGraph, sources: int | Iterable[int], cutoff: int | None = None
) -> Iterator[set[int]]:
    """Yields the vertices at distance 0, 1, 2 … from `sources`, one round-trip page per layer."""
    return BreadthFirstLayers.on(graph, sources=sources, cutoff=cutoff).layers()
