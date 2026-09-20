"""Uniform samples of vertices and edges, each drawn over one pass of the stream it reads."""

from __future__ import annotations

from typing import Any

from algorithms.base import Bounds, EdgeOriented, VertexOriented
from algorithms.streams import reservoir
from networkxternal.base_api import BaseGraph, Triple


class SampleNodes(VertexOriented[list[int]]):
    """A uniform sample of vertices, over one pass of the vertex scan."""

    BOUNDS = Bounds(retained_per_vertex=0, working_per_vertex=0, passes="one")

    def __init__(self, graph: BaseGraph, *, count: int, seed: int | None = None) -> None:
        super().__init__(graph)
        self.count = count
        """How many vertices the sample holds."""

        self.seed = seed
        """What seeds the draw, or nothing for a fresh one each call."""

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """Sixty-four vertices, which is what the declared bound is measured at."""
        return {"count": 64, "seed": 42}

    def run(self) -> list[int]:
        return reservoir(self.graph.scan_nodes(), self.count, self.seed)


class SampleEdges(EdgeOriented[list[Triple]]):
    """A uniform sample of edges, over one pass of the edge stream."""

    BOUNDS = Bounds(retained_per_vertex=0, working_per_vertex=0, passes="one")

    def __init__(self, graph: BaseGraph, *, count: int, seed: int | None = None) -> None:
        super().__init__(graph)
        self.count = count
        """How many edges the sample holds."""

        self.seed = seed
        """What seeds the draw, or nothing for a fresh one each call."""

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """Sixty-four edges, which is what the declared bound is measured at."""
        return {"count": 64, "seed": 42}

    def run(self) -> list[Triple]:
        return reservoir(self.graph.scan_edges(), self.count, self.seed)


def sample_nodes(graph: BaseGraph, count: int, seed: int | None = None) -> list[int]:
    """A uniform sample of `count` vertices, over one pass of the vertex scan."""
    return SampleNodes.on(graph, count=count, seed=seed).run()


def sample_edges(graph: BaseGraph, count: int, seed: int | None = None) -> list[Triple]:
    """A uniform sample of `count` edges, over one pass of the edge stream."""
    return SampleEdges.on(graph, count=count, seed=seed).run()
