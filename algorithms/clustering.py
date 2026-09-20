"""Clustering coefficients, built from triangle counts and degrees the engine already answers."""

from __future__ import annotations

from array import array
from collections.abc import Iterable

from algorithms.base import Bounds, EdgeOriented
from algorithms.results import VertexMap
from algorithms.triangles import TriangleCounts
from networkxternal.base_api import BaseGraph, Role


class ClusteringCoefficients(EdgeOriented[VertexMap[float]]):
    """How close each vertex's neighbours are to being a clique, as `networkx.clustering` answers it.

    The triangles are counted once and the degrees come from the engine, so this costs what
    `TriangleCounts` costs and nothing beyond it.
    """

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=32, passes="two")

    def __init__(self, graph: BaseGraph, *, nodes: Iterable[int] | None = None) -> None:
        super().__init__(graph)
        self.nodes = nodes
        """The subset of vertices to answer for, or every vertex when unset."""

    def run(self) -> VertexMap[float]:
        triangles = TriangleCounts(self.graph, nodes=self.nodes).run()
        wanted = triangles.order.nodes
        degrees = self.graph.degrees(wanted, Role.ANY)
        coefficients = array("d", [0.0]) * len(triangles.order)
        for index, (degree, closed) in enumerate(zip(degrees, triangles.held, strict=True)):
            pairs = degree * (degree - 1)
            coefficients[index] = 2.0 * closed / pairs if pairs else 0.0
        return VertexMap(triangles.order, coefficients)


def clustering_coefficients(graph: BaseGraph, nodes: Iterable[int] | None = None) -> VertexMap[float]:
    """The clustering coefficient of every vertex, in the subset asked for or the whole graph."""
    return ClusteringCoefficients(graph, nodes=nodes).run()
