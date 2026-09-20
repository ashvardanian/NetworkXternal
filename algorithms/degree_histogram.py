"""How many vertices hold each degree, tallied over one pass of the degree view."""

from __future__ import annotations

from collections import Counter

from algorithms.base import Bounds, VertexOriented
from networkxternal.base_api import BaseGraph, DegreeView, Role


class DegreeHistogram(VertexOriented[Counter[int]]):
    """How many vertices hold each degree, streamed rather than sorted into a list."""

    BOUNDS = Bounds(retained_per_vertex=0, working_per_vertex=0, passes="one")

    def __init__(self, graph: BaseGraph, *, role: Role = Role.ANY) -> None:
        super().__init__(graph)
        self.role = role
        """Which end of each edge the degree counts."""

    def run(self) -> Counter[int]:
        return Counter(degree for _, degree in DegreeView(self.graph, self.role))


def degree_histogram(graph: BaseGraph, role: Role = Role.ANY) -> Counter[int]:
    """How many vertices hold each degree, streamed rather than sorted into a list."""
    return DegreeHistogram.on(graph, role=role).run()
