"""Hubs and authorities, pushed along an edge stream or pulled per page of vertices."""

from __future__ import annotations

from abc import abstractmethod
from array import array
from typing import Any

from algorithms.base import Bounds, EdgeOriented, Sweeping, VertexOriented
from algorithms.results import DenseIndex, VertexMap
from networkxternal.base_api import BaseGraph, Orientation

type Scores = tuple[VertexMap[float], VertexMap[float]]
"""The hub score and the authority score of every vertex, sharing one index between them."""


def normalized(values: array) -> array:
    """The values scaled to sum to one, or left alone where they sum to nothing."""
    total = sum(values)
    return array("d", [value / total for value in values]) if total else values


class HITS(Sweeping[Scores], abstract=True):
    """The hub and authority score of every vertex, as `networkx.hits` answers them.

    Hubs push into authorities and authorities back into hubs, so the two vectors are the only state
    a sweep carries, and both answers share the one index.
    """

    def __init__(self, graph: BaseGraph, *, iterations: int = 100, tolerance: float = 1e-8) -> None:
        super().__init__(graph, iterations=iterations, tolerance=tolerance)

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """Three sweeps, which is what the declared bound is measured at."""
        return {"iterations": 3}

    def start(self) -> None:
        self.order = DenseIndex(self.graph.scan_nodes())
        self.count = len(self.order)
        self.hubs = array("d", [1.0 / self.count]) * self.count if self.count else array("d")
        self.authorities = array("d", self.hubs)

    @abstractmethod
    def push_authorities(self) -> array:
        """The authority mass every vertex draws from its in-neighbours' hub scores."""

    @abstractmethod
    def push_hubs(self, authorities: array) -> array:
        """The hub mass every vertex draws from its out-neighbours' authority scores."""

    def sweep(self) -> float:
        pushed_authorities = self.push_authorities()
        pushed_hubs = self.push_hubs(pushed_authorities)
        self.authorities = normalized(pushed_authorities)
        updated = normalized(pushed_hubs)
        drift = sum(abs(one - other) for one, other in zip(updated, self.hubs, strict=True))
        self.hubs = updated
        return drift

    def result(self) -> Scores:
        return VertexMap(self.order, self.hubs), VertexMap(self.order, self.authorities)


class ScatteredHITS(HITS, EdgeOriented[Scores]):
    """Both vectors pushed along every edge as it arrives, two sequential passes per sweep."""

    BOUNDS = Bounds(retained_per_vertex=24, working_per_vertex=40, passes="per sweep")

    def push_authorities(self) -> array:
        pushed = array("d", [0.0]) * self.count
        for source, target, _ in self.arcs():
            pushed[self.order[target]] += self.hubs[self.order[source]]
        return pushed

    def push_hubs(self, authorities: array) -> array:
        pushed = array("d", [0.0]) * self.count
        for source, target, _ in self.arcs():
            pushed[self.order[source]] += authorities[self.order[target]]
        return pushed


class GatheredHITS(HITS, VertexOriented[Scores]):
    """Both vectors pulled from a page of vertices' neighbours, two round trips per page per sweep."""

    BOUNDS = Bounds(retained_per_vertex=24, working_per_vertex=40, passes="per page per sweep")

    def push_authorities(self) -> array:
        pushed = array("d", [0.0]) * self.count
        for page in self.pages(self.order.nodes):
            for node, other, _ in self.arcs_of(page, self.graph.incoming_role):
                pushed[self.order[node]] += self.hubs[self.order[other]]
        return pushed

    def push_hubs(self, authorities: array) -> array:
        pushed = array("d", [0.0]) * self.count
        for page in self.pages(self.order.nodes):
            for node, other, _ in self.arcs_of(page, self.graph.outgoing_role):
                pushed[self.order[node]] += authorities[self.order[other]]
        return pushed


HITS.VARIANTS = {Orientation.EDGE: ScatteredHITS, Orientation.VERTEX: GatheredHITS}


def hits(graph: BaseGraph, iterations: int = 100, tolerance: float = 1e-8) -> Scores:
    """The hub and authority score of every vertex, in the orientation the store prefers."""
    return HITS.on(graph, iterations=iterations, tolerance=tolerance).run()
