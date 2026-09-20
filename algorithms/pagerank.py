"""PageRank, scattered along an edge stream or gathered per page of vertices."""

from __future__ import annotations

from abc import abstractmethod
from array import array
from typing import Any

from algorithms.base import Bounds, EdgeOriented, Sweeping, VertexOriented
from algorithms.results import DenseIndex, VertexMap
from networkxternal.base_api import BaseGraph, Orientation


class PageRank(Sweeping[VertexMap[float]], abstract=True):
    """The PageRank of every vertex, as `networkx.pagerank` answers it.

    `PageRank.on(graph)` picks the variant the store prefers. With `weight` unset every edge carries
    the same mass, which is what NetworkX calls `weight=None`.
    """

    def __init__(
        self,
        graph: BaseGraph,
        *,
        damping: float = 0.85,
        iterations: int = 100,
        tolerance: float = 1e-6,
        weight: str | None = "weight",
    ) -> None:
        super().__init__(graph, iterations=iterations, tolerance=tolerance)
        self.damping = damping
        """How much mass follows the edges rather than teleporting."""

        self.weight = weight
        """The edge attribute the mass is split by, or nothing to split it evenly."""

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """Three sweeps, which is what the declared bound is measured at."""
        return {"iterations": 3}

    def start(self) -> None:
        self.order = DenseIndex(self.graph.scan_nodes())
        self.count = len(self.order)
        self.share = 1.0 / self.count if self.count else 0.0
        self.ranks = array("d", [self.share]) * self.count
        self.scale = self.outgoing_scale()
        # Which vertices lead nowhere never changes, so the sum below walks only those.
        self.dangling_at = [position for position, spread in enumerate(self.scale) if not spread]

    def outgoing_scale(self) -> array:
        """The reciprocal of every vertex's outgoing weight, zero where it has none."""
        totals = array("d", [0.0]) * self.count
        self.total_outgoing(totals)
        return array("d", [1.0 / total if total else 0.0 for total in totals])

    @abstractmethod
    def total_outgoing(self, totals: array) -> None:
        """Sums every vertex's outgoing weight into `totals`, in whichever pass this variant drives."""

    @abstractmethod
    def push(self) -> array:
        """The mass landing on every vertex this sweep — the one thing the orientations disagree about."""

    def sweep(self) -> float:
        pushed = self.push()
        dangling = sum(map(self.ranks.__getitem__, self.dangling_at))
        leaked = self.damping * dangling * self.share + (1.0 - self.damping) * self.share
        drift = 0.0
        for index, mass in enumerate(pushed):
            updated = self.damping * mass + leaked
            drift += abs(updated - self.ranks[index])
            self.ranks[index] = updated
        return drift

    def result(self) -> VertexMap[float]:
        return VertexMap(self.order, self.ranks)


class ScatteredPageRank(PageRank, EdgeOriented[VertexMap[float]]):
    """Mass pushed along every edge as it arrives, one sequential pass per sweep.

    A row's two ends land in unrelated slots, so the pass is random-access in RAM and sequential
    against the store, which is the trade this package makes.
    """

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=32, passes="per sweep")

    def total_outgoing(self, totals: array) -> None:
        for source, _, held in self.arcs(self.weight):
            totals[self.order[source]] += held

    def push(self) -> array:
        pushed = array("d", [0.0]) * self.count
        for source, target, held in self.arcs(self.weight):
            position = self.order[source]
            pushed[self.order[target]] += self.ranks[position] * held * self.scale[position]
        return pushed


class GatheredPageRank(PageRank, VertexOriented[VertexMap[float]]):
    """Mass pulled from every vertex's in-neighbours, one round trip per page of vertices.

    A page is a contiguous run of the index, so a sweep writes sequentially and reads at random —
    the mirror of the scattered variant, and what a store answers natively when it never held an
    edge-ordered scan to begin with.
    """

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=32, passes="per page per sweep")

    def total_outgoing(self, totals: array) -> None:
        for page in self.pages(self.order.nodes):
            for node, _, held in self.arcs_of(page, self.graph.outgoing_role, self.weight):
                totals[self.order[node]] += held

    def push(self) -> array:
        pushed = array("d", [0.0]) * self.count
        for page in self.pages(self.order.nodes):
            for node, other, held in self.arcs_of(page, self.graph.incoming_role, self.weight):
                source = self.order[other]
                pushed[self.order[node]] += self.ranks[source] * held * self.scale[source]
        return pushed


PageRank.VARIANTS = {Orientation.EDGE: ScatteredPageRank, Orientation.VERTEX: GatheredPageRank}


def pagerank(
    graph: BaseGraph,
    damping: float = 0.85,
    iterations: int = 100,
    tolerance: float = 1e-6,
    weight: str | None = "weight",
) -> VertexMap[float]:
    """The PageRank of every vertex, in the orientation the store prefers."""
    return PageRank.on(graph, damping=damping, iterations=iterations, tolerance=tolerance, weight=weight).run()
