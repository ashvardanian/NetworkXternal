"""PageRank where the mass that teleports lands on a preference vector rather than everywhere."""

from __future__ import annotations

from array import array
from collections.abc import Mapping
from typing import Any

from algorithms.base import Bounds
from algorithms.pagerank import GatheredPageRank, PageRank, ScatteredPageRank
from algorithms.results import DenseIndex, VertexMap
from networkxternal.base_api import BaseGraph, Orientation


class PersonalizedPageRank(PageRank, abstract=True):
    """PageRank whose teleporting mass lands on `preference` rather than spread evenly.

    The two variants differ only in how they read the graph, which they take from `PageRank`.
    """

    def __init__(
        self,
        graph: BaseGraph,
        *,
        preference: Mapping[int, float],
        damping: float = 0.85,
        iterations: int = 100,
        tolerance: float = 1e-6,
        weight: str | None = "weight",
    ) -> None:
        super().__init__(graph, damping=damping, iterations=iterations, tolerance=tolerance, weight=weight)
        self.preference = preference
        """How the teleporting mass is split, which need not name every vertex."""

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """Three sweeps onto the first vertex the store holds, which is what the bound is measured at."""
        return {"preference": {next(iter(graph.scan_nodes()), 0): 1.0}, "iterations": 3}

    def start(self) -> None:
        self.order = DenseIndex(self.graph.scan_nodes())
        self.count = len(self.order)
        self.share = 1.0 / self.count if self.count else 0.0
        self.landing = self.landing_mass()
        self.ranks = array("d", self.landing)
        self.scale = self.outgoing_scale()
        self.dangling_at = [position for position, spread in enumerate(self.scale) if not spread]

    def landing_mass(self) -> array:
        """Where the teleporting mass lands, scaled to sum to one."""
        total = sum(self.preference.values())
        if not total:
            raise ValueError("A preference vector must carry some mass")
        landing = array("d", [0.0]) * self.count
        for node, mass in self.preference.items():
            landing[self.order[node]] = mass / total
        return landing

    def sweep(self) -> float:
        pushed = self.push()
        dangling = sum(map(self.ranks.__getitem__, self.dangling_at))
        drift = 0.0
        for index, mass in enumerate(pushed):
            landed = self.landing[index]
            updated = self.damping * (mass + dangling * landed) + (1.0 - self.damping) * landed
            drift += abs(updated - self.ranks[index])
            self.ranks[index] = updated
        return drift


class ScatteredPersonalizedPageRank(PersonalizedPageRank, ScatteredPageRank):
    """Mass pushed along every edge as it arrives, one sequential pass per sweep."""

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=40, passes="per sweep")


class GatheredPersonalizedPageRank(PersonalizedPageRank, GatheredPageRank):
    """Mass pulled from every vertex's in-neighbours, one round trip per page of vertices."""

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=40, passes="per page per sweep")


PersonalizedPageRank.VARIANTS = {
    Orientation.EDGE: ScatteredPersonalizedPageRank,
    Orientation.VERTEX: GatheredPersonalizedPageRank,
}


def personalized_pagerank(
    graph: BaseGraph,
    preference: Mapping[int, float],
    damping: float = 0.85,
    iterations: int = 100,
    tolerance: float = 1e-6,
    weight: str | None = "weight",
) -> VertexMap[float]:
    """The personalized PageRank of every vertex, in the orientation the store prefers."""
    return PersonalizedPageRank.on(
        graph,
        preference=preference,
        damping=damping,
        iterations=iterations,
        tolerance=tolerance,
        weight=weight,
    ).run()
