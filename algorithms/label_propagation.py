"""Label propagation, tallied over an edge stream or per page of vertices."""

from __future__ import annotations

from abc import abstractmethod
from array import array
from collections import Counter
from typing import Any

from algorithms.base import Bounds, EdgeOriented, Sweeping, VertexOriented
from algorithms.results import DenseIndex, VertexMap
from networkxternal.base_api import BaseGraph, Orientation


def most_carried(counted: Counter[int]) -> int:
    """The label most of a vertex's neighbours carry, ties going to the smallest so runs agree."""
    return min(counted.items(), key=lambda held: (-held[1], held[0]))[0]


class LabelPropagation(Sweeping[VertexMap[int]], abstract=True):
    """The community every vertex belongs to, by taking the label most of its neighbours carry.

    A sweep reads the labels as they stood when it began and writes a fresh set, so the answer does
    not depend on the order vertices were visited in — which is what lets the two variants agree.
    """

    def __init__(self, graph: BaseGraph, *, iterations: int = 32) -> None:
        super().__init__(graph, iterations=iterations)

    def settled(self, drift: float) -> bool:
        """A sweep that moved no vertex ends the walk; `iterations` caps one that keeps oscillating."""
        return not drift

    @classmethod
    def probing(cls, graph: BaseGraph) -> dict[str, Any]:
        """Three sweeps, which is what the declared bound is measured at."""
        return {"iterations": 3}

    def start(self) -> None:
        self.order = DenseIndex(self.graph.scan_nodes())
        self.count = len(self.order)
        self.labels = array("q", self.order.nodes)

    @abstractmethod
    def settle(self, pending: array) -> None:
        """Writes each vertex's new label into `pending` — the one thing the orientations disagree about."""

    def sweep(self) -> float:
        pending = array("q", self.labels)
        self.settle(pending)
        moved = sum(one != other for one, other in zip(self.labels, pending, strict=True))
        self.labels = pending
        return float(moved)

    def result(self) -> VertexMap[int]:
        return VertexMap(self.order, self.labels)


class ScatteredLabelPropagation(LabelPropagation, EdgeOriented[VertexMap[int]]):
    """Labels tallied as every edge arrives, in one sequential pass per sweep.

    The tallies are a counter per vertex reached, so this variant holds the graph's own width in RAM
    where the gathered one holds a page's.
    """

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=136, passes="per sweep")

    def settle(self, pending: array) -> None:
        tallies: dict[int, Counter[int]] = {}
        for source, target, _ in self.arcs():
            tallies.setdefault(target, Counter())[self.labels[self.order[source]]] += 1
        for node, counted in tallies.items():
            pending[self.order[node]] = most_carried(counted)


class GatheredLabelPropagation(LabelPropagation, VertexOriented[VertexMap[int]]):
    """Labels pulled from a page of vertices' neighbours, one round trip per page.

    Only the page being settled is tallied, so the counters never grow with the graph.
    """

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=16, passes="per page per sweep")

    def settle(self, pending: array) -> None:
        for page in self.pages(self.order.nodes):
            tallies: dict[int, Counter[int]] = {}
            for node, other, _ in self.arcs_of(page, self.graph.incoming_role):
                tallies.setdefault(node, Counter())[self.labels[self.order[other]]] += 1
            for node, counted in tallies.items():
                pending[self.order[node]] = most_carried(counted)


LabelPropagation.VARIANTS = {
    Orientation.EDGE: ScatteredLabelPropagation,
    Orientation.VERTEX: GatheredLabelPropagation,
}


def label_propagation(graph: BaseGraph, iterations: int = 32) -> VertexMap[int]:
    """The community every vertex belongs to, in the orientation the store prefers."""
    return LabelPropagation.on(graph, iterations=iterations).run()
