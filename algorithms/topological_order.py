"""A topological order, built by peeling away vertices whose incoming edges are gone."""

from __future__ import annotations

from array import array

from algorithms.base import Bounds, VertexOriented
from algorithms.results import DenseIndex
from algorithms.streams import adjacency
from networkxternal.base_api import BaseGraph, NetworkXternalError, Role


class TopologicalOrder(VertexOriented[list[int]]):
    """Every vertex before the ones its edges lead to, or the empty list where a cycle forbids it.

    Kahn's method over an in-degree array: one edge pass builds it, then one adjacency lookup per
    layer of vertices whose incoming edges have all been taken.
    """

    BOUNDS = Bounds(retained_per_vertex=8, working_per_vertex=16, passes="per layer")

    NEEDS_DIRECTION = True

    def run(self) -> list[int]:
        if not self.graph.is_directed():
            raise NetworkXternalError("An undirected graph has no topological order")
        order = DenseIndex(self.graph.scan_nodes())
        incoming = array("q", [0]) * len(order)
        for _, target, _ in self.graph.scan_edges():
            incoming[order[target]] += 1
        ordered: list[int] = []
        frontier = {node for node in order if not incoming[order[node]]}
        while frontier:
            ordered.extend(sorted(frontier))
            beyond: set[int] = set()
            for _, other, _ in adjacency(self.graph, frontier, Role.SOURCE):
                position = order[other]
                incoming[position] -= 1
                if not incoming[position]:
                    beyond.add(other)
            frontier = beyond
        return ordered if len(ordered) == len(order) else []


def topological_order(graph: BaseGraph) -> list[int]:
    """Every vertex before the ones its edges lead to, or the empty list where a cycle forbids it."""
    return TopologicalOrder.on(graph).run()
