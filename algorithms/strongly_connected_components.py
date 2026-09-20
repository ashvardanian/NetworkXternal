"""Strongly connected components, found by forward-backward reachability from a pivot."""

from __future__ import annotations

from array import array

from algorithms.base import Bounds, VertexOriented
from algorithms.results import DenseIndex, VertexMap
from algorithms.streams import adjacency
from networkxternal.base_api import BaseGraph, Role


def reachable_within(graph: BaseGraph, sources: set[int], role: Role, allowed: set[int]) -> set[int]:
    """Every vertex of `allowed` reachable from `sources` along edges in that role, layer by layer."""
    reached = set(sources)
    frontier = set(sources)
    while frontier:
        beyond = {
            other for _, other, _ in adjacency(graph, frontier, role) if other in allowed and other not in reached
        }
        reached |= beyond
        frontier = beyond
    return reached


class StronglyConnectedComponents(VertexOriented[VertexMap[int]]):
    """The strong component every vertex belongs to, named by the smallest vertex in it.

    Forward-backward: a pivot's descendants and its ancestors intersect in its own component, and the
    three remainders are independent of each other and of it. Each side is a layered traversal, so the
    cost is a round trip per layer rather than the recursion depth a depth-first search would need.
    """

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=32, passes="per round")

    def run(self) -> VertexMap[int]:
        order = DenseIndex(self.graph.scan_nodes())
        labels: dict[int, int] = {}
        pending = [set(order.nodes)]
        while pending:
            held = pending.pop()
            if not held:
                continue
            if len(held) == 1:
                node = held.pop()
                labels[node] = node
                continue
            pivot = min(held)
            forward = reachable_within(self.graph, {pivot}, Role.SOURCE, held)
            backward = reachable_within(self.graph, {pivot}, Role.TARGET, held)
            component = forward & backward
            name = min(component)
            labels.update(dict.fromkeys(component, name))
            pending.append(forward - component)
            pending.append(backward - component)
            pending.append(held - forward - backward)
        values = array("q", [0]) * len(order)
        for node, label in labels.items():
            values[order[node]] = label
        return VertexMap(order, values)


def strongly_connected_components(graph: BaseGraph) -> VertexMap[int]:
    """The strong component every vertex belongs to, named by the smallest vertex in it."""
    return StronglyConnectedComponents.on(graph).run()
