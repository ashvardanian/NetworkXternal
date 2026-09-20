"""Union-find components, converged in one sequential pass over the edge stream."""

from __future__ import annotations

from array import array

from algorithms.base import Bounds, EdgeOriented
from algorithms.results import DenseIndex, VertexMap
from networkxternal.base_api import BaseGraph


def find_root(parents: array, index: int) -> int:
    """The representative of the set holding `index`, compressing the path walked to reach it."""
    root = index
    while parents[root] != root:
        root = parents[root]
    while parents[index] != root:
        parents[index], index = root, parents[index]
    return root


def join_sets(parents: array, depths: array, left: int, right: int) -> None:
    """Joins the two sets holding `left` and `right`, hanging the shallower tree under the deeper one."""
    left, right = find_root(parents, left), find_root(parents, right)
    if left == right:
        return
    if depths[left] < depths[right]:
        left, right = right, left
    parents[right] = left
    if depths[left] == depths[right]:
        depths[left] += 1


class ConnectedComponents(EdgeOriented[VertexMap[int]]):
    """The component every vertex belongs to, named by the smallest vertex in it.

    Union-find over one sequential pass of the edge stream, which converges in that single pass however
    long the paths are. Holds three 8-byte slots per vertex and one page of edge rows, never an adjacency
    list and never a second sweep.
    """

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=24, passes="one")

    def run(self) -> VertexMap[int]:
        order = DenseIndex(self.graph.scan_nodes())
        count = len(order)
        parents = array("q", range(count))
        depths = array("q", [0]) * count
        for source, target, _ in self.graph.scan_edges():
            join_sets(parents, depths, order[source], order[target])
        smallest = array("q", order.nodes)
        for index in range(count):
            root = find_root(parents, index)
            if order.nodes[index] < smallest[root]:
                smallest[root] = order.nodes[index]
        labels = array("q", (smallest[find_root(parents, index)] for index in range(count)))
        return VertexMap(order, labels)


class WeaklyConnectedComponents(ConnectedComponents):
    """The weak component every vertex belongs to, which ignores the direction of every edge."""

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=24, passes="one")


def connected_components(graph: BaseGraph) -> VertexMap[int]:
    """The component every vertex belongs to, named by the smallest vertex in it."""
    return ConnectedComponents.on(graph).run()


def weakly_connected_components(graph: BaseGraph) -> VertexMap[int]:
    """The weak component every vertex belongs to, which ignores the direction of every edge."""
    return WeaklyConnectedComponents.on(graph).run()
