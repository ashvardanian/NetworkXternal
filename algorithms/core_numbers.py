"""Core numbers, peeled lowest degree first along one sequential edge stream per round."""

from __future__ import annotations

from array import array
from itertools import compress

from algorithms.base import Bounds, EdgeOriented
from algorithms.results import DenseIndex, VertexMap
from networkxternal.base_api import BaseGraph


def total_degrees(graph: BaseGraph, order: DenseIndex) -> array:
    """How many edge ends every vertex holds, counted in one sequential pass, self-loops excluded."""
    degrees = array("q", [0]) * len(order)
    for source, target, _ in graph.scan_edges():
        if source == target:
            continue
        degrees[order[source]] += 1
        degrees[order[target]] += 1
    return degrees


def peel_below(degrees: array, alive: bytearray, cores: array, level: int) -> bytearray:
    """Marks every surviving vertex at or under `level` as peeled at that level, and reports which."""
    peeled = bytearray(len(alive))
    for index, degree in enumerate(degrees):
        if not alive[index] or degree > level:
            continue
        peeled[index] = 1
        alive[index] = 0
        cores[index] = level
    return peeled


def lower_across_peeled(graph: BaseGraph, order: DenseIndex, degrees: array, peeled: bytearray) -> None:
    """Lowers the degree of every vertex reached from one peeled this round, in one edge pass."""
    for source, target, _ in graph.scan_edges():
        if source == target:
            continue
        left, right = order[source], order[target]
        degrees[right] -= peeled[left]
        degrees[left] -= peeled[right]


class CoreNumbers(EdgeOriented[VertexMap[int]]):
    """The largest `k` whose k-core holds every vertex, by peeling the lowest degree first.

    Self-loops carry no vertex into a core and are skipped, as NetworkX has it.
    """

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=25, passes="per round")

    def run(self) -> VertexMap[int]:
        order = DenseIndex(self.graph.scan_nodes())
        count = len(order)
        degrees = total_degrees(self.graph, order)
        cores = array("q", [0]) * count
        alive = bytearray(b"\x01") * count
        level = 0
        remaining = count
        while remaining:
            level = max(level, min(compress(degrees, alive)))
            peeled = peel_below(degrees, alive, cores, level)
            remaining -= peeled.count(1)
            lower_across_peeled(self.graph, order, degrees, peeled)
        return VertexMap(order, cores)


def core_numbers(graph: BaseGraph) -> VertexMap[int]:
    """The core number of every vertex, as `networkx.core_number` answers it."""
    return CoreNumbers(graph).run()
