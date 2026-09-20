"""Triangle counts, by intersecting sorted adjacency runs one vertex at a time."""

from __future__ import annotations

from array import array
from bisect import bisect_left
from collections.abc import Iterable, Sequence

from algorithms.base import Bounds, EdgeOriented
from algorithms.results import DenseIndex, VertexMap, gathered
from networkxternal.base_api import BaseGraph


def compact_run(neighbours: array, start: int, stop: int) -> int:
    """Sorts one vertex's neighbours in place, dropping repeats, and reports where its run now ends."""
    unique = sorted(set(neighbours[start:stop]))
    neighbours[start : start + len(unique)] = array("q", unique)
    return start + len(unique)


def sorted_adjacency(graph: BaseGraph, order: DenseIndex) -> tuple[array, array, array]:
    """Every vertex's neighbours as one sorted run, packed at 8 bytes an edge end, in two edge passes.

    Sorted, which `accumulate_shared`'s `bisect_left` needs for correctness, not merely for speed.
    """
    count = len(order)
    starts = array("q", [0]) * (count + 1)
    for source, target, _ in graph.scan_edges():
        starts[order[source] + 1] += source != target
        starts[order[target] + 1] += source != target
    for index in range(count):
        starts[index + 1] += starts[index]
    neighbours = array("q", [0]) * starts[count]
    cursors = array("q", starts[:count])
    for source, target, _ in graph.scan_edges():
        if source == target:
            continue
        left, right = order[source], order[target]
        neighbours[cursors[left]] = right
        neighbours[cursors[right]] = left
        cursors[left] += 1
        cursors[right] += 1
    ends = array("q", [compact_run(neighbours, starts[index], cursors[index]) for index in range(count)])
    return starts, ends, neighbours


def shared_count(left: Sequence[int], right: Sequence[int]) -> int:
    """How many vertices two sorted adjacency runs share, by walking each of them once."""
    left_index = right_index = shared = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] == right[right_index]:
            shared += 1
            left_index += 1
            right_index += 1
        elif left[left_index] < right[right_index]:
            left_index += 1
        else:
            right_index += 1
    return shared


def accumulate_shared(starts: array, ends: array, neighbours: array, counts: array) -> None:
    """Adds to both ends of every edge how many neighbours the two ends share, run against sorted run.

    Each run is sorted, so the neighbours below the vertex itself are skipped by a search rather than
    one at a time — every unordered pair is then visited exactly once.
    """
    runs = memoryview(neighbours)
    for index in range(len(counts)):
        run = runs[starts[index] : ends[index]]
        for other in run[bisect_left(run, index) :]:
            shared = shared_count(run, runs[starts[other] : ends[other]])
            counts[index] += shared
            counts[other] += shared


class TriangleCounts(EdgeOriented[VertexMap[int]]):
    """How many triangles every vertex takes part in, by intersecting two sorted adjacency runs at a time.

    Holds one packed adjacency of 8 bytes per edge end plus three 8-byte slots per vertex, built in two
    sequential edge passes; a self-loop and a repeated edge are dropped as the runs are sorted.
    """

    BOUNDS = Bounds(retained_per_vertex=16, working_per_vertex=24, passes="two")

    def __init__(self, graph: BaseGraph, *, nodes: Iterable[int] | None = None) -> None:
        super().__init__(graph)
        self.nodes = nodes
        """The subset of vertices to answer for, or every vertex when unset."""

    def run(self) -> VertexMap[int]:
        order = DenseIndex(self.graph.scan_nodes())
        starts, ends, neighbours = sorted_adjacency(self.graph, order)
        counts = array("q", [0]) * len(order)
        accumulate_shared(starts, ends, neighbours, counts)
        halved = array("q", (count // 2 for count in counts))
        return gathered(order, halved, self.nodes)


def triangle_counts(graph: BaseGraph, nodes: Iterable[int] | None = None) -> VertexMap[int]:
    """How many triangles every vertex takes part in, in the subset asked for or the whole graph."""
    return TriangleCounts(graph, nodes=nodes).run()
