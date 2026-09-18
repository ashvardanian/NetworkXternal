"""The measured operations, one function per workload, listed in a registry rather than commented in and out.

A run selects workloads by name — `--only`, `--skip`, or the `enabled` flag here — so switching one off
is a configuration change, never an edit that leaves dead code behind.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import batched

from networkxternal.algorithms import breadth_first_layers, neighbors_of_neighbors, pagerank
from networkxternal.base_api import BaseGraph


class Phase(StrEnum):
    """Which part of a run a workload belongs to, so a report groups by what is being stressed."""

    IMPORT = "import"
    READ = "read"
    ANALYTICS = "analytics"
    WRITE = "write"


@dataclass(frozen=True)
class Workload:
    """One measured operation, and how many logical operations one call performs."""

    name: str
    """The name printed in the report."""

    phase: Phase
    """Which part of the run it belongs to."""

    run: Callable[[BaseGraph, Sequence[tuple[int, int, float | None]]], int]
    """Runs the workload over a sample, answering how many operations it performed."""

    enabled: bool = True
    """Whether a default run includes it."""


def find_edge(graph: BaseGraph, sample: Sequence[tuple[int, int, float | None]]) -> int:
    for source, target, _ in sample:
        graph.has_edge(source, target)
    return len(sample)


def find_edges_of_node(graph: BaseGraph, sample: Sequence[tuple[int, int, float | None]]) -> int:
    for source, _, _ in sample:
        list(graph.edge_triples(source))
    return len(sample)


def find_neighbors(graph: BaseGraph, sample: Sequence[tuple[int, int, float | None]]) -> int:
    for source, _, _ in sample:
        list(graph.neighbors(source))
    return len(sample)


def count_degree(graph: BaseGraph, sample: Sequence[tuple[int, int, float | None]]) -> int:
    nodes = [source for source, _, _ in sample]
    for page in batched(nodes, graph.PAGE):
        graph.degrees(list(page), graph.outgoing_role)
    return len(nodes)


def find_neighbors_of_neighbors(graph: BaseGraph, sample: Sequence[tuple[int, int, float | None]]) -> int:
    for source, _, _ in sample:
        neighbors_of_neighbors(graph, source)
    return len(sample)


def traverse_breadth_first(graph: BaseGraph, sample: Sequence[tuple[int, int, float | None]]) -> int:
    reached = 0
    for source, _, _ in sample:
        reached += sum(len(layer) for layer in breadth_first_layers(graph, source, cutoff=3))
    return reached


def rank_pages(graph: BaseGraph, sample: Sequence[tuple[int, int, float | None]]) -> int:
    ranks = pagerank(graph, iterations=3, weight=None)
    return len(ranks)


def upsert_edge(graph: BaseGraph, sample: Sequence[tuple[int, int, float | None]]) -> int:
    for source, target, weight in sample:
        graph.add_edge(source, target, weight=weight)
    return len(sample)


def upsert_edges_batch(graph: BaseGraph, sample: Sequence[tuple[int, int, float | None]]) -> int:
    graph.add_edges_from_arrays(
        [source for source, _, _ in sample],
        [target for _, target, _ in sample],
        columns={"weight": [weight for _, _, weight in sample]},
    )
    return len(sample)


def remove_edge(graph: BaseGraph, sample: Sequence[tuple[int, int, float | None]]) -> int:
    for source, target, _ in sample:
        graph.remove_edges_from_arrays([source], [target])
    return len(sample)


WORKLOADS = (
    Workload("Random Reads: Find Edge", Phase.READ, find_edge),
    Workload("Random Reads: Find Edges of Node", Phase.READ, find_edges_of_node),
    Workload("Random Reads: Find Neighbors", Phase.READ, find_neighbors),
    Workload("Random Reads: Count Degree", Phase.READ, count_degree),
    Workload("Analytics: Neighbors of Neighbors", Phase.ANALYTICS, find_neighbors_of_neighbors),
    Workload("Analytics: Breadth-First to Depth 3", Phase.ANALYTICS, traverse_breadth_first),
    Workload("Analytics: PageRank, 3 Sweeps", Phase.ANALYTICS, rank_pages, enabled=False),
    Workload("Random Writes: Upsert Edge", Phase.WRITE, upsert_edge),
    Workload("Random Writes: Upsert Edges Batch", Phase.WRITE, upsert_edges_batch),
    Workload("Random Writes: Remove Edge", Phase.WRITE, remove_edge),
)
"""Every workload a run can measure, in the order a report prints them."""
