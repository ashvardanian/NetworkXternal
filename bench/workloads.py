"""The measured operations, one function per workload, listed in a registry rather than commented in and out.

A run selects workloads by name — `--only`, `--skip`, or the `enabled` flag here — so switching one off
is a configuration change, never an edit that leaves dead code behind.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import batched

from bench.limits import Budget
from networkxternal.algorithms import breadth_first_layers, neighbors_of_neighbors, pagerank
from networkxternal.base_api import BaseGraph

type Sample = Sequence[tuple[int, int, float | None]]

CHECK_EVERY = 4096
"""How many streamed rows pass between clock checks, so the check costs nothing on a fast scan."""


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

    run: Callable[[BaseGraph, Sequence[tuple[int, int, float | None]], Budget], int]
    """Runs the workload over a sample until its budget expires, answering how many operations it did."""

    enabled: bool = True
    """Whether a default run includes it."""


def find_edge(graph: BaseGraph, sample: Sample, budget: Budget) -> int:
    done = 0
    for source, target, _ in sample:
        if budget.expired():
            break
        graph.has_edge(source, target)
        done += 1
    return done


def find_edges_batch(graph: BaseGraph, sample: Sample, budget: Budget) -> int:
    """One engine-side lookup for the whole sample of pairs, against the per-pair workload above."""
    sources = [source for source, _, _ in sample]
    targets = [target for _, target, _ in sample]
    return sum(1 for _ in graph.find_pairs(sources, targets))


def scan_edges(graph: BaseGraph, sample: Sample, budget: Budget) -> int:
    """Streams the edge set once, which is the cost every whole-graph reader pays."""
    done = 0
    for _ in graph.scan_edges():
        done += 1
        if done % CHECK_EVERY == 0 and budget.expired():
            break
    return done


def find_edges_of_node(graph: BaseGraph, sample: Sample, budget: Budget) -> int:
    done = 0
    for source, _, _ in sample:
        if budget.expired():
            break
        sum(1 for _ in graph.edge_triples(source))
        done += 1
    return done


def find_neighbors(graph: BaseGraph, sample: Sample, budget: Budget) -> int:
    done = 0
    for source, _, _ in sample:
        if budget.expired():
            break
        sum(1 for _ in graph.neighbors(source))
        done += 1
    return done


def count_degree(graph: BaseGraph, sample: Sample, budget: Budget) -> int:
    nodes = [source for source, _, _ in sample]
    done = 0
    for page in batched(nodes, graph.PAGE):
        if budget.expired():
            break
        graph.degrees(list(page), graph.outgoing_role)
        done += len(page)
    return done


def find_neighbors_of_neighbors(graph: BaseGraph, sample: Sample, budget: Budget) -> int:
    done = 0
    for source, _, _ in sample:
        if budget.expired():
            break
        neighbors_of_neighbors(graph, source)
        done += 1
    return done


def traverse_breadth_first(graph: BaseGraph, sample: Sample, budget: Budget) -> int:
    reached = 0
    for source, _, _ in sample:
        if budget.expired():
            break
        reached += sum(len(layer) for layer in breadth_first_layers(graph, source, cutoff=3))
    return reached


def rank_pages(graph: BaseGraph, sample: Sample, budget: Budget) -> int:
    """One PageRank run; the budget is checked before it starts, since a sweep is indivisible."""
    if budget.expired():
        return 0
    return len(pagerank(graph, iterations=3, weight=None))


def upsert_edge(graph: BaseGraph, sample: Sample, budget: Budget) -> int:
    done = 0
    for source, target, weight in sample:
        if budget.expired():
            break
        graph.add_edge(source, target, weight=weight)
        done += 1
    return done


def upsert_edges_batch(graph: BaseGraph, sample: Sample, budget: Budget) -> int:
    graph.add_edges_from_arrays(
        [source for source, _, _ in sample],
        [target for _, target, _ in sample],
        columns={"weight": [weight for _, _, weight in sample]},
    )
    return len(sample)


def remove_edge(graph: BaseGraph, sample: Sample, budget: Budget) -> int:
    done = 0
    for source, target, _ in sample:
        if budget.expired():
            break
        graph.remove_edges_from_arrays([source], [target])
        done += 1
    return done


WORKLOADS = (
    Workload("Random Reads: Find Edge", Phase.READ, find_edge),
    Workload("Random Reads: Find Edges Batch", Phase.READ, find_edges_batch),
    Workload("Sequential Reads: Scan Edges", Phase.READ, scan_edges),
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
