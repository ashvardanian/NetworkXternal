"""The bounds every algorithm declares, measured rather than trusted.

Two things are checked. What an answer costs once the call returns, measured with `tracemalloc`
filtered to the algorithms themselves — a store compiles and caches statements on first use, and
billing that to whichever algorithm touched it first would tell nobody anything. And that an
algorithm holding no state per edge does not grow with the edges, measured unfiltered so a store's
own growth is caught too, which is the claim that makes it usable on a graph larger than memory.

Every algorithm here comes from the registry, so one that is written but never imported cannot go
unmeasured, and one whose declaration drifts from what it holds fails on the next run.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import pytest

import algorithms
from algorithms import REGISTRY
from algorithms.base import Algorithm
from bench.datasets import Dataset
from networkxternal.sqlite import SQLiteDiGraph, SQLiteGraph

SIZES = (2_000, 20_000)
"""Two graphs an order of magnitude apart, which is what makes a per-vertex claim checkable."""

TOLERANCE = 1.4
"""How far a measurement may exceed its declaration before the test calls the declaration wrong."""

PAGE = 256
"""The page every graph here is narrowed to, so the rows in flight are a constant the slope cancels."""

ANSWERS = [tracemalloc.Filter(True, str(Path(algorithms.__file__).parent / "*"))]
"""Only what an algorithm allocates; a store's own caches are one-time, shared, and not its to declare."""

NAMES = sorted(REGISTRY)
"""Every algorithm that declares a bound, which is what this file measures one by one."""


def filled(directory: Path, edges: int, directed: bool) -> SQLiteGraph:
    """A generated graph of about `edges` edges, in a SQLite file of its own."""
    shape = SQLiteDiGraph if directed else SQLiteGraph
    graph = shape(f"sqlite:///{directory}/bounds-{'di' if directed else ''}{edges}.db3")
    page = list(Dataset(f"bounds-{edges}", nodes=edges // 5, edges=edges).stream())
    graph.add_edges_from_arrays([source for source, _, _ in page], [target for _, target, _ in page])
    return graph


def driven(kind: type[Algorithm], graph: SQLiteGraph) -> object:
    """One algorithm run at the arguments its declaration assumes."""
    return kind.probe(graph).run()


def retained(kind: type[Algorithm], graph: SQLiteGraph) -> int:
    """How many bytes of the algorithms the answer still holds when the call returns."""
    tracemalloc.start(4)
    held = driven(kind, graph)
    snapshot = tracemalloc.take_snapshot().filter_traces(ANSWERS)
    tracemalloc.stop()
    size = sum(statistic.size for statistic in snapshot.statistics("filename"))
    del held
    return size


def peak(kind: type[Algorithm], graph: SQLiteGraph) -> int:
    """The high-water mark of every allocation the walk causes, the driver's included."""
    tracemalloc.start()
    driven(kind, graph)
    held = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return held


@pytest.fixture(scope="module")
def graphs(tmp_path_factory):
    """One undirected and one directed graph per size, built once, since building them is the slow part."""
    directory = tmp_path_factory.mktemp("bounds")
    held = {(edges, directed): filled(directory, edges, directed) for edges in SIZES for directed in (False, True)}
    SQLiteGraph.PAGE = PAGE
    SQLiteDiGraph.PAGE = PAGE
    yield held
    del SQLiteGraph.PAGE, SQLiteDiGraph.PAGE
    for graph in held.values():
        graph.close()


def sized(graphs, name: str, edges: int) -> SQLiteGraph:
    """The graph of that size this algorithm can answer for, directed where it insists."""
    return graphs[edges, REGISTRY[name].NEEDS_DIRECTION]


@pytest.mark.parametrize("name", NAMES)
def test_the_answer_costs_what_it_says(name, graphs):
    """The bytes an answer retains stay within the declaration, at the larger size."""
    kind = REGISTRY[name]
    graph = sized(graphs, name, SIZES[-1])
    held = retained(kind, graph)
    if kind.BOUNDS.retained_per_edge:
        measured = held / graph.count_edges()
        promised = kind.BOUNDS.retained_per_edge
        assert measured < promised * TOLERANCE, f"{name} retains {measured:.0f} bytes per edge, declaring {promised}"
        return
    measured = held / graph.count_nodes()
    promised = kind.BOUNDS.retained_per_vertex
    if promised == 0:
        assert measured < 8, f"{name} retains {measured:.0f} bytes per vertex, declaring none"
        return
    assert measured < promised * TOLERANCE, f"{name} retains {measured:.0f} bytes per vertex, declaring {promised}"


@pytest.mark.parametrize("name", NAMES)
def test_the_walk_grows_with_what_it_says_and_nothing_else(name, graphs):
    """Ten times the graph costs about ten times the peak, against whichever count the bound names."""
    kind = REGISTRY[name]
    small, large = (sized(graphs, name, edges) for edges in SIZES)
    growth = peak(kind, large) / max(peak(kind, small), 1)
    if kind.BOUNDS.retained_per_edge:
        allowed = large.count_edges() / small.count_edges()
        assert growth < allowed * TOLERANCE, f"{name} grew {growth:.1f}x where its edges grew {allowed:.1f}x"
        return
    allowed = large.count_nodes() / small.count_nodes()
    assert growth < allowed * TOLERANCE, f"{name} grew {growth:.1f}x where its vertices grew {allowed:.1f}x"
