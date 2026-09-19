"""The bounds every algorithm declares, measured rather than trusted.

Two things are checked. What an answer costs per vertex once the call returns, measured with
`tracemalloc` filtered to this package — the driver allocates far more than any algorithm here, and
billing SQLAlchemy's cursor buffers to PageRank would tell nobody anything. And that an algorithm
holding no state per edge does not grow with the edges, which is the claim that makes it usable on a
graph larger than memory.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import pytest

import networkxternal
from bench.datasets import Dataset
from networkxternal.algorithms import (
    BOUNDS,
    breadth_first_layers,
    connected_components,
    core_numbers,
    degree_histogram,
    pagerank,
    sample_edges,
    sample_nodes,
    triangle_counts,
)
from networkxternal.sqlite import SQLiteGraph

SIZES = (2_000, 20_000)
"""Two graphs an order of magnitude apart, which is what makes a per-vertex claim checkable."""

TOLERANCE = 1.4
"""How far a measurement may exceed its declaration before the test calls the declaration wrong."""

PAGE = 256
"""The page every graph here is narrowed to, so the rows in flight are a constant the slope cancels."""

OURS = tracemalloc.Filter(True, str(Path(networkxternal.__file__).parent / "*"))
"""Only this package's allocations; a driver's buffers are its own business and dwarf ours."""

WALKS = {
    "breadth_first_layers": lambda graph: [len(layer) for layer in breadth_first_layers(graph, 1)],
    "connected_components": connected_components,
    "pagerank": lambda graph: pagerank(graph, iterations=3),
    "core_numbers": core_numbers,
    "triangle_counts": triangle_counts,
    "degree_histogram": degree_histogram,
    "sample_nodes": lambda graph: sample_nodes(graph, 64, seed=42),
    "sample_edges": lambda graph: sample_edges(graph, 64, seed=42),
}
"""How each declared algorithm is driven, with the arguments its bound is declared for."""


def filled(directory: Path, edges: int) -> SQLiteGraph:
    """A generated graph of about `edges` edges, in a SQLite file of its own."""
    graph = SQLiteGraph(f"sqlite:///{directory}/bounds-{edges}.db3")
    page = list(Dataset(f"bounds-{edges}", nodes=edges // 5, edges=edges).stream())
    graph.add_edges_from_arrays([source for source, _, _ in page], [target for _, target, _ in page])
    return graph


def retained(run, graph) -> int:
    """How many bytes of this package the answer still holds when the call returns."""
    tracemalloc.start(4)
    held = run(graph)
    snapshot = tracemalloc.take_snapshot().filter_traces([OURS])
    tracemalloc.stop()
    size = sum(statistic.size for statistic in snapshot.statistics("filename"))
    del held
    return size


def peak(run, graph) -> int:
    """The high-water mark of every allocation the walk causes, the driver's included."""
    tracemalloc.start()
    run(graph)
    held = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return held


@pytest.fixture(scope="module")
def graphs(tmp_path_factory):
    """One graph per size, built once, since building them is the slow part."""
    directory = tmp_path_factory.mktemp("bounds")
    held = {edges: filled(directory, edges) for edges in SIZES}
    SQLiteGraph.PAGE = PAGE
    yield held
    del SQLiteGraph.PAGE
    for graph in held.values():
        graph.close()


@pytest.mark.parametrize("name", sorted(WALKS))
def test_every_algorithm_declares_a_bound(name):
    """Nothing lands without a declaration, which is what makes the rest of this file possible."""
    assert name in BOUNDS


@pytest.mark.parametrize("name", sorted(WALKS))
def test_the_answer_costs_what_it_says(name, graphs):
    """The bytes an answer retains per vertex stay within the declaration, at the larger size."""
    declared = BOUNDS[name].retained_per_vertex
    graph = graphs[SIZES[-1]]
    measured = retained(WALKS[name], graph) / graph.count_nodes()
    if declared == 0:
        assert measured < 8, f"{name} retains {measured:.0f} bytes per vertex, declaring none"
        return
    assert measured < declared * TOLERANCE, f"{name} retains {measured:.0f} bytes per vertex, declaring {declared}"


@pytest.mark.parametrize("name", sorted(WALKS))
def test_the_walk_grows_with_the_vertices_and_not_the_edges(name, graphs):
    """Ten times the graph costs about ten times the peak, never a hundred."""
    small, large = (graphs[edges] for edges in SIZES)
    growth = peak(WALKS[name], large) / max(peak(WALKS[name], small), 1)
    vertices = large.count_nodes() / small.count_nodes()
    assert growth < vertices * TOLERANCE, f"{name} grew {growth:.1f}x where its vertices grew {vertices:.1f}x"
