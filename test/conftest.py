"""Fixtures shared by the suite: a SQLite-backed graph of every shape, and a tiny weighted graph to fill it."""

from __future__ import annotations

import networkx
import pytest

from networkxternal.sqlite import SQLiteDiGraph, SQLiteGraph, SQLiteMultiDiGraph, SQLiteMultiGraph

EDGES = [
    (1, 2, 4.0),
    (2, 3, 20.0),
    (3, 4, 10.0),
    (4, 5, 3.0),
    (5, 3, 2.0),
    (4, 1, 5.0),
    (8, 6, 4.0),
    (8, 7, 2.0),
    (6, 1, 3.0),
    (7, 1, 2.0),
]
"""A weighted graph small enough to assert on by hand, with a self-loop and an isolated vertex."""

SHAPES = {
    "graph": SQLiteGraph,
    "digraph": SQLiteDiGraph,
    "multigraph": SQLiteMultiGraph,
    "multidigraph": SQLiteMultiDiGraph,
}


@pytest.fixture(params=list(SHAPES), ids=list(SHAPES))
def empty(request, tmp_path):
    """An empty graph of one shape, in a SQLite file of its own."""
    with SHAPES[request.param](f"sqlite:///{tmp_path}/{request.param}.db3") as graph:
        yield graph


@pytest.fixture
def populated(empty):
    """The `EDGES` graph, with every edge carrying its weight."""
    sources = [source for source, _, _ in EDGES]
    targets = [target for _, target, _ in EDGES]
    weights = [weight for _, _, weight in EDGES]
    empty.add_edges_from_arrays(sources, targets, columns={"weight": weights})
    return empty


@pytest.fixture
def edges():
    """The edge list every fixture and assertion here is built from."""
    return EDGES


@pytest.fixture
def reference(populated):
    """The same graph in RAM, of the NetworkX class the fixture's shape mirrors."""
    classes = {
        (False, False): networkx.Graph,
        (True, False): networkx.DiGraph,
        (False, True): networkx.MultiGraph,
        (True, True): networkx.MultiDiGraph,
    }
    held = classes[(populated.is_directed(), populated.is_multigraph())]()
    held.add_weighted_edges_from(EDGES)
    return held


@pytest.fixture
def undirected():
    """The `EDGES` graph in RAM, undirected, which the semi-external algorithms are compared against."""
    held = networkx.Graph()
    held.add_weighted_edges_from(EDGES)
    return held
