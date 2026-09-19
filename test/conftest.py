"""Fixtures shared by the suite: one graph of every shape, and a tiny weighted graph to fill it.

The backend is chosen by `NETWORKXTERNAL_TEST_BACKEND`, so the same assertions run against SQLite by
default and against a server when one is up, without a second copy of the suite.
"""

from __future__ import annotations

import os
from importlib import import_module

import networkx
import pytest

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
    (3, 3, 1.5),
]
"""A weighted graph small enough to assert on by hand, with a branch, a cycle, a bridge and a self-loop."""

BACKENDS = {
    "sqlite": ("networkxternal.sqlite", "SQLite", "sqlite:///{directory}/{shape}.db3"),
    "ustore": ("networkxternal.ustore", "UStore", "{directory}/{shape}"),
    "ustore-ram": ("networkxternal.ustore", "UStore", ":memory:"),
    "mongodb": ("networkxternal.mongodb", "Mongo", "mongodb://localhost:27017/test_{shape}"),
    "clickhouse": ("networkxternal.clickhouse", "ClickHouse", "clickhouse://graph:graph@localhost:8123/test_{shape}"),
    "postgres": ("networkxternal.postgres", "Postgres", "postgresql+psycopg://graph:graph@localhost:5432/test_{shape}"),
    "neo4j": ("networkxternal.neo4j", "Neo4J", "bolt://localhost:7687/test{shape}"),
    "memgraph": ("networkxternal.memgraph", "Memgraph", "bolt://localhost:7688/test{shape}"),
    "mysql": ("networkxternal.mysql", "MySQL", "mysql://root:graph@127.0.0.1:3306/test_{shape}"),
}
"""Where each backend's graph classes live and how a test addresses one of its graphs."""

SHAPES = ("Graph", "DiGraph", "MultiGraph", "MultiDiGraph")
"""The four NetworkX shapes every backend implements, named as NetworkX names them."""


def graph_url(shape: str, directory) -> str:
    """The connection string this run addresses one shape of graph by."""
    _, _, template = BACKENDS[os.getenv("NETWORKXTERNAL_TEST_BACKEND", "sqlite")]
    return os.getenv("NETWORKXTERNAL_TEST_URL", template).format(directory=directory, shape=shape.lower())


def open_graph(shape: str, url: str):
    """Opens one shape of graph on this run's backend."""
    module, prefix, _ = BACKENDS[os.getenv("NETWORKXTERNAL_TEST_BACKEND", "sqlite")]
    return getattr(import_module(module), f"{prefix}{shape}")(url)


@pytest.fixture(params=SHAPES, ids=[shape.lower() for shape in SHAPES])
def empty(request, tmp_path):
    """An empty graph of one shape, in a store of its own, cleared before the test and closed after."""
    url = graph_url(request.param, tmp_path)
    with open_graph(request.param, url) as graph:
        graph.opened_as = (request.param, url)
        """The shape and connection string a test reopens this graph by."""
        graph.clear()
        yield graph


@pytest.fixture
def open_as():
    """Opens a named shape on a given connection string, for a test that reopens a stored graph."""
    return open_graph


@pytest.fixture
def second(empty):
    """A second graph object addressing the same store, which an in-memory database cannot offer."""
    shape, url = empty.opened_as
    if ":memory:" in url or not type(empty).SHARES_STORE:
        pytest.skip(f"{type(empty).__name__} does not address one store from two objects")
    with open_graph(shape, url) as graph:
        yield graph


@pytest.fixture
def populated(empty):
    """The `EDGES` graph, with every edge carrying its weight."""
    empty.add_weighted_edges_from(EDGES)
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
