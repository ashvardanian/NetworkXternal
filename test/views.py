"""The views: what they answer, what they cost, and which protocols they really implement."""

from __future__ import annotations

import tracemalloc
from collections.abc import Mapping, Set

import pytest

from networkxternal.base_api import AdjacencyView


def test_nodes_is_a_set(populated):
    """A vertex view supports the set operations NetworkX's does, because it is one."""
    assert isinstance(populated.nodes, Set)
    assert populated.nodes & {1, 2, 99} == {1, 2}
    assert 99 not in populated.nodes
    assert len(populated.nodes) == populated.count_nodes()


def test_edges_is_a_set(populated):
    """An edge view is a set of the tuples it yields."""
    assert isinstance(populated.edges, Set)
    assert len(populated.edges) == populated.count_edges()


def test_adjacency_is_a_mapping(populated):
    """Adjacency answers as a mapping, and its lookups reach the store."""
    assert isinstance(populated.adj, Mapping)
    assert isinstance(populated.adj, AdjacencyView)
    assert set(populated.adj[1]) == set(populated.neighbors(1))
    assert len(populated.adj) == populated.count_nodes()
    assert 99 not in populated.adj


def test_adjacency_does_not_hold_the_graph(populated):
    """Reading one vertex's neighbours costs that vertex, not the graph."""
    tracemalloc.start()
    held = populated.adj
    after_view = tracemalloc.get_traced_memory()[0]
    held[1]
    tracemalloc.stop()
    assert after_view < 4096, "taking the view should allocate nothing of consequence"


def test_predecessors_answer_through_the_view(populated):
    if not populated.is_directed():
        pytest.skip("An undirected graph has no separate predecessors")
    assert set(populated.pred[2]) == set(populated.predecessors(2))


def test_a_missing_vertex_raises(populated):
    with pytest.raises(KeyError):
        populated.adj[99]
