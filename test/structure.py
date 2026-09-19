"""Directed structure: strong and weak components, and a topological order."""

from __future__ import annotations

import networkx
import pytest

from networkxternal.algorithms import (
    strongly_connected_components,
    topological_order,
    weakly_connected_components,
)
from networkxternal.base_api import NetworkXternalError

ARCS = [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 4), (6, 1), (6, 7)]
"""Two cycles joined by a bridge, with a vertex that leads into one and past it."""

CHAIN = [(1, 2), (2, 3), (3, 4), (1, 4)]
"""A graph with an order, and more than one valid one."""


@pytest.fixture
def directed(empty):
    """The `ARCS` graph, skipped on the shapes that cannot hold a direction."""
    if not empty.is_directed():
        pytest.skip("Strong components and topological order are directed questions")
    empty.add_edges_from(ARCS)
    return empty


def grouped(labels: dict[int, int]) -> set[frozenset[int]]:
    """The labelling as a set of components, which is how NetworkX answers."""
    held: dict[int, set[int]] = {}
    for node, name in labels.items():
        held.setdefault(name, set()).add(node)
    return {frozenset(component) for component in held.values()}


def test_strong_components_match_networkx(directed):
    held = networkx.DiGraph(ARCS)
    assert grouped(strongly_connected_components(directed)) == {
        frozenset(component) for component in networkx.strongly_connected_components(held)
    }


def test_weak_components_match_networkx(directed):
    held = networkx.DiGraph(ARCS)
    assert grouped(weakly_connected_components(directed)) == {
        frozenset(component) for component in networkx.weakly_connected_components(held)
    }


def test_a_cycle_has_no_topological_order(directed):
    assert topological_order(directed) == []


def test_a_chain_has_one(empty):
    if not empty.is_directed():
        pytest.skip("An undirected graph has no topological order")
    empty.add_edges_from(CHAIN)
    order = topological_order(empty)
    assert len(order) == empty.count_nodes()
    positions = {node: index for index, node in enumerate(order)}
    assert all(positions[source] < positions[target] for source, target in CHAIN)


def test_an_undirected_graph_refuses_a_topological_order(populated):
    if populated.is_directed():
        pytest.skip("A directed graph answers this question")
    with pytest.raises(NetworkXternalError):
        topological_order(populated)
