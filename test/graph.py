"""The NetworkX contract over a store: vertices, edges, attributes and degrees, against NetworkX itself."""

from __future__ import annotations

import pytest


def test_empty(empty):
    assert empty.number_of_nodes() == 0
    assert empty.number_of_edges() == 0
    assert list(empty.nodes) == []
    assert list(empty.edges) == []


def test_counts_match_networkx(populated, reference):
    held = reference
    assert populated.number_of_nodes() == held.number_of_nodes()
    assert populated.number_of_edges() == held.number_of_edges()
    assert sorted(populated.nodes) == sorted(held.nodes)


def test_membership(populated):
    assert 1 in populated
    assert 99 not in populated
    assert populated.has_edge(1, 2)
    assert not populated.has_edge(1, 5)


def test_neighbors_match_networkx(populated, reference):
    held = reference
    for node in sorted(held.nodes):
        assert list(populated.neighbors(node)) == sorted(held.neighbors(node))


def test_degrees_match_networkx(populated, reference):
    held = reference
    for node in sorted(held.nodes):
        assert populated.degree[node] == held.degree[node]


def test_weighted_degrees_match_networkx(populated, reference):
    held = reference
    weighted = populated.degree(weight="weight")
    for node, degree in weighted:
        assert degree == pytest.approx(held.degree(node, weight="weight"))


def test_size_with_weight(populated, reference):
    held = reference
    assert populated.size() == held.size()
    assert populated.size(weight="weight") == pytest.approx(held.size(weight="weight"))


def test_edge_attributes_round_trip(populated):
    stored = populated.get_edge_data(1, 2)
    weights = stored.values() if populated.is_multigraph() else [stored]
    assert all(held["weight"] == pytest.approx(4.0) for held in weights)
    populated.set_edge_attributes(7, name="label")
    assert set(populated.get_edge_attributes("label").values()) == {7}


def test_node_attributes_round_trip(populated):
    populated.add_node(1, name="first")
    assert populated.nodes[1]["name"] == "first"
    assert populated.get_node_attributes("name") == {1: "first"}


def test_isolated_node_survives(empty):
    empty.add_node(42)
    assert 42 in empty
    assert empty.number_of_nodes() == 1
    assert empty.number_of_edges() == 0


def test_remove_edge_then_node(populated):
    populated.remove_edge(1, 2)
    assert not populated.has_edge(1, 2)
    populated.remove_node(1)
    assert 1 not in populated
    assert all(1 not in edge[:2] for edge in populated.edges)


def test_clear(populated):
    populated.clear()
    assert populated.number_of_nodes() == 0
    assert populated.number_of_edges() == 0


def test_clear_edges_keeps_nodes(populated):
    held = populated.number_of_nodes()
    populated.clear_edges()
    assert populated.number_of_edges() == 0
    assert populated.number_of_nodes() == held
