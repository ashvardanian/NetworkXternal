"""Semi-external algorithms against the answers NetworkX gives for the same graph in RAM."""

from __future__ import annotations

import networkx
import pytest

from algorithms import (
    breadth_first_layers,
    connected_components,
    core_numbers,
    neighbors_of_neighbors,
    pagerank,
    sample_edges,
    sample_nodes,
    shortest_path_lengths,
    triangle_counts,
)


def simple(populated):
    """Skips the shapes whose NetworkX counterpart answers differently than the undirected reference."""
    if populated.is_directed():
        pytest.skip("The reference graph is undirected")
    return populated


def test_breadth_first_layers(populated, undirected):
    graph = simple(populated)
    layers = list(breadth_first_layers(graph, 8))
    held = list(networkx.bfs_layers(undirected, 8))
    assert [sorted(layer) for layer in layers] == [sorted(layer) for layer in held]


def test_shortest_path_lengths(populated, undirected):
    graph = simple(populated)
    assert shortest_path_lengths(graph, 1) == dict(networkx.shortest_path_length(undirected, 1))


def test_neighbors_of_neighbors(populated, undirected):
    graph = simple(populated)
    reached = neighbors_of_neighbors(graph, 8)
    held = {node for node, length in networkx.shortest_path_length(undirected, 8).items() if length == 2}
    assert reached == held


def test_connected_components(populated, undirected):
    graph = simple(populated)
    labels = connected_components(graph)
    held = {frozenset(component) for component in networkx.connected_components(undirected)}
    grouped: dict[int, set[int]] = {}
    for node, label in labels.items():
        grouped.setdefault(label, set()).add(node)
    assert {frozenset(component) for component in grouped.values()} == held


def without_loops(held):
    """The reference graph without its self-loop, which NetworkX refuses in these two algorithms."""
    stripped = held.copy()
    stripped.remove_edges_from(networkx.selfloop_edges(stripped))
    return stripped


def test_core_numbers(populated, undirected):
    graph = simple(populated)
    assert core_numbers(graph) == networkx.core_number(without_loops(undirected))


def test_triangle_counts(populated, undirected):
    graph = simple(populated)
    assert triangle_counts(graph) == networkx.triangles(without_loops(undirected))


def test_pagerank(populated, undirected):
    graph = simple(populated)
    ranks = pagerank(graph, iterations=60, tolerance=1e-10)
    held = networkx.pagerank(undirected, tol=1e-10)
    for node, rank in held.items():
        assert ranks[node] == pytest.approx(rank, abs=1e-3)


def test_samples_stay_inside_the_graph(populated):
    nodes = sample_nodes(populated, 3, seed=42)
    assert len(nodes) == 3
    assert all(node in populated for node in nodes)
    edges = sample_edges(populated, 3, seed=42)
    assert len(edges) == 3
    assert all(populated.has_edge(source, target) for source, target, _ in edges)


def test_connected_components_across_a_long_path(empty):
    """A path longer than any label-propagation sweep cap, which union-find settles in one edge pass."""
    if empty.is_directed():
        pytest.skip("The reference graph is undirected")
    length = 2048
    empty.add_edges_from([(node, node + 1) for node in range(length)])
    empty.add_node(10_000)
    labels = connected_components(empty)
    assert labels[length] == 0
    assert labels[0] == 0
    assert labels[10_000] == 10_000
    assert len(set(labels.values())) == 2


def test_pagerank_unweighted(populated, undirected):
    graph = simple(populated)
    ranks = pagerank(graph, iterations=100, tolerance=1e-11, weight=None)
    held = networkx.pagerank(undirected, tol=1e-10, max_iter=1000, weight=None)
    for node, rank in held.items():
        assert ranks[node] == pytest.approx(rank, abs=1e-6)


def test_pagerank_weighted(populated, undirected):
    graph = simple(populated)
    ranks = pagerank(graph, iterations=100, tolerance=1e-11, weight="weight")
    held = networkx.pagerank(undirected, tol=1e-10, max_iter=1000, weight="weight")
    for node, rank in held.items():
        assert ranks[node] == pytest.approx(rank, abs=1e-6)


def test_core_numbers_on_a_denser_graph(empty):
    if empty.is_directed() or empty.is_multigraph():
        pytest.skip("The reference graph is a simple undirected one")
    edges = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6), (4, 3), (6, 1), (7, 1)]
    empty.add_edges_from(edges)
    held = networkx.Graph(edges)
    assert core_numbers(empty) == networkx.core_number(held)
    assert triangle_counts(empty) == networkx.triangles(held)


def test_triangle_counts_of_a_subset(populated, undirected):
    graph = simple(populated)
    assert triangle_counts(graph, [1, 3, 4]) == {
        node: count for node, count in networkx.triangles(without_loops(undirected)).items() if node in {1, 3, 4}
    }
