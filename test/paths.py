"""Weighted shortest paths, against the answers NetworkX gives for the same graph in RAM."""

from __future__ import annotations

import networkx
import pytest

from networkxternal.algorithms import delta_stepping_lengths, dijkstra_lengths


def undirected_only(populated):
    """Skips the shapes whose NetworkX counterpart answers differently than the undirected reference."""
    if populated.is_directed():
        pytest.skip("The reference graph is undirected")
    return populated


def test_dijkstra_matches_networkx(populated, undirected):
    graph = undirected_only(populated)
    assert dijkstra_lengths(graph, 1) == networkx.single_source_dijkstra_path_length(undirected, 1)


def test_delta_stepping_matches_dijkstra(populated, undirected):
    graph = undirected_only(populated)
    held = networkx.single_source_dijkstra_path_length(undirected, 1)
    for delta in (0.5, 2.0, 100.0):
        assert delta_stepping_lengths(graph, 1, delta=delta) == held, f"delta {delta}"


def test_unweighted_paths_count_edges(populated, undirected):
    """With no weight named, a path costs its number of edges, as NetworkX has it."""
    graph = undirected_only(populated)
    held = networkx.single_source_shortest_path_length(undirected, 1)
    assert dijkstra_lengths(graph, 1, weight=None) == dict.fromkeys(held, 0.0) | {
        node: float(length) for node, length in held.items()
    }


def test_a_cutoff_stops_the_walk(populated, undirected):
    graph = undirected_only(populated)
    held = networkx.single_source_dijkstra_path_length(undirected, 1, cutoff=9.0)
    assert dijkstra_lengths(graph, 1, cutoff=9.0) == held
    assert delta_stepping_lengths(graph, 1, cutoff=9.0) == held
