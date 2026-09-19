"""Clustering, label propagation and k-truss, against NetworkX where it has an answer."""

from __future__ import annotations

import networkx
import pytest

from networkxternal.algorithms import clustering_coefficients, k_truss, label_propagation

CLIQUES = [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (6, 7), (7, 8), (8, 6)]
"""Two triangles joined through one vertex, and a third triangle apart from both."""


@pytest.fixture
def cliques(empty):
    """The `CLIQUES` graph, on the undirected shapes NetworkX answers these questions for."""
    if empty.is_directed() or empty.is_multigraph():
        pytest.skip("The reference answers these for a simple undirected graph")
    empty.add_edges_from(CLIQUES)
    return empty


def test_clustering_matches_networkx(cliques):
    held = networkx.Graph(CLIQUES)
    ours = clustering_coefficients(cliques)
    theirs = networkx.clustering(held)
    assert all(ours[node] == pytest.approx(theirs[node]) for node in theirs)


def test_clustering_of_a_subset_matches(cliques):
    held = networkx.Graph(CLIQUES)
    ours = clustering_coefficients(cliques, [1, 3])
    assert set(ours) == {1, 3}
    assert all(ours[node] == pytest.approx(networkx.clustering(held, node)) for node in ours)


def test_k_truss_matches_networkx(cliques):
    held = networkx.Graph(CLIQUES)
    ours = {(min(source, target), max(source, target)) for source, target, _ in k_truss(cliques, 3)}
    theirs = {(min(source, target), max(source, target)) for source, target in networkx.k_truss(held, 3).edges}
    assert ours == theirs


def test_a_higher_truss_keeps_less(cliques):
    assert len(k_truss(cliques, 4)) <= len(k_truss(cliques, 3))


def test_label_propagation_separates_the_parts(cliques):
    """The triangle apart from the others gets a label of its own, and the answer is stable."""
    labels = label_propagation(cliques)
    grouped: dict[int, set[int]] = {}
    for node, label in labels.items():
        grouped.setdefault(label, set()).add(node)
    assert {6, 7, 8} in grouped.values()
    assert label_propagation(cliques) == labels
