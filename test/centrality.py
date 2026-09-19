"""Hubs, authorities, a preference vector and betweenness, against NetworkX."""

from __future__ import annotations

import networkx
import pytest

from networkxternal.algorithms import betweenness_centrality, hits, personalized_pagerank

ARCS = [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (6, 1)]
"""A directed graph with a cycle, a tail and a vertex pointing into it."""


@pytest.fixture
def directed(empty):
    """The `ARCS` graph, on the shapes that can hold a direction."""
    if not empty.is_directed():
        pytest.skip("Hubs and authorities are a directed question")
    empty.add_edges_from(ARCS)
    return empty


def test_hits_matches_networkx(directed):
    held = networkx.DiGraph(ARCS)
    hubs, authorities = hits(directed)
    their_hubs, their_authorities = networkx.hits(held, max_iter=500, tol=1e-12)
    assert all(hubs[node] == pytest.approx(their_hubs[node], abs=1e-6) for node in their_hubs)
    assert all(authorities[node] == pytest.approx(their_authorities[node], abs=1e-6) for node in their_authorities)


def test_personalized_pagerank_matches_networkx(populated, undirected):
    if populated.is_directed():
        pytest.skip("The reference graph is undirected")
    ours = personalized_pagerank(populated, {1: 1.0}, weight=None, tolerance=1e-10)
    theirs = networkx.pagerank(undirected, personalization={1: 1.0}, weight=None, tol=1e-10, max_iter=1000)
    assert all(ours[node] == pytest.approx(theirs[node], abs=1e-6) for node in theirs)


def test_a_preference_must_carry_mass(populated):
    with pytest.raises(ValueError):
        personalized_pagerank(populated, {1: 0.0})


def test_betweenness_matches_networkx(populated, undirected):
    if populated.is_directed() or populated.is_multigraph():
        pytest.skip("The reference graph is a simple undirected one")
    ours = betweenness_centrality(populated)
    theirs = networkx.betweenness_centrality(undirected, normalized=False)
    assert all(ours[node] == pytest.approx(theirs[node], abs=1e-9) for node in theirs)


def test_sampling_sources_answers_for_every_vertex(populated):
    """A sampled run still scores every vertex, on fewer sources and so at lower cost."""
    if populated.is_directed() or populated.is_multigraph():
        pytest.skip("The reference graph is a simple undirected one")
    sampled = betweenness_centrality(populated, samples=3, seed=42)
    assert set(sampled) == set(populated.nodes)
    assert all(score >= 0 for score in sampled.values())
