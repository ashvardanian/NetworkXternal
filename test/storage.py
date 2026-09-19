"""The storage verbs against the paths the shared fixture is too small to reach.

Ten edges and a page of a thousand or more mean no backend has ever fetched a second page, walked a
vertex whose degree exceeds one, or been asked about a pair twice. Every one of those is rewritten by
the waves ahead, so each gets an assertion here first.
"""

from __future__ import annotations

import pytest

from networkxternal.base_api import AttributeStore, Role

PAGE = 8
"""The page every graph in this module is narrowed to, so a second page costs nine edges rather than a thousand."""


@pytest.fixture
def narrow(empty):
    """The fixture's graph with its page narrowed, so paging and hubs are reachable in a few edges."""
    type(empty).PAGE = PAGE
    yield empty
    del type(empty).PAGE


def star(graph, centre: int, count: int) -> list[tuple[int, int]]:
    """Attaches `count` leaves to one vertex, which is a degree no fixture has held."""
    pairs = [(centre, centre + index + 1) for index in range(count)]
    graph.add_edges_from_arrays([source for source, _ in pairs], [target for _, target in pairs])
    return pairs


# region Paging


def test_scan_edges_yields_every_edge_once_across_pages(narrow):
    """A graph of more than one page is walked exactly once, which no fixture has ever checked."""
    pairs = star(narrow, 1, PAGE * 3 + 1)
    found = list(narrow.scan_edges())
    assert len(found) == len(pairs)
    assert len({(source, target) for source, target, _ in found}) == len(pairs)


def test_scan_edges_resumes_from_a_cursor(narrow):
    """Resuming after a mid-stream edge yields the suffix, and nothing before it."""
    star(narrow, 1, PAGE * 3 + 1)
    whole = list(narrow.scan_edges())
    middle = whole[len(whole) // 2]
    assert list(narrow.scan_edges(after=middle)) == whole[whole.index(middle) + 1 :]


def test_scan_edges_honours_a_limit(narrow):
    """A limit stops the walk, however many pages the graph holds."""
    star(narrow, 1, PAGE * 3 + 1)
    assert len(list(narrow.scan_edges(limit=PAGE + 3))) == PAGE + 3


def test_adjacent_edges_walks_a_hub_beyond_one_page(narrow):
    """A vertex of more degree than a page reports every incident edge, once."""
    pairs = star(narrow, 1, PAGE * 3 + 1)
    found = [triple for node, triple in narrow.adjacent_edges([1], Role.ANY) if node == 1]
    assert len(found) == len(pairs)
    assert len(set(found)) == len(pairs)


def test_degrees_counts_a_hub_beyond_one_page(narrow):
    """The engine counts the hub's ends, whatever the page is."""
    pairs = star(narrow, 1, PAGE * 3 + 1)
    assert narrow.degrees([1], narrow.outgoing_role)[0] == len(pairs)


def test_removing_a_hub_drops_every_incident_edge(narrow):
    """A vertex of more degree than one write batch leaves nothing of itself behind."""
    type(narrow).WRITE = PAGE
    try:
        star(narrow, 1, PAGE * 3 + 1)
        narrow.remove_nodes_from([1])
        assert not narrow.has_node(1)
        assert all(1 not in (source, target) for source, target, _ in narrow.scan_edges())
    finally:
        del type(narrow).WRITE


# endregion Paging


# region Pairs


def test_find_pairs_answers_a_repeated_pair_twice(populated):
    """A pair given twice is answered twice, once per position that asked for it."""
    found = list(populated.find_pairs([1, 1], [2, 2]))
    assert sorted(position for position, _ in found) == [0, 1]


def test_find_pairs_is_symmetric_on_an_undirected_graph(populated):
    """An undirected pair is answered in either orientation."""
    if populated.is_directed():
        pytest.skip("A directed graph answers a pair only as it was asked")
    forward = [triple for _, triple in populated.find_pairs([1], [2])]
    backward = [triple for _, triple in populated.find_pairs([2], [1])]
    assert forward == backward


def test_find_pairs_skips_a_pair_that_holds_nothing(populated):
    """A pair with no edge yields nothing, and does not shift the positions of the others."""
    found = list(populated.find_pairs([1, 1, 2], [99, 2, 3]))
    assert {position for position, _ in found} == {1, 2}


def test_a_directed_self_loop_counts_once_in_each_direction(populated):
    """A directed self-loop is one out-edge and one in-edge, as NetworkX has it."""
    if not populated.is_directed():
        pytest.skip("An undirected self-loop counts twice in one degree instead")
    assert populated.degrees([3], Role.SOURCE)[0] == populated.out_degree[3]
    assert populated.out_degree[3] == 2
    assert populated.in_degree[3] == 3


def test_an_undirected_self_loop_counts_twice(populated):
    """An undirected self-loop contributes both of its ends to the degree."""
    if populated.is_directed():
        pytest.skip("A directed graph splits the loop across the two roles")
    assert populated.degree[3] == 5


# endregion Pairs


# region Attributes


def test_merge_documents_takes_one_entry_for_every_key(populated):
    """One document merged into many keys reaches all of them."""
    nodes = sorted(populated.nodes)[:3]
    populated.merge_documents(AttributeStore.NODES, nodes, [{"shared": 7}])
    assert [found.get("shared") for found in populated.read_documents(AttributeStore.NODES, nodes)] == [7, 7, 7]


def test_merge_documents_takes_one_entry_per_key(populated):
    """One document per key reaches its own key alone."""
    nodes = sorted(populated.nodes)[:3]
    entries = [{"own": index} for index in range(len(nodes))]
    populated.merge_documents(AttributeStore.NODES, nodes, entries)
    found = populated.read_documents(AttributeStore.NODES, nodes)
    assert [entry.get("own") for entry in found] == [0, 1, 2]


def test_dropping_an_edge_document_clears_its_weight(populated):
    """A weight lifted into a column leaves with the document, so a re-added edge starts clean."""
    edge = next(iter(populated.find_pairs([1], [2])))[1][2]
    populated.drop_documents(AttributeStore.EDGES, [edge])
    assert populated.read_documents(AttributeStore.EDGES, [edge])[0].get("weight") is None


# endregion Attributes
