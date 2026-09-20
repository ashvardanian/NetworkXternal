"""The answer types: what they cost, what they promise, and how they compare to a dict."""

from __future__ import annotations

import tracemalloc
from array import array

import pytest

from algorithms.results import DenseIndex, VertexMap, gathered, sparsely

NODES = (3, 11, 40, 97)
"""Sparse identifiers in key order, which is the order every backend scans in."""


@pytest.fixture
def scores() -> VertexMap[float]:
    return VertexMap(DenseIndex(NODES), array("d", [0.5, 1.5, 2.5, 3.5]))


def test_it_reads_as_a_mapping(scores):
    """Everything `Mapping` promises works, so a caller cannot tell it from a dict."""
    assert scores[11] == 1.5
    assert len(scores) == 4
    assert list(scores) == list(NODES)
    assert scores.get(99) is None
    assert set(scores.values()) == {0.5, 1.5, 2.5, 3.5}
    assert dict(scores) == {3: 0.5, 11: 1.5, 40: 2.5, 97: 3.5}


def test_a_missing_vertex_raises(scores):
    with pytest.raises(KeyError):
        scores[99]
    assert 99 not in scores
    assert "not a vertex" not in scores


def test_it_compares_equal_to_a_dict_either_way(scores):
    """The oracle comparisons in the suite read `ours == networkx_answer` and the reverse."""
    held = {3: 0.5, 11: 1.5, 40: 2.5, 97: 3.5}
    assert scores == held
    assert held == scores
    assert scores != {3: 0.5}
    assert scores != {3: 0.5, 11: 1.5, 40: 2.5, 97: 9.9}


def test_it_costs_the_array_and_not_a_dict():
    """The claim is per vertex and counts the boxed keys a dict holds, which `getsizeof` does not."""
    count = 10_000
    tracemalloc.start()
    opened = tracemalloc.get_traced_memory()[0]
    held = VertexMap(DenseIndex(range(count)), array("d", [1.0]) * count)
    ours = tracemalloc.get_traced_memory()[0] - opened
    before_dict = tracemalloc.get_traced_memory()[0]
    plain = held.to_dict()
    theirs = tracemalloc.get_traced_memory()[0] - before_dict
    tracemalloc.stop()
    assert ours / count < 17, f"{ours / count:.1f} bytes a vertex, where the two arrays are sixteen"
    assert ours * 4 < theirs, f"{ours / count:.1f} against the dict's {theirs / count:.1f} bytes a vertex"
    assert len(plain) == count


def test_freeze_detaches_from_a_running_walk(scores):
    """A sweep moves the live array; the frozen copy keeps what it was handed."""
    held = scores.freeze()
    scores.held[1] = 99.0
    assert scores[11] == 99.0
    assert held[11] == 1.5
    assert held.order is scores.order, "freezing copies the values, never the index"


def test_a_subset_carries_an_index_of_its_own():
    """An algorithm asked for some vertices answers over those, in the same type."""
    order = DenseIndex(NODES)
    held = gathered(order, array("q", [1, 2, 3, 4]), [97, 11])
    assert held == {11: 2, 97: 4}
    assert gathered(order, array("q", [1, 2, 3, 4]), None) == dict(zip(NODES, (1, 2, 3, 4), strict=True))


def test_a_sparse_walk_answers_over_what_it_reached():
    """A traversal settles a fraction of the graph and pays for that fraction."""
    held = sparsely({40: 2, 3: 0}, "q")
    assert held == {3: 0, 40: 2}
    assert list(held) == [3, 40], "the answer is in key order however the walk reached it"


def test_the_index_answers_positions_and_membership():
    order = DenseIndex(NODES)
    assert order[40] == 2
    assert 40 in order
    assert 41 not in order
    assert list(order.items()) == [(3, 0), (11, 1), (40, 2), (97, 3)]
