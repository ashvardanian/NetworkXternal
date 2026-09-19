"""Edge identifiers across two writers, which one process's lock cannot make safe on its own."""

from __future__ import annotations

import pytest

from networkxternal.base_api import NetworkXternalError


def test_two_instances_never_hand_out_one_identifier(empty, second):
    """Two graph objects on one store lease disjoint runs of identifiers."""
    if not type(empty).SHARES_IDENTIFIERS:
        pytest.skip(f"{type(empty).__name__} holds no durable counter, so it serves one writer")
    if not type(empty).MULTIGRAPH:
        pytest.skip("A simple graph reuses the identifier of a pair it already holds")
    first = empty.add_edges_from_arrays([1, 1, 1], [2, 2, 2])
    other = second.add_edges_from_arrays([3, 3, 3], [4, 4, 4])
    assert set(first).isdisjoint(other)


def test_two_instances_keep_every_edge(empty, second):
    """Edges written through either object are all there, counted through either object."""
    empty.add_edges_from_arrays([1, 2, 3], [2, 3, 4])
    second.add_edges_from_arrays([5, 6, 7], [6, 7, 8])
    assert empty.count_edges() == 6
    assert second.count_edges() == 6


OTHER_SHAPE = {"Graph": "DiGraph", "DiGraph": "Graph", "MultiGraph": "MultiDiGraph", "MultiDiGraph": "MultiGraph"}
"""The shape a graph is reopened as, to prove the store remembers what it was written as."""


def test_a_graph_refuses_the_shape_it_was_not_written_as(empty, open_as):
    """Opening a stored graph as another shape is refused rather than answered differently."""
    shape, url = empty.opened_as
    if ":memory:" in url or not type(empty).SHARES_STORE:
        pytest.skip(f"{type(empty).__name__} does not address one store from two objects")
    empty.add_edges_from_arrays([1], [2])
    with pytest.raises(NetworkXternalError):
        open_as(OTHER_SHAPE[shape], url).close()
