"""Whether one graph instance survives being driven from several threads, which 3.14t makes routine."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

THREADS = 4
"""How many threads share the one graph instance under test."""

EDGES_PER_THREAD = 64
"""How many edges each thread inserts, over a vertex range of its own."""


def insert_range(graph, thread: int) -> None:
    """Inserts one thread's own edges, none of which another thread touches."""
    first = thread * EDGES_PER_THREAD * 2
    sources = [first + index * 2 for index in range(EDGES_PER_THREAD)]
    targets = [first + index * 2 + 1 for index in range(EDGES_PER_THREAD)]
    graph.add_edges_from_arrays(sources, targets)


def test_import_leaves_the_gil_as_it_found_it():
    """Importing the package must not re-enable the GIL on a free-threaded build."""
    before = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    import networkxternal  # noqa: F401

    after = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    assert after == before


def test_concurrent_writers_keep_every_edge(empty):
    """Threads writing disjoint vertex ranges through one instance lose no edge to a race."""
    if not type(empty).CONCURRENT:
        pytest.skip(f"{type(empty).__name__} does not serve several threads from one instance")
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        for outcome in [pool.submit(insert_range, empty, thread) for thread in range(THREADS)]:
            outcome.result()
    assert empty.number_of_edges() == THREADS * EDGES_PER_THREAD


def claim_keyed(graph, thread: int) -> list[int]:
    """Inserts one thread's keyed edges over a pair every thread shares, answering the keys it used."""
    first = 10_000 + thread * EDGES_PER_THREAD
    keys = list(range(first, first + EDGES_PER_THREAD))
    graph.add_edges_from_arrays([1] * len(keys), [2] * len(keys), keys)
    return keys


def test_a_chosen_key_never_collides_with_an_allocated_one(empty):
    """Threads choosing keys and threads letting the graph choose never land on the same identifier."""
    if not type(empty).MULTIGRAPH:
        pytest.skip("Only a multigraph takes edge keys")
    if not type(empty).CONCURRENT:
        pytest.skip(f"{type(empty).__name__} does not serve several threads from one instance")
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        chosen = [
            key for outcome in [pool.submit(claim_keyed, empty, t) for t in range(THREADS)] for key in outcome.result()
        ]
    allocated = empty.add_edges_from_arrays([3] * THREADS, [4] * THREADS, [None] * THREADS)
    assert set(allocated).isdisjoint(chosen)
    assert len(set(allocated)) == THREADS


def test_concurrent_readers_agree(populated):
    """Several threads reading one instance see the same graph."""
    if not type(populated).CONCURRENT:
        pytest.skip(f"{type(populated).__name__} does not serve several threads from one instance")
    expected = populated.number_of_edges()
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        counted = [outcome.result() for outcome in [pool.submit(populated.number_of_edges) for _ in range(THREADS)]]
    assert counted == [expected] * THREADS
