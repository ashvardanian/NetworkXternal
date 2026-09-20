"""The two orientations of one algorithm answer the same thing, whichever way they drove the store."""

from __future__ import annotations

import pytest

from algorithms import REGISTRY
from algorithms.base import Algorithm
from networkxternal.base_api import Orientation

PAIRED = sorted(name for name, kind in REGISTRY.items() if kind.VARIANTS)
"""Every algorithm offering a choice, which is the only set this file has anything to say about."""


def variants(name: str) -> tuple[type[Algorithm], type[Algorithm]]:
    """The edge-oriented and vertex-oriented classes a paired algorithm chooses between."""
    held = REGISTRY[name].VARIANTS
    return held[Orientation.EDGE], held[Orientation.VERTEX]


def answered(kind, graph):
    """What one variant answers, as something comparable whether it is one map or a pair of them."""
    held = kind.probe(graph).run()
    return tuple(dict(one) for one in held) if isinstance(held, tuple) else dict(held)


@pytest.mark.parametrize("name", PAIRED)
def test_both_orientations_answer_the_same(name, populated):
    """A scatter over the edge stream and a gather per vertex page reach one answer."""
    scattered, gathered = variants(name)
    assert answered(scattered, populated) == answered(gathered, populated)


@pytest.mark.parametrize("name", PAIRED)
def test_the_store_picks_one_of_the_two(name, populated):
    """`on` answers with the variant matching what the store declares, and honours an override."""
    scattered, gathered = variants(name)
    asked = REGISTRY[name].probing(populated)
    assert type(REGISTRY[name].on(populated, **asked)).ORIENTATION is populated.SCAN_ORIENTATION
    assert isinstance(REGISTRY[name].on(populated, orientation=Orientation.EDGE, **asked), scattered)
    assert isinstance(REGISTRY[name].on(populated, orientation=Orientation.VERTEX, **asked), gathered)


@pytest.mark.parametrize("name", PAIRED)
def test_a_self_loop_is_counted_once_by_both(name, populated):
    """The one place the orientations could disagree, since a loop reaches its vertex from both ends."""
    assert any(source == target for source, target, _ in populated.scan_edges()), "the fixture lost its self-loop"
    scattered, gathered = variants(name)
    assert answered(scattered, populated) == answered(gathered, populated)


def test_an_unpaired_algorithm_ignores_what_the_store_prefers():
    """A class declaring no variants is the answer to `on` whatever the store would rather do."""
    from algorithms.core_numbers import CoreNumbers

    assert not CoreNumbers.VARIANTS
